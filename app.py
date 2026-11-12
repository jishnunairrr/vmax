"""V-max: chat + document Q&A (RAG) server.

Run:  python -m uvicorn app:app --port 8000
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
INDEX_FILE = DATA_DIR / "index.json"
ENV_FILE = ROOT / ".env"

# The key is read from NVIDIA_API_KEY in the .env file (not uploaded to GitHub).
# Leave this empty; a value here would override .env and be shared with the code.
API_KEY = ""

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
CHUNK_SIZE = 1100          # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 6                  # chunks sent to the model per question
EMBED_BATCH = 32
MAX_TOOL_ROUNDS = 3        # tool calls the model may chain per question
MAX_WEB_RESULTS = 5
SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".csv"}

DEFAULT_CHAT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3.5-embed-1b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

CHAT_SYSTEM = (
    "You are V-max, a helpful, precise assistant. "
    "Answer clearly and concisely. Use Markdown when it helps readability."
)
RAG_SYSTEM = (
    "You are V-max, a helpful, precise assistant. "
    "The user has uploaded documents. Numbered excerpts from them are provided "
    "with each question. Base your answer on those excerpts and cite them inline "
    "with their number in square brackets, like [1] or [2][4]. "
    "If the excerpts do not contain the answer, say so plainly, then answer from "
    "general knowledge only if you can, marking it as not from the documents. "
    "Never invent citations. Use Markdown when it helps readability."
)
TOOLS_NOTE = (
    " Today's date is {today}. You have two tools: get_weather gives the current "
    "weather and a short forecast for a place, and web_search finds news, prices, "
    "scores and anything recent or likely to have changed. Use them whenever a "
    "question needs live or up-to-date information instead of saying you cannot "
    "access it; do not use them for things you already know. Web results arrive "
    "numbered; cite the ones you use inline with that number in square brackets, "
    "like [3]."
)


# ---------------------------------------------------------------- settings --
def settings() -> dict:
    """API_KEY above wins; a NVIDIA_API_KEY in .env is used if it is empty."""
    load_dotenv(ENV_FILE, override=True)
    return {
        "key": (API_KEY or os.getenv("NVIDIA_API_KEY", "")).strip(),
        "base": os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        "chat_model": os.getenv("NVIDIA_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip(),
        "embed_model": os.getenv("NVIDIA_EMBED_MODEL", DEFAULT_EMBED_MODEL).strip(),
    }


def require_key() -> dict:
    cfg = settings()
    if not cfg["key"] or cfg["key"].startswith("nvapi-your"):
        raise HTTPException(
            400,
            "No API key found. Add NVIDIA_API_KEY to the .env file in the project folder.",
        )
    return cfg


def nvidia_error(status: int, body: str) -> str:
    detail = ""
    try:
        data = json.loads(body)
        detail = data.get("detail") or data.get("message") or ""
        if isinstance(data.get("error"), dict):
            detail = data["error"].get("message", detail)
        elif isinstance(data.get("error"), str):
            detail = data["error"]
    except Exception:
        detail = body[:300]
    if status in (401, 403):
        return "The API key was rejected. Check NVIDIA_API_KEY in the .env file."
    if status == 404:
        return f"The model could not be found. {detail}".strip()
    if status == 429:
        return "Rate limit reached. Wait a moment and send again."
    return f"Model service error {status}. {detail}".strip()


# ------------------------------------------------------------------- store --
class Store:
    """Documents, their text chunks and embeddings. Persisted under data/."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.vectors: dict[str, np.ndarray] = {}
        self.load()

    def load(self) -> None:
        if not INDEX_FILE.exists():
            return
        try:
            for doc in json.loads(INDEX_FILE.read_text(encoding="utf-8")):
                vec_path = DATA_DIR / f"{doc['id']}.npy"
                if vec_path.exists():
                    self.docs[doc["id"]] = doc
                    self.vectors[doc["id"]] = np.load(vec_path)
        except Exception as exc:  # corrupt index: start clean rather than crash
            print(f"Could not load saved documents: {exc}")

    def save_index(self) -> None:
        INDEX_FILE.write_text(json.dumps(list(self.docs.values())), encoding="utf-8")

    def add(self, doc: dict, vectors: np.ndarray) -> None:
        self.docs[doc["id"]] = doc
        self.vectors[doc["id"]] = vectors
        np.save(DATA_DIR / f"{doc['id']}.npy", vectors)
        self.save_index()

    def remove(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)
        self.vectors.pop(doc_id, None)
        (DATA_DIR / f"{doc_id}.npy").unlink(missing_ok=True)
        self.save_index()

    @staticmethod
    def public(doc: dict) -> dict:
        return {k: doc[k] for k in ("id", "name", "pages", "chunk_count", "size", "added")}

    def owned(self, owner: str) -> list[dict]:
        """Documents uploaded by this visitor, oldest first."""
        docs = [d for d in self.docs.values() if d.get("owner") == owner]
        return sorted(docs, key=lambda d: d["added"])

    def summary(self, owner: str) -> list[dict]:
        return [self.public(d) for d in self.owned(owner)]

    def search(self, query_vec: np.ndarray, k: int, embed_model: str, owner: str) -> list[dict]:
        hits = []
        q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
        for doc in self.owned(owner):
            doc_id = doc["id"]
            vecs = self.vectors[doc_id]
            if doc.get("embed_model") != embed_model or vecs.shape[1] != q.shape[0]:
                continue  # embedded with a different model; not comparable
            scores = vecs @ q
            for i in np.argsort(-scores)[:k]:
                chunk = doc["chunks"][int(i)]
                hits.append(
                    {
                        "doc_id": doc_id,
                        "doc": doc["name"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                        "score": float(scores[int(i)]),
                    }
                )
        hits.sort(key=lambda h: -h["score"])
        return hits[:k]


store = Store()


# -------------------------------------------------------------- extraction --
def extract_pages(name: str, raw: bytes) -> list[tuple[int | None, str]]:
    """Return [(page_number or None, text)]."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                reader.decrypt("")
            return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
        except Exception as exc:
            raise HTTPException(400, f"Could not read this PDF ({exc}).")
    if ext == ".docx":
        from docx import Document

        try:
            doc = Document(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(400, f"Could not read this Word file ({exc}).")
        parts = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return [(None, "\n".join(parts))]
    for enc in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return [(None, raw.decode(enc))]
        except UnicodeError:
            continue
    return [(None, raw.decode("utf-8", errors="ignore"))]


def split_text(text: str) -> list[str]:
    """Split into ~CHUNK_SIZE character chunks on sentence boundaries, with overlap."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > CHUNK_SIZE:  # a single very long "sentence"
            if current:
                chunks.append(current)
                current = ""
            chunks.append(piece[:CHUNK_SIZE])
            piece = piece[CHUNK_SIZE - CHUNK_OVERLAP:]
        if current and len(current) + len(piece) + 1 > CHUNK_SIZE:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:].split(" ", 1)[-1] + " " + piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


# --------------------------------------------------------------- embedding --
async def embed(texts: list[str], input_type: str, cfg: dict) -> np.ndarray:
    """input_type is 'passage' for document chunks and 'query' for questions."""
    out: list[list[float]] = []
    headers = {"Authorization": f"Bearer {cfg['key']}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        for start in range(0, len(texts), EMBED_BATCH):
            payload = {
                "model": cfg["embed_model"],
                "input": texts[start:start + EMBED_BATCH],
                "input_type": input_type,
                "encoding_format": "float",
                "truncate": "END",
            }
            try:
                res = await client.post(f"{cfg['base']}/embeddings", json=payload, headers=headers)
            except httpx.HTTPError as exc:
                raise HTTPException(502, f"Could not reach the model service ({exc.__class__.__name__}).")
            if res.status_code != 200:
                raise HTTPException(502, nvidia_error(res.status_code, res.text))
            rows = sorted(res.json()["data"], key=lambda r: r["index"])
            out.extend(r["embedding"] for r in rows)
    vecs = np.asarray(out, dtype=np.float32)
    return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)


# ------------------------------------------------------------------- tools --
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather and a 3-day forecast for a city or place.",
            "parameters": {
                "type": "object",
                "properties": {"place": {"type": "string", "description": "City or place name, e.g. 'Kochi' or 'Paris, France'"}},
                "required": ["place"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current or recent information: news, prices, scores, events, facts that may have changed.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A short, specific search query"}},
                "required": ["query"],
            },
        },
    },
]

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


async def get_weather(place: str) -> str:
    """Live weather from Open-Meteo (free, no key)."""
    place = place.strip()
    async with httpx.AsyncClient(timeout=20) as client:
        spot = None
        for name in dict.fromkeys([place, place.split(",")[0].strip()]):
            res = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search", params={"name": name, "count": 1}
            )
            found = res.json().get("results") if res.status_code == 200 else None
            if found:
                spot = found[0]
                break
        if not spot:
            return f"No place called '{place}' was found. Ask the user to check the spelling."
        res = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": spot["latitude"],
                "longitude": spot["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        if res.status_code != 200:
            return "The weather service did not respond. Try again shortly."
        data = res.json()
    now, daily = data["current"], data["daily"]
    where = ", ".join(x for x in (spot.get("name"), spot.get("admin1"), spot.get("country")) if x)
    lines = [
        f"Weather for {where} (local time {now['time'].replace('T', ' ')}, source: Open-Meteo)",
        f"Now: {WEATHER_CODES.get(now['weather_code'], 'unknown')}, {now['temperature_2m']} C "
        f"(feels like {now['apparent_temperature']} C), humidity {now['relative_humidity_2m']}%, "
        f"wind {now['wind_speed_10m']} km/h, precipitation {now['precipitation']} mm",
    ]
    for i, day in enumerate(daily["time"]):
        lines.append(
            f"{day}: {WEATHER_CODES.get(daily['weather_code'][i], 'unknown')}, "
            f"{daily['temperature_2m_min'][i]} to {daily['temperature_2m_max'][i]} C, "
            f"rain chance {daily['precipitation_probability_max'][i]}%"
        )
    return "\n".join(lines)


def _search_sync(query: str) -> list[dict]:
    from ddgs import DDGS

    for backend in ("yahoo", "bing", "google", "auto"):  # each engine fails often; try in turn
        try:
            rows = DDGS(timeout=12).text(query, max_results=MAX_WEB_RESULTS, backend=backend)
        except Exception:
            continue
        results = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "text": r.get("body", "")}
            for r in rows or []
            if r.get("href")
        ]
        if results:
            return results
    return []


async def web_search(query: str) -> list[dict]:
    return await asyncio.to_thread(_search_sync, query.strip())


# --------------------------------------------------------------------- api --
app = FastAPI(title="V-max")

VISITOR_COOKIE = "vmax_visitor"


@app.middleware("http")
async def visitor_cookie(request: Request, call_next):
    """Give each browser a random id so its documents stay private to it."""
    visitor = request.cookies.get(VISITOR_COOKIE, "")
    is_new = not re.fullmatch(r"[0-9a-f]{32}", visitor)
    if is_new:
        visitor = secrets.token_hex(16)
    request.state.visitor = visitor
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            VISITOR_COOKIE, visitor, max_age=365 * 24 * 3600, httponly=True, samesite="lax"
        )
    return response


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    use_documents: bool = True


@app.get("/api/status")
def status():
    cfg = settings()
    return {
        "key_configured": bool(cfg["key"]) and not cfg["key"].startswith("nvapi-your"),
        "chat_model": cfg["chat_model"],
        "embed_model": cfg["embed_model"],
    }


@app.get("/api/documents")
def list_documents(request: Request):
    return {"documents": store.summary(request.state.visitor)}


@app.post("/api/documents")
async def upload_document(request: Request, file: UploadFile = File(...)):
    name = Path(file.filename or "document").name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED:
        raise HTTPException(400, f"{ext or 'This file type'} is not supported. Upload a PDF, DOCX, TXT, MD or CSV file.")
    cfg = require_key()
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "This file is larger than 25 MB.")
    if not raw:
        raise HTTPException(400, "This file is empty.")

    pages = extract_pages(name, raw)
    chunks = [
        {"page": page_no, "text": chunk}
        for page_no, text in pages
        for chunk in split_text(text)
    ]
    if not chunks:
        raise HTTPException(
            400,
            "No readable text was found. If this is a scanned PDF, run OCR on it first.",
        )

    vectors = await embed([c["text"] for c in chunks], "passage", cfg)
    doc = {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "pages": len(pages) if ext == ".pdf" else None,
        "chunk_count": len(chunks),
        "size": len(raw),
        "added": time.time(),
        "embed_model": cfg["embed_model"],
        "owner": request.state.visitor,
        "chunks": chunks,
    }
    store.add(doc, vectors)
    return store.public(doc)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str, request: Request):
    if store.docs.get(doc_id, {}).get("owner") != request.state.visitor:
        raise HTTPException(404, "Document not found.")
    store.remove(doc_id)
    return {"ok": True}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    cfg = require_key()
    history = [m for m in req.messages if m.role in ("user", "assistant") and m.content.strip()]
    if not history or history[-1].role != "user":
        raise HTTPException(400, "Send a question to get an answer.")
    question = history[-1].content
    model = (req.model or cfg["chat_model"]).strip()
    visitor = request.state.visitor

    async def stream():
        sources: list[dict] = []
        if req.use_documents and store.owned(visitor):
            try:
                query_vec = (await embed([question], "query", cfg))[0]
                sources = store.search(query_vec, TOP_K, cfg["embed_model"], visitor)
            except HTTPException as exc:
                yield sse({"type": "error", "message": exc.detail})
                return

        messages = [{"role": "system", "content": RAG_SYSTEM if sources else CHAT_SYSTEM}]
        messages += [{"role": m.role, "content": m.content} for m in history[:-1]]
        if sources:
            for n, s in enumerate(sources, 1):
                s["n"] = n
            excerpts = "\n\n".join(
                f"[{s['n']}] {s['doc']}" + (f", page {s['page']}" if s["page"] else "") + f"\n{s['text']}"
                for s in sources
            )
            yield sse({"type": "sources", "sources": sources})
            messages.append(
                {"role": "user", "content": f"Document excerpts:\n\n{excerpts}\n\nQuestion: {question}"}
            )
        else:
            messages.append({"role": "user", "content": question})

        today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
        messages[0]["content"] += TOOLS_NOTE.format(today=today)
        headers = {"Authorization": f"Bearer {cfg['key']}", "Accept": "text/event-stream"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=20)) as client:
                for round_no in range(MAX_TOOL_ROUNDS + 1):
                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": 0.2 if sources else 0.6,
                        "top_p": 0.95,
                        "max_tokens": 4096,
                        "stream": True,
                    }
                    if round_no < MAX_TOOL_ROUNDS:  # the last round must answer
                        payload["tools"] = TOOLS
                        payload["tool_choice"] = "auto"
                    calls: dict[int, dict] = {}
                    said = ""
                    async with client.stream(
                        "POST", f"{cfg['base']}/chat/completions", json=payload, headers=headers
                    ) as res:
                        if res.status_code != 200:
                            body = (await res.aread()).decode("utf-8", errors="ignore")
                            yield sse({"type": "error", "message": nvidia_error(res.status_code, body)})
                            return
                        async for line in res.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                choices = json.loads(data).get("choices") or []
                            except json.JSONDecodeError:
                                continue
                            if not choices:
                                continue
                            delta = choices[0].get("delta") or {}
                            for part in delta.get("tool_calls") or []:
                                call = calls.setdefault(part.get("index", 0), {"id": "", "name": "", "args": ""})
                                fn = part.get("function") or {}
                                call["id"] = part.get("id") or call["id"]
                                call["name"] += fn.get("name") or ""
                                call["args"] += fn.get("arguments") or ""
                            if delta.get("content"):
                                said += delta["content"]
                                yield sse({"type": "token", "text": delta["content"]})
                            elif delta.get("reasoning_content") or delta.get("reasoning"):
                                yield sse({"type": "thinking"})
                    if not calls:
                        break

                    ordered = [calls[i] for i in sorted(calls)]
                    messages.append(
                        {
                            "role": "assistant",
                            "content": said,
                            "tool_calls": [
                                {
                                    "id": c["id"] or f"call_{round_no}_{i}",
                                    "type": "function",
                                    "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                                }
                                for i, c in enumerate(ordered)
                            ],
                        }
                    )
                    for i, c in enumerate(ordered):
                        try:
                            args = json.loads(c["args"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        try:
                            if c["name"] == "get_weather":
                                place = str(args.get("place", ""))
                                yield sse({"type": "status", "text": f"Checking the weather in {place}"})
                                result = await get_weather(place)
                            elif c["name"] == "web_search":
                                query = str(args.get("query", ""))
                                yield sse({"type": "status", "text": f"Searching the web for {query}"})
                                found = await web_search(query)
                                for hit in found:
                                    sources.append(
                                        {"n": len(sources) + 1, "doc": hit["title"] or hit["url"],
                                         "page": None, "text": hit["text"], "url": hit["url"]}
                                    )
                                    hit["n"] = len(sources)
                                if found:
                                    yield sse({"type": "sources", "sources": sources})
                                    result = "\n\n".join(
                                        f"[{h['n']}] {h['title']} ({h['url']})\n{h['text']}" for h in found
                                    )
                                else:
                                    result = (
                                        "The search returned no results. Tell the user the search failed and "
                                        "answer from what you know, saying it may be out of date."
                                    )
                            else:
                                result = f"There is no tool called {c['name']}."
                        except Exception as exc:
                            result = f"The tool failed ({exc.__class__.__name__}). Tell the user live data is unavailable right now."
                        messages.append(
                            {"role": "tool", "tool_call_id": c["id"] or f"call_{round_no}_{i}", "content": result}
                        )
        except httpx.HTTPError as exc:
            yield sse({"type": "error", "message": f"Lost the connection to the model service ({exc.__class__.__name__}). Send again."})
            return
        yield sse({"type": "done"})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
