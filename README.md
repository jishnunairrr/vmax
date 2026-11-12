# V-max

A chat assistant with document Q&A. Ask anything, or upload a PDF, Word, text, Markdown or CSV file and get answers drawn from it, with citations to the page each fact came from.

It runs on NVIDIA's hosted API: a Nemotron model for chat and a Nemotron embedding model for document search.

## Run it

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and paste your API key from [build.nvidia.com](https://build.nvidia.com) after `NVIDIA_API_KEY=`.

```bash
.venv\Scripts\python.exe -m uvicorn app:app --port 8000
```

Open http://localhost:8000.

On macOS or Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

## How it works

- `app.py` is the server. An uploaded file is split into passages, each passage is embedded, and the vectors are saved under `data/`. A question is embedded the same way, the closest passages are sent to the chat model along with the question, and the answer streams back.
- `static/` is the interface.

## Settings

Set these in `.env` to override the defaults:

| Name | Default |
| --- | --- |
| `NVIDIA_CHAT_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b` |
| `NVIDIA_EMBED_MODEL` | `nvidia/nemotron-3.5-embed-1b` |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` |

Scanned PDFs without a text layer are not supported; run OCR on them first.
