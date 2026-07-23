# 07 - FastAPI Inference Server

# Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt

uvicorn main:app --reload
```

Open the automatically generated Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# Goal

Expose the inference engine built in Project 06 as a REST API using FastAPI.

This project demonstrates:

- REST endpoints
- Request validation with Pydantic
- JSON responses
- Token streaming
- Automatic API documentation (Swagger UI)

---

# Concepts

- FastAPI
- Uvicorn (ASGI server)
- Pydantic
- Request & Response Models
- StreamingResponse
- REST API
- Swagger UI
- Dependency Injection (intro)

---

# Project Structure

```text
07_fastapi_inference_server/
│
├── main.py         # FastAPI routes
├── engine.py       # Inference engine (from Project 06)
├── schemas.py      # Pydantic request/response models
├── README.md
└── requirements.txt
```

### main.py

Creates the FastAPI application and exposes REST endpoints.

```text
POST /generate
POST /stream
POST /benchmark
```

### engine.py

Contains the reusable inference engine implemented in Project 06.

Responsibilities:

- Tokenization
- Greedy decoding
- Streaming generation
- KV cache optimization
- Benchmarking

### schemas.py

Defines request and response schemas using Pydantic.

---

# API Endpoints

## GET /

Health check.

Returns

```json
{
  "status": "healthy"
}
```

---

## POST /generate

Input

```json
{
    "prompt": "Why is the sky blue?"
}
```

Output

```json
{
    "response": "The sky appears blue because..."
}
```

---

## POST /stream

Input

```json
{
    "prompt": "Why is the sky blue?"
}
```

Streams tokens back to the client as they are generated.

---

## POST /benchmark

Benchmarks

- Vanilla streaming
- KV-cached streaming

Reports

- TTFT (Time To First Token)
- Total generation time

---

# How to Run

Start the server

```bash
uvicorn main:app --reload
```

or

```bash
uvicorn main:app --reload --port 8001
```

Open Swagger

```text
http://127.0.0.1:8000/docs
```

(or `8001` if using another port)

Try the endpoints directly from the browser.

You can also test with curl.

Example

```bash
curl -X POST http://127.0.0.1:8000/generate \
-H "Content-Type: application/json" \
-d '{"prompt":"Why is the sky blue?"}'
```

---

# Notes

## `response_model` vs `Response`

Use `response_model` only when FastAPI should serialize Python objects into JSON.

| Return Type | response_model |
|--------------|---------------|
| dict | ✅ |
| BaseModel | ✅ |
| list | ✅ |
| StreamingResponse | ❌ |
| FileResponse | ❌ |
| HTMLResponse | ❌ |
| PlainTextResponse | ❌ |

`StreamingResponse` is already a complete HTTP response object, so FastAPI bypasses Pydantic serialization and streams bytes directly to the client.

---

## Why Uvicorn?

FastAPI is the web framework.

Uvicorn is the ASGI server that actually runs the application and serves HTTP requests.

```text
    Request ⬇️       Response ⬆️

👤 Client
    ⇅
🚀 Uvicorn (ASGI)
    ⇅
⚡ FastAPI
    ⇅
🧠 Inference Engine
    ⇅
🤗 Hugging Face Model
```

Run the application with

```bash
uvicorn main:app --reload
```

where

- `main` → `main.py`
- `app` → `app = FastAPI()`
- `--reload` → automatically reloads after code changes

---

# What I Learned

- Wrapped an inference engine behind REST APIs.
- Used FastAPI routing with Pydantic request validation.
- Returned structured JSON responses.
- Streamed generated tokens using `StreamingResponse`.
- Exposed interactive API documentation through Swagger.
- Benchmarked inference through an API endpoint.
- Understood the roles of FastAPI, Uvicorn, and Pydantic in a production inference service.

---

# Next Project

**08_vllm**

Replace the custom Hugging Face inference engine with vLLM to introduce production-grade inference, continuous batching, and higher throughput while keeping the API interface largely unchanged.