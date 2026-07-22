"""
Project 07

FastAPI Inference Server

Agenda:
1. Build a REST API around the inference engine.
2. Return complete responses.
3. Stream responses token-by-token.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from engine import InferenceEngine
from schemas import (
    GenerateRequest,
    StreamRequest,
    GenerateResponse,
)


# (1) Config
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

# (2) Load Engine
engine = InferenceEngine(
    model_name=MODEL_NAME,
)



# (3) FastAPI App
app = FastAPI(
    title="Inference Server",
)

# (4) Health Check
@app.get("/")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME,
    }



# (5) Generate
@app.post("/generate",
    response_model=GenerateResponse)
def generate(
    request: GenerateRequest,
):

    response = engine.generate(
        prompt=request.prompt,
    )
    return GenerateResponse(
        response=response,
    )

# (6) Stream
@app.post("/stream")
def stream(
    request: StreamRequest,
):

    def token_generator():

        for token in engine.stream_generate_kv_cache(
            prompt=request.prompt,
        ):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
    )



# (7) Benchmark
@app.post("/benchmark")
def benchmark(
    request: GenerateRequest,
):

    engine.benchmark_streaming(
        prompt=request.prompt,
    )

    return {
        "status": "completed",
    }