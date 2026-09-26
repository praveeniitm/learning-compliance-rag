FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt -e .
COPY data ./data
COPY app ./app
# Index, memory, cache and audit logs live in volumes so they survive restarts.
VOLUME ["/app/indexes", "/app/.cache", "/app/logs"]
EXPOSE 7860
# OPENAI_API_KEY (or LLM_BASE_URL/EMBED_BASE_URL for Ollama/vLLM) is passed at runtime, never baked in.
CMD ["python", "-c", "import uvicorn, sys; sys.path.insert(0, 'app'); from app import app; uvicorn.run(app, host='0.0.0.0', port=7860)"]
