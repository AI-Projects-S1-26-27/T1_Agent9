# Game-style web demo (resource_war.demo.web). Works on Render, Fly.io, Hugging Face Spaces (Docker) or any container host.
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY config ./config
COPY experiments ./experiments
COPY report ./report
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
ENV PORT=8765
EXPOSE 8765
CMD ["sh", "-c", "python -m resource_war.demo.web --host 0.0.0.0 --port ${PORT} --no-browser"]
