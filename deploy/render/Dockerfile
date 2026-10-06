# Render (free web service, Docker runtime): FastAPI backend + Streamlit frontend in one container.
# uvicorn listens only inside the container (127.0.0.1:8000); Streamlit listens on $PORT, which Render sets.
# Build context is the repo root; .dockerignore keeps data, photos and docs out, and only app files are copied.
FROM python:3.11-slim

RUN useradd -m -u 1000 app
USER app
ENV HOME=/home/app PATH=/home/app/.local/bin:$PATH PYTHONUNBUFFERED=1 API_URL=http://127.0.0.1:8000 PORT=10000
WORKDIR /home/app/merchmix

COPY --chown=app deploy/render/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=app config.py .
COPY --chown=app .streamlit .streamlit
COPY --chown=app backend backend
COPY --chown=app frontend frontend
COPY --chown=app outputs outputs

EXPOSE 10000
CMD uvicorn backend.api:app --host 127.0.0.1 --port 8000 & \
    exec streamlit run frontend/app.py --server.address 0.0.0.0 --server.port "$PORT" --server.headless true
