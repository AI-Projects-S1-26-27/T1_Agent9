# Deploying the demo

Nothing here is deployed yet. The repo is ready for either route below.

## A. Streamlit dashboard (Streamlit Community Cloud)
1. Push the repo (or your fork) to GitHub.
2. On share.streamlit.io choose "New app", pick the repo and branch, and set the main file to `streamlit_app.py`.
3. Under Advanced settings pick Python 3.11, 3.12 or 3.13. Dependencies come from `requirements.txt`.
4. Deploy. The app reads the saved results in `experiments/results/`, so it does not run the long tournaments.

Local check: `streamlit run streamlit_app.py`

## B. Game-style web demo (Render, Fly.io, Hugging Face Spaces)
It is a small Python server, so it needs a container host. The `Dockerfile` builds it and listens on `$PORT`.
- Local check: `docker build -t resource-war . && docker run -p 8765:8765 resource-war`, then open http://127.0.0.1:8765
- Render: New > Web Service > connect the repo > Runtime: Docker. No other settings needed.
- Without Docker: `python -m resource_war.demo.web --host 0.0.0.0 --port $PORT --no-browser`

By default the server binds to 127.0.0.1 only. `--host 0.0.0.0` is for hosts you control; the demo has no login, so share the URL only as intended.
