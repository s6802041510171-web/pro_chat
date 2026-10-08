# Thesis Guide Chatbot

A Thai-language chatbot that answers questions using the thesis preparation and formatting guide in `data/Dataset.md`.

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `TYPHOON_API_KEY` in `.env`.
3. Start the app with `python app.py`.
4. Open `http://localhost:10000` (or open `public/index.html`; the file page connects to the local API at port 10000).

## Deploy on Vercel

Import this repository as a Vercel project and add `TYPHOON_API_KEY` under the project's Environment Variables for the environments you use. Redeploy after saving the variable. The frontend and FastAPI backend use the same domain, and the API key stays server-side.

Vercel serves the files in `public/` as static assets and detects the FastAPI application exported by `app.py`.
