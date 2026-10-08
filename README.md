# Thesis Guide Chatbot

A Thai-language chatbot that answers questions using the thesis preparation and formatting guide in `data/Dataset.md`.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `TYPHOON_API_KEY` in `.env`.
3. Start the API with `python app.py`.
4. Open `index.html` in a browser. The page expects the API at `http://localhost:10000`.

The API retrieves relevant sections from the Markdown guide and sends those excerpts to Typhoon with the user's question. The API key belongs in `.env`, which is ignored by Git.
