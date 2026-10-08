# Thesis Guide Chatbot

A Thai-language thesis assistant that answers from `data/Dataset.md` and includes document checks, academic writing help, citation review, research alignment review, and a request draft wizard.

## Features

- Ask questions about the thesis guide.
- Upload DOCX or text-based PDF files (up to 4 MB) for preliminary A4, margin, font, page-number, approval-page, five-chapter structure, and citation consistency checks.
- Improve Thai or English academic prose, review title/objective/hypothesis/statistics alignment, and format citation lists in APA 7 or IEEE style.
- Create a step-by-step Word request draft for common thesis milestones. The generated file is a draft, not an official university form; official form templates can be added when supplied.

Automated checks are indicative and should be confirmed against the current graduate school requirements. For AI review, extracted document text is sent to Typhoon; uploaded files are not retained by this app.

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `TYPHOON_API_KEY` in `.env`.
3. Start the app with `python app.py`.
4. Open `http://localhost:10000` (or open `public/index.html`; the file page connects to the local API at port 10000).

## Deploy on Vercel

Import this repository as a Vercel project and add `TYPHOON_API_KEY` under the project's Environment Variables for the environments you use. Redeploy after saving the variable. The frontend and FastAPI backend use the same domain, and the API key stays server-side.

Vercel serves the files in `public/` as static assets and detects the FastAPI application exported by `app.py`.
