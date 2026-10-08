# Thesis Guide Chatbot

A Thai-language thesis assistant that answers from `data/Dataset.md` and includes document checks, academic writing help, citation review, research alignment review, and a request draft wizard.

## Features

- Ask questions about the thesis guide.
- Upload DOCX or text-based PDF files (up to 4.4 MB) for preliminary A4, margin, font, page-number, approval-page, five-chapter structure, and citation consistency checks.
- Improve Thai or English academic prose, review title/objective/hypothesis/statistics alignment, and format citation lists in APA 7 or IEEE style.
- Create a step-by-step Word request draft for common thesis milestones. The generated file is a draft, not an official university form; official form templates can be added when supplied.
- Connect a Google account to review an existing Google Doc and save academic writing results to a new Google Doc.

Automated checks are indicative and should be confirmed against the current graduate school requirements. For AI review, extracted document text is sent to Typhoon; uploaded files are not retained by this app.

The upload cap is 4.4 MB because Vercel Functions accept request bodies up to 4.5 MB total, including multipart form overhead. Larger uploads require direct-to-storage uploads (for example, Vercel Blob) rather than sending the file through the app function.

## Run locally

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set `TYPHOON_API_KEY` in `.env`.
3. For Google Docs, create a Google OAuth 2.0 Web application client ID, enable the Google Docs API, add the app's exact origin (for local use, `http://localhost:10000`) to Authorized JavaScript origins, and set `GOOGLE_CLIENT_ID` in `.env`. Configure the Google OAuth consent screen for your users.
4. Start the app with `python app.py`.
5. Open `http://localhost:10000` (or open `public/index.html`; the file page connects to the local API at port 10000).

Google Docs uses OAuth scopes `documents.readonly` and `drive.file`. The user approves access in Google; the app keeps the access token in page memory only. Document text is sent to Typhoon when AI review is requested. Add the deployed site's exact HTTPS origin to the OAuth client as well. Google may require OAuth consent-screen verification before general public use.

## Deploy on Vercel

Import this repository as a Vercel project and add `TYPHOON_API_KEY` and `GOOGLE_CLIENT_ID` under the project's Environment Variables for the environments you use. Add the Vercel site's exact origin to the OAuth client's Authorized JavaScript origins, then redeploy after saving the variables. The frontend and FastAPI backend use the same domain, and the API keys stay server-side (Google Client ID is public configuration).

Vercel serves the files in `public/` as static assets and detects the FastAPI application exported by `app.py`.
