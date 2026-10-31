# 📄 ATS Resume Analyzer

Upload your resume (PDF, DOCX or TXT) and get an **estimated ATS score**, a section-by-section
breakdown, missing keywords and concrete improvements. Optionally paste a job description to
check how well your resume matches that specific role.

Built with **Streamlit** and **Google Gemini Flash**.

## Features
- Resume parsing for PDF, DOCX (including tables) and TXT
- AI analysis with Gemini: score, strengths, prioritised fixes, rewrite suggestions
- Keyword matching against an optional job description
- Rule-based format checks (email, phone, sections, bullets, metrics, length)
- Headline score = 80% AI score + 20% format checks
- Downloadable report (.md)

> ⚠️ The score is an **estimate**. Real ATS systems (Workday, Greenhouse, Lever...) do not
> publish a universal score. Treat it as guidance, not a guarantee.

## Project structure
```
.
├── app.py               # Streamlit app
├── requirements.txt     # Python dependencies
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

## Run locally
```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini API key (free key: https://aistudio.google.com/apikey)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then edit secrets.toml and paste your key

# 4. Run
streamlit run app.py
```
You can also skip step 3 and paste the key into the sidebar of the running app.

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | `.streamlit/secrets.toml`, environment variable, or sidebar | required |
| `GEMINI_MODEL` | same as above | `gemini-3.5-flash` |

If you get a "model not found" error, Google has probably renamed or retired the model.
Check the current list at https://ai.google.dev/gemini-api/docs/models and set `GEMINI_MODEL`
(for example `gemini-flash-latest`).

## Push to GitHub
1. Create an empty repository on https://github.com/new (do **not** add a README there).
2. In your project folder run:
```bash
git init
git add .
git commit -m "Initial commit: ATS resume analyzer"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
3. Make sure `.streamlit/secrets.toml` is **not** in the repo (it is listed in `.gitignore`).
   Never commit your API key.

## Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Select your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:
```toml
GEMINI_API_KEY = "your-gemini-api-key-here"
```
5. Click **Deploy**. After a minute or two you will get a public `*.streamlit.app` URL.

To update the app later, just `git push`; Streamlit redeploys automatically.

## Privacy
Resume text is sent to Google's Gemini API for analysis and is not stored by this app.
Review Google's API data terms before using real personal data in a public deployment.

## Troubleshooting
- **"Very little text could be extracted"**: the PDF is probably a scanned image. Export a
  text-based PDF or use DOCX.
- **Quota / 429 error**: free-tier rate limit reached; wait a minute and retry.
- **Invalid API key**: regenerate a key in Google AI Studio and update your secrets.# cv-generator
