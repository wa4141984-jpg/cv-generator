"""
ATS Resume Analyzer
-------------------
Upload a resume (PDF / DOCX / TXT) and get:
  * an estimated ATS score (0-100)
  * a section-by-section breakdown
  * keyword gaps (optionally against a job description)
  * concrete improvements and rewrite suggestions

UI: Streamlit  |  AI: Google Gemini Flash (google-genai SDK)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
from typing import Any

import streamlit as st
from docx import Document
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "gemini-3.5-flash"  # override with GEMINI_MODEL (secret / env / sidebar)
MAX_FILE_MB = 5
MAX_RESUME_CHARS = 15_000
MAX_JD_CHARS = 6_000
MIN_TEXT_CHARS = 150  # below this we assume a scanned / image-only resume

SCORE_KEYS = {
    "keyword_match": "Keywords & relevance",
    "formatting": "Formatting & ATS-parsability",
    "content_quality": "Content quality & impact",
    "structure": "Sections & structure",
    "readability": "Readability & language",
}

SYSTEM_PROMPT = """You are an expert technical recruiter and ATS (Applicant Tracking \
System) specialist. You evaluate resumes strictly and honestly. You never invent \
facts that are not in the resume. You respond with a single valid JSON object and \
nothing else."""


# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from PDF, DOCX or TXT bytes."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ValueError("This PDF is password-protected.") from exc
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages).strip()

    if name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:  # many resumes put content in tables
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        parts.append(cell.text.strip())
        return "\n".join(parts).strip()

    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore").strip()

    raise ValueError("Unsupported file type. Please upload a PDF, DOCX or TXT file.")


# --------------------------------------------------------------------------- #
# Deterministic (non-AI) checks
# --------------------------------------------------------------------------- #
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")
LINK_RE = re.compile(r"(linkedin\.com|github\.com|gitlab\.com|behance\.net|portfolio)", re.I)
METRIC_RE = re.compile(r"\d+(?:[.,]\d+)?\s?(?:%|\+|x|k|m|million|billion)|[$€£]\s?\d", re.I)
BULLET_RE = re.compile(r"^\s*[•\-\u2022\u25CF\u25AA*▪►]\s+", re.M)

SECTION_PATTERNS = {
    "Experience": r"\b(work experience|professional experience|experience|employment)\b",
    "Education": r"\b(education|academic|qualifications)\b",
    "Skills": r"\b(skills|technical skills|core competencies|technologies)\b",
    "Summary": r"\b(summary|profile|objective|about me)\b",
    "Projects": r"\b(projects|personal projects)\b",
    "Certifications": r"\b(certifications?|licenses?|courses)\b",
}


def heuristic_checks(text: str) -> dict[str, Any]:
    """Cheap, explainable checks that complement the AI analysis."""
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    lowered = text.lower()

    sections_found = [
        name for name, pat in SECTION_PATTERNS.items() if re.search(pat, lowered)
    ]
    has_email = bool(EMAIL_RE.search(text))
    has_phone = bool(PHONE_RE.search(text))
    has_links = bool(LINK_RE.search(text))
    bullets = len(BULLET_RE.findall(text))
    metrics = len(METRIC_RE.findall(text))

    checks = [
        ("Email address present", has_email, 15),
        ("Phone number present", has_phone, 10),
        ("LinkedIn / GitHub / portfolio link", has_links, 5),
        ("Experience section detected", "Experience" in sections_found, 15),
        ("Education section detected", "Education" in sections_found, 10),
        ("Skills section detected", "Skills" in sections_found, 15),
        ("Bullet points used", bullets >= 5, 10),
        ("Quantified achievements (numbers, %, $)", metrics >= 3, 10),
        ("Length is reasonable (250-1000 words)", 250 <= word_count <= 1000, 10),
    ]
    earned = sum(points for _, ok, points in checks if ok)
    total = sum(points for _, _, points in checks)
    return {
        "score": round(100 * earned / total),
        "checks": [{"label": label, "passed": ok} for label, ok, _ in checks],
        "word_count": word_count,
        "sections_found": sections_found,
        "bullets": bullets,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
def build_prompt(resume_text: str, job_description: str) -> str:
    jd_block = (
        f"JOB DESCRIPTION:\n\"\"\"\n{job_description[:MAX_JD_CHARS]}\n\"\"\"\n"
        if job_description.strip()
        else "JOB DESCRIPTION: none provided. Evaluate as a general, well-formed "
        "resume and infer the most likely target role from the content.\n"
    )
    return f"""Analyze the resume below as an ATS would, then as a recruiter would.

RESUME:
\"\"\"
{resume_text[:MAX_RESUME_CHARS]}
\"\"\"

{jd_block}
Return ONE JSON object with exactly this shape (all scores are integers 0-100):
{{
  "overall_score": 0,
  "target_role": "short string",
  "summary": "2-3 sentence honest verdict",
  "scores": {{
    "keyword_match": 0,
    "formatting": 0,
    "content_quality": 0,
    "structure": 0,
    "readability": 0
  }},
  "matched_keywords": ["keywords/skills present in the resume that matter for the role"],
  "missing_keywords": ["important keywords/skills absent from the resume"],
  "strengths": ["specific strengths, max 5"],
  "improvements": [
    {{
      "priority": "High | Medium | Low",
      "area": "e.g. Summary, Experience, Skills, Formatting",
      "issue": "what is wrong",
      "fix": "exactly how to fix it"
    }}
  ],
  "rewrite_suggestions": [
    {{
      "original": "a weak line copied from the resume",
      "improved": "a stronger version using action verbs and metrics WITHOUT inventing facts - use placeholders like [X%] where a number is unknown"
    }}
  ],
  "ats_red_flags": ["formatting or content problems that can break ATS parsing"]
}}

Rules:
- Be strict: most resumes score between 40 and 80. Do not inflate.
- Give 5-8 improvements ordered by priority and 3-5 rewrite suggestions.
- Only quote "original" lines that really appear in the resume.
- If a job description is provided, keyword_match must reflect overlap with it.
- Output raw JSON only: no markdown fences, no commentary."""


def parse_llm_json(raw: str) -> dict[str, Any]:
    """Parse model output into a dict, tolerating code fences / stray text."""
    if not raw or not raw.strip():
        raise ValueError("The model returned an empty response.")
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("The model response did not contain JSON.") from None
        return json.loads(cleaned[start : end + 1])


def _clamp(value: Any, default: int = 0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def _str_list(value: Any, limit: int = 15) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def normalize_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Validate / clamp the model output so the UI never crashes on odd data."""
    scores_in = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    scores = {key: _clamp(scores_in.get(key)) for key in SCORE_KEYS}

    overall = data.get("overall_score")
    overall = _clamp(overall, default=round(sum(scores.values()) / len(scores)))

    improvements = []
    for item in data.get("improvements") or []:
        if isinstance(item, dict) and (item.get("issue") or item.get("fix")):
            priority = str(item.get("priority", "Medium")).strip().title()
            improvements.append(
                {
                    "priority": priority if priority in {"High", "Medium", "Low"} else "Medium",
                    "area": str(item.get("area", "General")),
                    "issue": str(item.get("issue", "")),
                    "fix": str(item.get("fix", "")),
                }
            )
    order = {"High": 0, "Medium": 1, "Low": 2}
    improvements.sort(key=lambda i: order[i["priority"]])

    rewrites = []
    for item in data.get("rewrite_suggestions") or []:
        if isinstance(item, dict) and item.get("original") and item.get("improved"):
            rewrites.append(
                {"original": str(item["original"]), "improved": str(item["improved"])}
            )

    return {
        "overall_score": overall,
        "target_role": str(data.get("target_role") or "Not detected"),
        "summary": str(data.get("summary") or ""),
        "scores": scores,
        "matched_keywords": _str_list(data.get("matched_keywords")),
        "missing_keywords": _str_list(data.get("missing_keywords")),
        "strengths": _str_list(data.get("strengths"), 8),
        "improvements": improvements,
        "rewrite_suggestions": rewrites,
        "ats_red_flags": _str_list(data.get("ats_red_flags"), 10),
    }


def analyze_resume(
    resume_text: str, job_description: str, api_key: str, model: str
) -> dict[str, Any]:
    """Call Gemini and return a normalized analysis dict."""
    from google import genai  # imported lazily so the UI loads even if SDK is missing
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(resume_text, job_description)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
    )

    last_error: Exception | None = None
    for _ in range(2):  # one retry if the JSON comes back malformed
        response = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
        try:
            return normalize_analysis(parse_llm_json(response.text))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Could not read the AI response ({last_error}). Please try again.")


def combine_scores(ai_score: int, format_score: int) -> int:
    """Headline score = 80% AI judgement + 20% deterministic format checks."""
    return round(0.8 * ai_score + 0.2 * format_score)


# --------------------------------------------------------------------------- #
# Report export
# --------------------------------------------------------------------------- #
def build_report(final_score: int, analysis: dict[str, Any], local: dict[str, Any]) -> str:
    lines = [
        "# ATS Resume Report",
        f"\n**ATS score: {final_score}/100**  (AI: {analysis['overall_score']}, "
        f"format checks: {local['score']})",
        f"\n**Target role:** {analysis['target_role']}",
        f"\n{analysis['summary']}\n",
        "## Score breakdown",
    ]
    lines += [f"- {SCORE_KEYS[k]}: {v}/100" for k, v in analysis["scores"].items()]
    lines += ["\n## Strengths"] + [f"- {s}" for s in analysis["strengths"]]
    lines += ["\n## Improvements"]
    lines += [
        f"- **[{i['priority']}] {i['area']}** - {i['issue']} -> {i['fix']}"
        for i in analysis["improvements"]
    ]
    lines += ["\n## Missing keywords", ", ".join(analysis["missing_keywords"]) or "None"]
    lines += ["\n## Rewrite suggestions"]
    for r in analysis["rewrite_suggestions"]:
        lines += [f"- Before: {r['original']}", f"  After: {r['improved']}"]
    if analysis["ats_red_flags"]:
        lines += ["\n## ATS red flags"] + [f"- {f}" for f in analysis["ats_red_flags"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # noqa: BLE001  (no secrets.toml present)
        pass
    return os.environ.get(name, default)


def score_color(score: int) -> str:
    return "🟢" if score >= 75 else "🟡" if score >= 55 else "🔴"


def render_results(analysis: dict[str, Any], local: dict[str, Any]) -> None:
    final = combine_scores(analysis["overall_score"], local["score"])

    top1, top2, top3 = st.columns(3)
    top1.metric(f"{score_color(final)} ATS Score", f"{final}/100")
    top2.metric("AI content score", f"{analysis['overall_score']}/100")
    top3.metric("Format checks", f"{local['score']}/100")
    st.progress(final / 100)
    st.markdown(f"**Detected target role:** {analysis['target_role']}")
    if analysis["summary"]:
        st.info(analysis["summary"])

    tab_over, tab_fix, tab_kw, tab_rewrite, tab_checks = st.tabs(
        ["📊 Overview", "🛠️ Improvements", "🔑 Keywords", "✍️ Rewrites", "✅ Format checks"]
    )

    with tab_over:
        st.subheader("Score breakdown")
        for key, label in SCORE_KEYS.items():
            val = analysis["scores"][key]
            st.write(f"{score_color(val)} **{label}** - {val}/100")
            st.progress(val / 100)
        if analysis["strengths"]:
            st.subheader("Strengths")
            for s in analysis["strengths"]:
                st.markdown(f"- {s}")
        if analysis["ats_red_flags"]:
            st.subheader("ATS red flags")
            for f in analysis["ats_red_flags"]:
                st.warning(f)

    with tab_fix:
        if not analysis["improvements"]:
            st.write("No improvements returned.")
        icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        for item in analysis["improvements"]:
            with st.expander(
                f"{icons[item['priority']]} {item['priority']} · {item['area']}",
                expanded=item["priority"] == "High",
            ):
                st.markdown(f"**Issue:** {item['issue']}")
                st.markdown(f"**Fix:** {item['fix']}")

    with tab_kw:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matched")
            st.write(", ".join(analysis["matched_keywords"]) or "None found")
        with c2:
            st.subheader("❌ Missing")
            st.write(", ".join(analysis["missing_keywords"]) or "None found")
        st.caption("Only add keywords you can honestly back up with experience.")

    with tab_rewrite:
        if not analysis["rewrite_suggestions"]:
            st.write("No rewrite suggestions returned.")
        for r in analysis["rewrite_suggestions"]:
            st.markdown(f"**Before:** {r['original']}")
            st.success(f"**After:** {r['improved']}")
        st.caption("Replace [X] placeholders with your real numbers.")

    with tab_checks:
        for c in local["checks"]:
            st.write(("✅ " if c["passed"] else "❌ ") + c["label"])
        st.caption(
            f"{local['word_count']} words · {local['bullets']} bullet points · "
            f"{local['metrics']} quantified items · sections: "
            f"{', '.join(local['sections_found']) or 'none detected'}"
        )

    st.download_button(
        "⬇️ Download report (.md)",
        data=build_report(final, analysis, local),
        file_name="ats_resume_report.md",
        mime="text/markdown",
    )


def main() -> None:
    st.set_page_config(page_title="ATS Resume Analyzer", page_icon="📄", layout="wide")
    st.title("📄 ATS Resume Analyzer")
    st.caption("Upload your resume and get an estimated ATS score with actionable fixes.")

    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = get_secret("GEMINI_API_KEY")
        if api_key:
            st.success("Gemini API key loaded from secrets.")
        else:
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Get a free key at https://aistudio.google.com/apikey",
            )
        model = st.text_input("Gemini model", value=get_secret("GEMINI_MODEL", DEFAULT_MODEL))
        st.divider()
        st.caption(
            "⚠️ The score is an *estimate*. Real ATS products (Workday, Greenhouse, "
            "Lever...) do not publish a universal score."
        )
        st.caption("🔒 Your resume is sent to Google's Gemini API for analysis.")

    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
    with right:
        job_description = st.text_area(
            "Job description (optional but recommended)",
            height=180,
            placeholder="Paste the job posting here to get keyword matching for that role...",
        )

    if st.button("🚀 Analyze resume", type="primary", disabled=uploaded is None):
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            st.stop()
        data = uploaded.getvalue()
        if len(data) > MAX_FILE_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_FILE_MB} MB.")
            st.stop()
        try:
            text = extract_text(uploaded.name, data)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the file: {exc}")
            st.stop()
        if len(text) < MIN_TEXT_CHARS:
            st.error(
                "Very little text could be extracted. If your resume is a scanned image "
                "or heavily designed PDF, an ATS will struggle too - export a text-based "
                "PDF or DOCX and try again."
            )
            st.stop()

        with st.spinner("Analyzing your resume with Gemini..."):
            try:
                analysis = analyze_resume(text, job_description, api_key, model.strip())
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                if "API key" in msg or "API_KEY" in msg or "403" in msg or "401" in msg:
                    st.error("Gemini rejected the API key. Please check it.")
                elif "404" in msg or "not found" in msg.lower():
                    st.error(f"Model '{model}' was not found. Change it in the sidebar.")
                elif "429" in msg or "quota" in msg.lower():
                    st.error("Rate limit / quota reached. Wait a minute and try again.")
                else:
                    st.error(f"Analysis failed: {msg}")
                st.stop()

        st.session_state["result"] = {
            "analysis": analysis,
            "local": heuristic_checks(text),
            "file_id": hashlib.md5(data).hexdigest(),
        }

    # Results live in session_state so they survive reruns (e.g. clicking Download).
    if "result" in st.session_state:
        st.divider()
        render_results(st.session_state["result"]["analysis"], st.session_state["result"]["local"])


if __name__ == "__main__":
    main()
