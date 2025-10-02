# streamlit_faizan_linkedin_dashboard.py
"""
Attractive Streamlit dashboard for Faizan Ahmad's LinkedIn / Resume
Features:
- Upload or paste profile (PDF/DOCX/TXT/HTML)
- Clean profile card with contact, headline, summary
- Experience timeline (Plotly)
- Top skills chart and wordcloud
- AI-powered suggestions (headline rewrite, 1-paragraph summary, suggested skills) using OpenAI (optional)
- Export parsed CSV/JSON

How to run:
1. pip install -r requirements.txt
   Example: pip install streamlit pandas plotly wordcloud pillow pdfplumber python-docx openai
2. streamlit run streamlit_faizan_linkedin_dashboard.py

Note: To enable AI features, set your OpenAI API key in the sidebar (or set OPENAI_API_KEY env var).
"""

from io import BytesIO
import re
import tempfile
from datetime import datetime
from collections import Counter
import os

import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from PIL import Image

# Optional libraries
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx
except Exception:
    docx = None
try:
    import openai
except Exception:
    openai = None

st.set_page_config(page_title="Faizan — Profile Dashboard", layout="wide")

# ----------------- Styling -----------------
st.markdown("""
<style>
.main .block-container{padding:1rem 2rem}
.profile-card{background:linear-gradient(90deg,#0ea5a4,#7c3aed);color:white;padding:18px;border-radius:12px}
.small-muted{color:rgba(255,255,255,0.85);font-size:12px}
.kpi{background:#f8fafc;padding:12px;border-radius:8px}
</style>
""", unsafe_allow_html=True)

# ----------------- Helpers -----------------
MONTH_MAP = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12}


def parse_date(text):
    if not text or pd.isna(text):
        return None
    t = str(text).strip()
    if re.search(r'present|current', t, flags=re.I):
        return 'Present'
    m = re.search(r'([A-Za-z]{3,9})\s+(\d{4})', t)
    if m:
        mon = m.group(1)[:3].lower()
        if mon in MONTH_MAP:
            return datetime(int(m.group(2)), MONTH_MAP[mon], 1)
    m2 = re.search(r'(\d{4})', t)
    if m2:
        return datetime(int(m2.group(1)), 1, 1)
    try:
        return pd.to_datetime(t, errors='coerce')
    except:
        return None


def extract_text_from_pdf_bytes(b):
    if not pdfplumber:
        return ""
    text = ""
    with pdfplumber.open(BytesIO(b)) as pdf:
        for p in pdf.pages:
            text += (p.extract_text() or "") + "\n"
    return text


def extract_text_from_docx_bytes(b):
    if not docx:
        return ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmp.write(b); tmp.flush(); tmp.close()
    d = docx.Document(tmp.name)
    return "\n".join([p.text for p in d.paragraphs])


def parse_experience_from_text(text):
    """Parse experience lines heuristically from profile text.
    Returns a list of dicts with keys: title, company, start, end, raw
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    exps = []
    for i, ln in enumerate(lines):
        if re.search(r"\d{4}", ln) or re.search(r'present|current', ln, flags=re.I):
            context = ln
            prev = lines[i-1] if i>0 else ""
            if prev and not re.search(r"\d{4}", prev):
                context = prev + ' | ' + ln
            date_part = re.search(r'([A-Za-z]{3,9}\s*\d{4}|\d{4}|Present|present|Current)(\s*[-–—]\s*([A-Za-z]{3,9}\s*\d{4}|\d{4}|Present|present|Current))?', context)
            start, end = None, None
            if date_part:
                piece = date_part.group(0)
                parts = re.split(r'[-–—]', piece)
                if len(parts) >= 2:
                    start = parse_date(parts[0].strip())
                    end = parse_date(parts[1].strip())
                else:
                    start = parse_date(parts[0].strip())
            title, company = "", ""
            # Match patterns like "Senior Dev, ACME · Jan 2020 – Present"
            m = re.match(r"(.+?)[,@·\|]\s*([A-Za-z0-9 &\-\.\'\"]+)", context)
            if m:
                title = m.group(1).strip()
                company = m.group(2).strip()
            else:
                # fallback 'Title at Company'
                if ' at ' in context.lower():
                    parts = re.split(r' at ', context, flags=re.I)
                    title = parts[0].strip()
                    company = parts[1].split('|')[0].strip()
                else:
                    if prev and not re.search(r"\d", prev):
                        title = prev
            exps.append({'title': title, 'company': company, 'start': start, 'end': end, 'raw': context})
    df = pd.DataFrame(exps)
    if df.empty:
        return []
    df = df.drop_duplicates(subset=['raw'])
    return df.to_dict('records')


def compute_duration_months(start, end):
    if not start:
        return None
    if start == 'Present' or end == 'Present':
        return None
    s = start if isinstance(start, datetime) else (start.to_pydatetime() if hasattr(start,'to_pydatetime') else None)
    e = end if isinstance(end, datetime) else None
    if not s:
        return None
    if not e:
        e = datetime.now()
    return (e.year - s.year) * 12 + (e.month - s.month)


def top_keywords(text, n=30):
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    stop = set(['the','and','for','with','that','this','from','have','has','are','was','were','will','can','but','not','you','your','our','we','in','on','of','to','a','an','as','by','is'])
    toks = [t for t in tokens if t not in stop]
    c = Counter(toks)
    return c.most_common(n)


# ----------------- UI -----------------

