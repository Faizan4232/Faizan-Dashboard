# Faizan-Dashboard.py
"""
Faizan Dashboard - Streamlit single-file (no plotly)
Features:
- Accepts uploaded LinkedIn/resume PDF/DOCX/TXT/HTML or uses /mnt/data/Profile LinkedIN.pdf if present
- Parses Experience, Skills (heuristics)
- Shows experience table + Gantt-like timeline (matplotlib)
- Shows top skills (bar chart) and wordcloud (if installed)
- Export CSV/JSON buttons
- Optional OpenAI suggestions (enter key in sidebar)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from collections import Counter
import re
import os

# Optional parsing libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import docx
except Exception:
    docx = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Optional OpenAI (only used if key provided)
try:
    import openai
except Exception:
    openai = None

st.set_page_config(page_title="Faizan - LinkedIn Dashboard", layout="wide")

# ---------------- Helpers ----------------
MONTHS = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'sept':9,'oct':10,'nov':11,'dec':12
}

def parse_date(text):
    """Return a datetime if possible, or 'Present' or None."""
    if not text or pd.isna(text):
        return None
    t = str(text).strip()
    if re.search(r'present|current', t, flags=re.I):
        return 'Present'
    # Try Month Year like "Jan 2020"
    m = re.search(r'([A-Za-z]{3,9})\s+(\d{4})', t)
    if m:
        mon = m.group(1)[:3].lower()
        if mon in MONTHS:
            return datetime(int(m.group(2)), MONTHS[mon], 1)
    # Try year only
    m2 = re.search(r'(\d{4})', t)
    if m2:
        try:
            return datetime(int(m2.group(1)), 1, 1)
        except:
            return None
    # fallback to pandas parser
    try:
        dt = pd.to_datetime(t, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.to_pydatetime()
    except:
        return None

def extract_text_from_pdf_bytes(b):
    if not pdfplumber:
        return ""
    text = ""
    with pdfplumber.open(BytesIO(b)) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
    return text

def extract_text_from_docx_bytes(b):
    if not docx:
        return ""
    tmp = BytesIO(b)
    # python-docx cannot read BytesIO directly, write temp file
    import tempfile
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".docx")
    tmpf.write(b); tmpf.flush(); tmpf.close()
    d = docx.Document(tmpf.name)
    texts = [p.text for p in d.paragraphs]
    return "\n".join(texts)

def parse_experience_from_text(text):
    """
    Heuristic parsing: find lines with years or 'Present' and try to extract title/company and dates.
    Returns list of dicts: title, company, start (datetime or 'Present' or None), end
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    exps = []
    for i, ln in enumerate(lines):
        if re.search(r'\d{4}', ln) or re.search(r'present|current', ln, flags=re.I):
            context = ln
            prev = lines[i-1] if i>0 else ""
            if prev and not re.search(r'\d{4}', prev):
                # include previous non-date line as context (likely title/company)
                context = prev + " | " + ln
            # find date substring
            date_part = re.search(r'([A-Za-z]{3,9}\s*\d{4}|\d{4}|Present|present|Current)(\s*[-–—]\s*([A-Za-z]{3,9}\s*\d{4}|\d{4}|Present|present|Current))?', context)
            start = end = None
            if date_part:
                piece = date_part.group(0)
                parts = re.split(r'[-–—]', piece)
                if len(parts) >= 2:
                    start = parse_date(parts[0].strip())
                    end = parse_date(parts[1].strip())
                else:
                    start = parse_date(parts[0].strip())
            # Try extracting title and company
            title = company = ""
            m = re.match(r'(.+?)[,@·\|]\s*([A-Za-z0-9 &\-\.\']+)', context)
            if m:
                title = m.group(1).strip()
                company = m.group(2).strip()
            else:
                if ' at ' in context.lower():
                    parts = re.split(r' at ', context, flags=re.I)
                    title = parts[0].strip()
                    company = parts[1].split('|')[0].strip()
                else:
                    # fallback to previous line as title/company
                    if prev and not re.search(r'\d', prev):
                        title = prev
            exps.append({'title': title, 'company': company, 'start': start, 'end': end, 'raw': context})
    # deduplicate
    if not exps:
        return []
    df = pd.DataFrame(exps).drop_duplicates(subset=['raw'])
    return df.to_dict('records')

def compute_duration_months(start, end):
    if not start:
        return None
    if start == 'Present' or end == 'Present':
        return None
    s = start if isinstance(start, datetime) else None
    e = None
    if end == 'Present' or end is None:
        e = datetime.now()
    else:
        e = end if isinstance(end, datetime) else None
    if not s or not e:
        return None
    return (e.year - s.year) * 12 + (e.month - s.month)

def top_keywords(text, n=25):
    tokens = re.findall(r"[A-Za-z]{3,}", text.lower())
    stop = set(['the','and','for','with','that','this','from','have','has','are','was','were','will','can','but','not','you','your','our','we','in','on','of','to','a','an','as','by','is'])
    toks = [t for t in tokens if t not in stop]
    c = Counter(toks)
    return c.most_common(n)

# ---------------- UI ----------------
st.title("Faizan — LinkedIn Resume Dashboard")

st.sidebar.header("Input / Settings")
mode = st.sidebar.selectbox("Input mode", ["Load packaged PDF (if present)", "Upload file (PDF/DOCX/TXT/HTML)", "Paste profile text"])
openai_key = st.sidebar.text_input("Optional OpenAI API Key (for suggestions)", type="password")
if openai_key and openai:
    openai.api_key = openai_key

profile_text = ""

if mode == "Load packaged PDF (if present)":
    default_path = "/mnt/data/Profile LinkedIN.pdf"
    if os.path.exists(default_path):
        st.sidebar.success(f"Found packaged PDF at {default_path}")
        with open(default_path, "rb") as f:
            b = f.read()
            if pdfplumber:
                profile_text = extract_text_from_pdf_bytes(b)
            else:
                st.sidebar.warning("pdfplumber not installed — text extraction may fail. Install pdfplumber and redeploy.")
                try:
                    profile_text = b.decode('utf-8', errors='ignore')
                except:
                    profile_text = ""
    else:
        st.sidebar.error("No packaged PDF found at /mnt/data/Profile LinkedIN.pdf")
elif mode == "Upload file (PDF/DOCX/TXT/HTML)":
    uploaded = st.sidebar.file_uploader("Upload resume/profile file", type=["pdf","docx","txt","html"])
    if uploaded:
        data = uploaded.read()
        if uploaded.name.lower().endswith(".pdf"):
            if pdfplumber:
                profile_text = extract_text_from_pdf_bytes(data)
            else:
                st.sidebar.warning("pdfplumber not installed — extracted raw bytes as fallback.")
                try:
                    profile_text = data.decode('utf-8', errors='ignore')
                except:
                    profile_text = ""
        elif uploaded.name.lower().endswith(".docx"):
            if docx:
                profile_text = extract_text_from_docx_bytes(data)
            else:
                st.sidebar.warning("python-docx not installed — cannot parse DOCX on server.")
        else:
            try:
                profile_text = data.decode('utf-8', errors='ignore')
            except:
                profile_text = str(data)
else:
    profile_text = st.sidebar.text_area("Paste About / Experience / Skills text", height=200)

if not profile_text or len(profile_text.strip()) < 20:
    st.info("Paste or upload your profile text (About + Experience + Skills) or use the packaged PDF option.")
    st.stop()

# Show raw preview
with st.expander("Raw extracted text (first 2000 chars)"):
    st.text_area("Raw preview", profile_text[:2000], height=300)

# Parse experience and keywords
experiences = parse_experience_from_text(profile_text)
if experiences:
    df_exp = pd.DataFrame(experiences)
    df_exp['duration_months'] = df_exp.apply(lambda r: compute_duration_months(r['start'], r['end']), axis=1)
else:
    df_exp = pd.DataFrame(columns=['title','company','start','end','duration_months','raw'])

st.subheader("Experience")
if not df_exp.empty:
    st.dataframe(df_exp[['title','company','start','end','duration_months']])
else:
    st.info("No experience entries parsed. Try uploading a PDF export or paste the Experience section lines explicitly.")

# Timeline plot using matplotlib
if not df_exp.empty:
    # Prepare timeline rows that have start dates
    tdf = df_exp.copy()
    tdf = tdf[tdf['start'].notnull()].reset_index(drop=True)
    if not tdf.empty:
        # ensure start/end are datetimes
        def norm_date(v):
            if v == 'Present' or v is None:
                return datetime.now()
            if isinstance(v, datetime):
                return v
            try:
                return pd.to_datetime(v)
            except:
                return datetime.now()
        tdf['start_dt'] = tdf['start'].apply(norm_date)
        tdf['end_dt'] = tdf['end'].apply(lambda v: norm_date(v) if v != 'Present' else datetime.now())
        # sort by start
        tdf = tdf.sort_values('start_dt')
        y = np.arange(len(tdf))
        fig, ax = plt.subplots(figsize=(10, 0.7*len(tdf)+1))
        for i, row in tdf.iterrows():
            start_num = mdates.date2num(row['start_dt'])
            end_num = mdates.date2num(row['end_dt'])
            ax.barh(y[i], end_num - start_num, left=start_num, height=0.6, align='center')
        ax.set_yticks(y)
        labels = [f"{r['title']} — {r['company']}" if r['company'] else r['title'] for _, r in tdf.iterrows()]
        ax.set_yticklabels(labels)
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.tight_layout()
        st.subheader("Experience Timeline")
        st.pyplot(fig)
    else:
        st.info("No valid start dates to draw timeline.")
        
# Top keywords / skills
st.subheader("Top keywords / inferred skills")
kw = top_keywords(profile_text, n=30)
if kw:
    kdf = pd.DataFrame(kw, columns=['keyword','count'])
    st.dataframe(kdf.head(25))
    # simple bar chart
    st.bar_chart(kdf.set_index('keyword')['count'])
    # wordcloud if available
    if WordCloud:
        wc = WordCloud(width=800, height=300).generate(" ".join([w for w,c in kw for _ in range(min(3,c))]))
        st.image(wc.to_array(), use_column_width=True)
    else:
        st.info("wordcloud package not installed. Add 'wordcloud' to requirements if you want wordclouds.")
else:
    st.info("No keywords found.")

# AI suggestions (optional)
if openai and openai_key:
    st.subheader("AI Suggestions")
    # Attempt to create a short punchy headline from first 300 chars of text
    prompt = "Create a concise, attention-grabbing LinkedIn headline (max 12 words) based on this profile text:\n\n" + profile_text[:1200]
    try:
        resp = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=60, temperature=0.7)
        headline = resp.choices[0].text.strip()
        st.markdown("**Suggested headline:**")
        st.write(headline)
    except Exception as e:
        st.error(f"OpenAI call failed: {e}")
elif openai_key and not openai:
    st.info("OpenAI library isn't installed in this environment; suggestions won't run.")

# Export parsed data
st.subheader("Export parsed profile data")
if not df_exp.empty:
    csv_bytes = df_exp.to_csv(index=False).encode('utf-8')
    st.download_button("Download parsed CSV", csv_bytes, file_name="parsed_profile.csv", mime="text/csv")
    st.download_button("Download parsed JSON", df_exp.to_json(orient='records').encode('utf-8'), file_name="parsed_profile.json")

st.write("")
st.caption("If you still get `ModuleNotFoundError` on deploy, ensure your `requirements.txt` in GitHub includes the libraries you need (streamlit,pandas,matplotlib,...).")
