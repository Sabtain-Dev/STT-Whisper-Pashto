import streamlit as st
import torch
import librosa
import os
import sqlite3
import hashlib
import csv
import io
import requests
import json
from datetime import datetime
from transformers import pipeline as hf_pipeline
from difflib import SequenceMatcher
import sys
from pathlib import Path

# Add project root to Python path so it can find 'utils'
ROOT_DIR = Path(__file__).parent.parent  # Goes up from app/ to root
sys.path.append(str(ROOT_DIR))

# Optional: Print to verify (you can remove later)
print(f"Project root added to path: {ROOT_DIR}")

# Import decoupled core inference components cleanly from your package modules
from utils.inference import PashtoTranscriber

try:
    from jiwer import wer as jiwer_wer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

# ── Config & File Path Workspace Layout ───────────────────────────────────────
MERGED_MODEL_PATH = "Sabtain-Dev/STT-Whisper-Pashto" 
FP16_ENABLED = os.environ.get("FP16", "False").lower() == "true"

# Centralized Local App Storage Parameters
DRIVE_BASE  = "./workspace_data" # Configured for standard execution portability
DB_PATH     = os.path.join(DRIVE_BASE, "pashto_app.db")
AUDIO_STORE = os.path.join(DRIVE_BASE, "audio_uploads")
TEMP_DIR    = os.path.join("app", "temp")

os.makedirs(AUDIO_STORE, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

SUPPORTED_FORMATS = ["wav", "mp3", "mp4", "m4a", "flac", "ogg", "opus", "webm", "aac", "wma"]

NLLB_MODEL   = "facebook/nllb-200-distilled-600M"
PASHTO_CODE  = "pbt_Arab"
ENGLISH_CODE = "eng_Latn"
URDU_CODE    = "urd_Arab"

# ── AI Config ─────────────────────────────────────────────────────────────────
HF_TOKEN         = os.environ.get("HF_TOKEN", "")
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/"

# ── Database ──────────────────────────────────────────────────────────────────
def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "username TEXT UNIQUE NOT NULL,"
        "password_hash TEXT NOT NULL,"
        "email TEXT,"
        "created_at TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS transcriptions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "user_id INTEGER NOT NULL,"
        "filename TEXT NOT NULL,"
        "original_transcription TEXT,"
        "edited_transcription TEXT,"
        "reference_text TEXT,"
        "wer_score REAL,"
        "translation_en TEXT,"
        "translation_ur TEXT,"
        "audio_filepath TEXT,"
        "created_at TEXT,"
        "FOREIGN KEY(user_id) REFERENCES users(id))"
    )
    conn.commit()
    existing_columns = {row[1] for row in c.execute("PRAGMA table_info(transcriptions)")}
    migrations = {
        "translation_en":       "ALTER TABLE transcriptions ADD COLUMN translation_en TEXT",
        "translation_ur":       "ALTER TABLE transcriptions ADD COLUMN translation_ur TEXT",
        "wer_score":            "ALTER TABLE transcriptions ADD COLUMN wer_score REAL",
        "reference_text":       "ALTER TABLE transcriptions ADD COLUMN reference_text TEXT",
        "edited_transcription": "ALTER TABLE transcriptions ADD COLUMN edited_transcription TEXT",
        "audio_filepath":       "ALTER TABLE transcriptions ADD COLUMN audio_filepath TEXT",
    }
    for col, sql in migrations.items():
        if col not in existing_columns:
            c.execute(sql)
    conn.commit()
    conn.close()

init_db()

def _hash(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def create_user(username, password, email=""):
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, email, created_at) VALUES (?, ?, ?, ?)",
            (username, _hash(password), email, datetime.now().isoformat())
        )
        conn.commit()
        return True, "Account created! Please log in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."
    finally:
        conn.close()

def verify_user(username, password):
    conn = get_db()
    row = conn.execute(
        "SELECT id, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if row and row[1] == _hash(password):
        return True, row[0]
    return False, None

def save_transcription(user_id, filename, original, audio_filepath=None, reference=None, wer_score=None):
    conn = get_db()
    c = conn.cursor()
    existing = c.execute(
        "SELECT id FROM transcriptions "
        "WHERE user_id=? AND filename=? AND original_transcription=? AND (audio_filepath=? OR audio_filepath IS NULL)",
        (user_id, filename, original, audio_filepath)
    ).fetchone()
    if existing:
        conn.close()
        return existing[0]
    c.execute(
        "INSERT INTO transcriptions "
        "(user_id, filename, original_transcription, edited_transcription, "
        "reference_text, wer_score, translation_en, translation_ur, audio_filepath, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, filename, original, original, reference, wer_score,
         None, None, audio_filepath, datetime.now().isoformat())
    )
    conn.commit()
    tid = c.lastrowid
    conn.close()
    return tid

def update_transcription(tid, edited, reference=None, wer_score=None):
    conn = get_db()
    conn.execute(
        "UPDATE transcriptions SET edited_transcription=?, reference_text=?, wer_score=? WHERE id=?",
        (edited, reference, wer_score, tid)
    )
    conn.commit()
    conn.close()

def delete_transcription(tid, audio_filepath=None):
    conn = get_db()
    conn.execute("DELETE FROM transcriptions WHERE id=?", (tid,))
    conn.commit()
    conn.close()
    if audio_filepath and os.path.exists(audio_filepath):
        try:
            os.remove(audio_filepath)
            parent_dir = os.path.dirname(audio_filepath)
            if parent_dir != AUDIO_STORE and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
        except OSError as e:
            st.warning(f"Could not delete audio file: {e}")

def save_translations(tid, en_text, ur_text):
    conn = get_db()
    conn.execute(
        "UPDATE transcriptions SET translation_en=?, translation_ur=? WHERE id=?",
        (en_text, ur_text, tid)
    )
    conn.commit()
    conn.close()

def get_user_history(user_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, filename, original_transcription, edited_transcription, "
        "reference_text, wer_score, translation_en, translation_ur, audio_filepath, created_at "
        "FROM transcriptions WHERE user_id=? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return rows

# ── Whisper Core Pipeline Model Caching Layout (Loaded Once On Startup) ───────
@st.cache_resource
def load_cached_transcription_engine():
    """
    Initializes your structural object pipeline model single-instance map wrapper 
    directly pointing toward your target Hugging Face repository at app launch.
    """
    # Simply instantiates your production class component using clean model path targets
    return PashtoTranscriber(model_id_or_path=MERGED_MODEL_PATH, hf_token=HF_TOKEN)

# ── Translation Engine — NLLB-200 600M ────────────────────────────────────────
@st.cache_resource
def load_translation_model():
    return hf_pipeline(
        "translation",
        model=NLLB_MODEL,
        device=-1,
        torch_dtype=torch.float32,
    )

def translate_text(text, target_lang):
    if not text or not text.strip():
        return ""
    try:
        translator = load_translation_model()
        result = translator(
            text,
            src_lang=PASHTO_CODE,
            tgt_lang=target_lang,
            max_length=512,
            num_beams=4,
        )
        return result[0]["translation_text"]
    except Exception as e:
        return "Translation error: " + str(e)

def verify_translation(original_pashto, translated_text, target_lang):
    if not translated_text or "error" in translated_text.lower():
        return None, "Could not verify", ""
    try:
        translator = load_translation_model()
        back = translator(
            translated_text,
            src_lang=target_lang,
            tgt_lang=PASHTO_CODE,
            max_length=512,
            num_beams=4,
        )
        back_text = back[0]["translation_text"]
        ratio = SequenceMatcher(None, original_pashto.strip(), back_text.strip()).ratio()
        score = round(ratio * 100, 1)
        if score >= 75:
            label = "✅ High confidence"
        elif score >= 50:
            label = "⚠️ Medium confidence"
        else:
            label = "❌ Low confidence — review recommended"
        return score, label, back_text
    except Exception as e:
        return None, f"Verification error: {e}", ""

def compute_wer(reference, hypothesis):
    if not JIWER_AVAILABLE or not reference.strip():
        return None
    try:
        score = jiwer_wer(reference.strip(), hypothesis.strip())
        return round(score * 100, 2)
    except Exception:
        return None

# ── Export Matrix Helpers ─────────────────────────────────────────────────────
def make_csv_bytes(rows, headers):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for row in rows:
        w.writerow([str(cell) if cell is not None else "" for cell in row])
    return ("\ufeff" + buf.getvalue()).encode("utf-8")

def make_excel_bytes(rows, headers):
    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Transcriptions"
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_idx, value=header)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")
        rtl_cols = {"Pashto", "Urdu", "Original", "Edited", "Reference"}
        for row_idx, row in enumerate(rows, 2):
            for col_idx, (header, cell_val) in enumerate(zip(headers, row), 1):
                val = str(cell_val) if cell_val is not None else ""
                cell = ws.cell(row=row_idx, column=col_idx, value=val)
                if any(rtl in header for rtl in rtl_cols):
                    cell.alignment = Alignment(horizontal="right")
        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        return buf.read()
    except ImportError:
        return make_csv_bytes(rows, headers)

# ── AI Chat — Strategy Free Approach ──────────────────────────────────────────
def _try_hf_inference(prompt, system_msg):
    if not HF_TOKEN:
        return None
    req_headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    models_to_try = [
        ("HuggingFaceH4/zephyr-7b-beta", f"<|system|>\n{system_msg}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"),
        ("mistralai/Mistral-7B-Instruct-v0.1", f"[INST] {system_msg}\n\n{prompt} [/INST]"),
        ("google/flan-t5-large", f"{system_msg}\n\n{prompt}"),
        ("gpt2-medium", f"{system_msg}\n\n{prompt}"),
    ]
    for model_id, formatted_prompt in models_to_try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        payload = {
            "inputs": formatted_prompt,
            "parameters": {"max_new_tokens": 400, "temperature": 0.7, "do_sample": True, "return_full_text": False},
            "options": {"wait_for_model": True, "use_cache": False}
        }
        try:
            r = requests.post(url, headers=req_headers, json=payload, timeout=90)
            if r.status_code == 200:
                data = r.json()
                text = ""
                if isinstance(data, list) and data:
                    text = data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    text = data.get("generated_text", "").strip()
                if text and len(text) > 10:
                    return text
        except Exception:
            continue
    return None

def _try_openai_compatible(prompt, system_msg):
    if not HF_TOKEN:
        return None
    try:
        url = "https://api-inference.huggingface.co/v1/chat/completions"
        req_headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}
        payload = {
            "model": "HuggingFaceH4/zephyr-7b-beta",
            "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.7,
        }
        r = requests.post(url, headers=req_headers, json=payload, timeout=60)
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            if text:
                return text
    except Exception:
        pass
    return None

def _try_pollinations(prompt, system_msg):
    try:
        import urllib.parse
        combined = f"{system_msg}\n\nUser question: {prompt}"
        encoded = urllib.parse.quote(combined[:800])
        r = requests.get(f"https://text.pollinations.ai/{encoded}", timeout=60)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except Exception:
        pass
    return None

def ask_ai_free(prompt, system_msg):
    result = _try_hf_inference(prompt, system_msg)
    if result: return result, None
    result = _try_openai_compatible(prompt, system_msg)
    if result: return result, None
    result = _try_pollinations(prompt, system_msg)
    if result: return result, None
    return None, ("All AI strategies failed.\n\n1. **Set HF_TOKEN**\n2. **Wait 30 seconds**\n3. **Check network connection**")

# ── UI Layout Components ──────────────────────────────────────────────────────
def wer_badge(wer_score):
    if wer_score is None:
        return
    if wer_score <= 15: bg, label = "#22c55e", "Excellent"
    elif wer_score <= 35: bg, label = "#f59e0b", "Good"
    elif wer_score <= 60: bg, label = "#f97316", "Fair"
    else: bg, label = "#ef4444", "Poor"
    accuracy = max(0, round(100 - wer_score, 1))
    st.markdown(
        f'<div style="display:flex;gap:10px;align-items:center;margin:8px 0;">'
        f'<div style="background:{bg};color:#fff;font-weight:700;padding:5px 14px;border-radius:20px;font-size:0.85rem;">'
        f'WER: {wer_score}%</div>'
        f'<div style="background:#1e293b;color:#94a3b8;font-size:0.82rem;padding:5px 12px;border-radius:20px;">'
        f'~{accuracy}% accuracy — {label}</div></div>', unsafe_allow_html=True
    )

def show_translation_box(tid, pashto_text, existing_en, existing_ur):
    st.markdown("**Translate:**")
    col_en, col_ur, col_both = st.columns(3)
    with col_en:
        if st.button("To English", key="en_" + str(tid), use_container_width=True):
            with st.spinner("Translating to English..."):
                en = translate_text(pashto_text, ENGLISH_CODE)
            save_translations(tid, en, existing_ur)
            st.session_state["t_en_" + str(tid)] = en
            st.rerun()
    with col_ur:
        if st.button("To Urdu", key="ur_" + str(tid), use_container_width=True):
            with st.spinner("Translating to Urdu..."):
                ur = translate_text(pashto_text, URDU_CODE)
            save_translations(tid, existing_en, ur)
            st.session_state["t_ur_" + str(tid)] = ur
            st.rerun()
    with col_both:
        if st.button("Both Languages", key="both_" + str(tid), use_container_width=True):
            with st.spinner("Translating..."):
                en = translate_text(pashto_text, ENGLISH_CODE)
                ur = translate_text(pashto_text, URDU_CODE)
            save_translations(tid, en, ur)
            st.session_state["t_en_" + str(tid)] = en
            st.session_state["t_ur_" + str(tid)] = ur
            st.rerun()

    en_val = st.session_state.get("t_en_" + str(tid), existing_en or "")
    ur_val = st.session_state.get("t_ur_" + str(tid), existing_ur or "")

    if en_val:
        st.text_area("English", value=en_val, height=85, key="disp_en_" + str(tid))
        col_dl_en, col_vfy_en = st.columns(2)
        with col_dl_en:
            st.download_button("Download English", data=en_val.encode("utf-8"), file_name="en_" + str(tid) + ".txt", mime="text/plain; charset=utf-8", key="dl_en_" + str(tid))
        with col_vfy_en:
            if st.button("Verify English", key="btn_vfy_en_" + str(tid), use_container_width=True):
                with st.spinner("Back-translating to verify..."):
                    score, lbl, back = verify_translation(pashto_text, en_val, ENGLISH_CODE)
                st.session_state["vfy_en_result_" + str(tid)] = (score, lbl, back)
        vfy = st.session_state.get("vfy_en_result_" + str(tid))
        if vfy:
            score, lbl, back = vfy
            st.info(f"**Translation Quality:** {lbl} (similarity {score}%)")
            if back:
                with st.expander("Back-translated Pashto"): st.text(back)

    if ur_val:
        st.text_area("Urdu", value=ur_val, height=85, key="disp_ur_" + str(tid))
        col_dl_ur, col_vfy_ur = st.columns(2)
        with col_dl_ur:
            st.download_button("Download Urdu", data=ur_val.encode("utf-8"), file_name="ur_" + str(tid) + ".txt", mime="text/plain; charset=utf-8", key="dl_ur_" + str(tid))
        with col_vfy_ur:
            if st.button("Verify Urdu", key="btn_vfy_ur_" + str(tid), use_container_width=True):
                with st.spinner("Back-translating to verify..."):
                    score, lbl, back = verify_translation(pashto_text, ur_val, URDU_CODE)
                st.session_state["vfy_ur_result_" + str(tid)] = (score, lbl, back)
        vfy = st.session_state.get("vfy_ur_result_" + str(tid))
        if vfy:
            score, lbl, back = vfy
            st.info(f"**Translation Quality:** {lbl} (similarity {score}%)")
            if back:
                with st.expander("Back-translated Pashto"): st.text(back)

# ── Application Page Routing Layout ───────────────────────────────────────────
def about_page():
    st.markdown("""
    <style>
    .about-hero{background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:16px; padding:2.5rem 2rem;margin-bottom:1.5rem;color:white;}
    .about-hero h1{font-size:2.2rem;font-weight:800;margin-bottom:0.3rem;}
    .about-hero p{font-size:1.05rem;color:#c7d2fe;}
    .feature-card{background:#1e293b;border:1px solid #334155;border-radius:12px; padding:1.2rem 1.4rem;margin-bottom:0.8rem;color:#e2e8f0;}
    .feature-card h4{color:#a78bfa;margin-bottom:0.3rem;font-size:1rem;}
    .feature-card p{font-size:0.88rem;color:#94a3b8;margin:0;}
    .step-box{background:#0f172a;border-left:4px solid #6366f1;padding:0.8rem 1rem; border-radius:0 8px 8px 0;margin-bottom:0.6rem;color:#e2e8f0;font-size:0.9rem;}
    </style>
    <div class="about-hero">
        <h1>🎙️ Pashto Whisper STT</h1>
        <p>A fine-tuned Speech-to-Text system for Pakistani Pashto — built for agriculture, health, food, services domains.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("What is this app?")
    st.markdown("This application uses a fine-tuned OpenAI Whisper model trained specifically on Pakistani Pashto speech.")
    st.markdown("---")

    st.subheader("Features")
    features = [
        ("🎤 Multi-format Audio Upload",   "WAV, MP3, M4A, OGG, FLAC, MP4, OPUS, WEBM, AAC, WMA."),
        ("📝 Pashto Transcription",        "Merged Whisper model — loaded once, fast processing cycle."),
        ("✏️ Editable Transcriptions",     "Correct mistakes and save the improved version."),
        ("📊 WER Scoring",                 "Paste reference text to get Word Error Rate + accuracy %."),
        ("🌐 English & Urdu Translation",  "NLLB-200 600M runs locally — no API needed."),
        ("✅ Translation Verification",    "Back-translates output and shows similarity score."),
        ("📁 CSV & Excel Export",          "UTF-8 BOM encoding — Pashto/Urdu text displays correctly."),
        ("🤖 AI Assistant",                "Three-strategy free AI — works without paid subscriptions."),
        ("🔒 User Accounts",               "Private accounts — your data stored separately."),
        ("📂 Audio Playback in History",   "Replay any uploaded audio directly from history."),
    ]
    col1, col2 = st.columns(2)
    for i, (title, desc) in enumerate(features):
        target = col1 if i % 2 == 0 else col2
        with target:
            st.markdown(f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

def auth_page():
    st.markdown(
        '<style>.auth-title{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;}.auth-sub{color:#94a3b8;font-size:1rem;margin-bottom:2rem;}</style>'
        '<div class="auth-title">Pashto Whisper</div>'
        '<div class="auth-sub">Merged LoRA Fine-Tuned Speech Transcription</div>', unsafe_allow_html=True
    )
    tab_login, tab_signup = st.tabs(["Login", "Create Account"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True, type="primary"):
                ok, uid = verify_user(username.strip(), password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user_id   = uid
                    st.session_state.username  = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    with tab_signup:
        with st.form("signup_form"):
            new_user  = st.text_input("Username")
            new_email = st.text_input("Email (optional)")
            new_pass  = st.text_input("Password", type="password")
            confirm   = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Create Account", use_container_width=True, type="primary"):
                if len(new_user.strip()) < 3: st.error("Username must be at least 3 characters.")
                elif len(new_pass) < 6: st.error("Password must be at least 6 characters.")
                elif new_pass != confirm: st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(new_user.strip(), new_pass, new_email.strip())
                    st.success(msg) if ok else st.error(msg)

# ── Dynamic Core Transcription Execution Routing ──────────────────────────────
def transcribe_page():
    st.header("Transcribe Audio")
    st.caption("Supported: " + ", ".join("." + f for f in SUPPORTED_FORMATS))
    
    # Lazy call the global runtime engine once upon routing access
    engine_instance = load_cached_transcription_engine()
    
    uploaded_files = st.file_uploader(
        "Upload audio files", 
        type=SUPPORTED_FORMATS, 
        accept_multiple_files=True
    )
    ref_text = st.text_area(
        "Reference text (optional) — paste correct Pashto text to calculate WER", 
        height=70, 
        placeholder="Paste correct Pashto text here for WER..."
    )
    
    if not uploaded_files:
        st.info("Upload audio files above to begin transcription.")
        return
        
    st.markdown(f"**{len(uploaded_files)} file(s) selected**")
    if not st.button("Transcribe All", type="primary", use_container_width=True):
        return
        
    progress_bar = st.progress(0, text="Starting pipeline processes...")
    all_results = []
    
    for i, f in enumerate(uploaded_files):
        st.markdown("---")
        st.markdown(f"**[{i+1}/{len(uploaded_files)}] {f.name}**")
        
        # Step 1: Save temporary files inside your explicitly requested workspace path
        tmp_path = os.path.join(TEMP_DIR, f"upload_{i}_{f.name}")
        with open(tmp_path, "wb") as out:
            out.write(f.getbuffer())
            
        # Step 2: Establish dedicated structural directory routes for historical data persistence
        user_dir = os.path.join(AUDIO_STORE, str(st.session_state.user_id))
        os.makedirs(user_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_path = os.path.join(user_dir, f"{ts}_{f.name}")
        
        with open(saved_path, "wb") as out:
            out.write(f.getbuffer())
            
        # Step 3: Validate and Audio Playback Execution Layer
        st.audio(tmp_path)
        
        # Step 4: Core Generation Pipeline Process
        with st.spinner(f"Decoding target speech parameters for: {f.name}..."):
            try:
                # Use decoupled pipeline engine logic securely
                text = engine_instance.transcribe(tmp_path)
                wer_score = compute_wer(ref_text, text) if ref_text.strip() else None
                
                # Persist details downstream directly within database matrix structure
                tid = save_transcription(
                    st.session_state.user_id, f.name, text, 
                    audio_filepath=saved_path, reference=ref_text.strip() or None, wer_score=wer_score
                )
                all_results.append((tid, f.name, text, wer_score))
                
                # Step 5: Display Result Text Block Matrix UI
                st.text_area(f"Pashto Transcription — {f.name}", value=text, height=100, key=f"res_{i}_{tid}")
                if wer_score is not None:
                    wer_badge(wer_score)
                    
            except Exception as e:
                st.error(f"Execution Error processing {f.name}: {str(e)}")
                
            finally:
                # Step 6: Automated space lifecycle maintenance cleanup
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except OSError: pass
                    
        progress_bar.progress((i + 1) / len(uploaded_files), text=f"Completed {i+1}/{len(uploaded_files)}")
        
    st.markdown("---")
    st.success(f"All {len(uploaded_files)} file(s) transcribed and saved!")
    
    # Step 7: Export / Session Download Interface Configuration Setup
    if all_results:
        headers = ["#", "Filename", "Transcription", "WER (%)", "Saved At"]
        rows = [[idx, fname, text, ws or "", datetime.now().isoformat()] for idx, (tid, fname, text, ws) in enumerate(all_results, 1)]
        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            st.download_button("Download Session as CSV", data=make_csv_bytes(rows, headers), file_name="session_transcriptions.csv", mime="text/csv; charset=utf-8")
        with col_xlsx:
            st.download_button("Download Session as Excel", data=make_excel_bytes(rows, headers), file_name="session_transcriptions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def history_page():
    st.header("My Transcription History")
    rows = get_user_history(st.session_state.user_id)
    if not rows:
        st.info("No transcriptions yet. Head to Transcribe to get started.")
        return
    st.markdown(f"**{len(rows)} transcription(s) on record**")
    export_headers = ["ID", "Filename", "Original", "Edited", "Reference", "WER (%)", "English", "Urdu", "Audio Path", "Date"]
    export_rows = [list(r) for r in rows]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button("Export All as CSV", data=make_csv_bytes(export_rows, export_headers), file_name="all_transcriptions.csv", mime="text/csv; charset=utf-8", use_container_width=True)
    with col_b:
        st.download_button("Export All as Excel", data=make_excel_bytes(export_rows, export_headers), file_name="all_transcriptions.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with col_c:
        lines = []
        for r in rows:
            tid, fname, orig, edited, ref, ws, en_t, ur_t, audio_fp, ts = r
            lines += ["="*60, f"File : {fname}", f"Date : {ts[:16].replace('T',' ')}", f"WER : {str(ws)+'%' if ws is not None else 'N/A'}", f"Pashto :\n{edited or orig or ''}"]
            if en_t: lines.append(f"English:\n{en_t}")
            if ur_t: lines.append(f"Urdu :\n{ur_t}")
            lines.append("")
        st.download_button("Export All as TXT", data="\n".join(lines).encode("utf-8"), file_name="all_transcriptions.txt", mime="text/plain; charset=utf-8", use_container_width=True)
    st.markdown("---")
    for r in rows:
        tid, filename, original, edited, reference, wer_score, en_t, ur_t, audio_filepath, created_at = r
        date_str = created_at[:16].replace("T", " ") if created_at else ""
        label = f"{filename} | {date_str}"
        if wer_score is not None: label += f" | WER {wer_score}%"
        with st.expander(label):
            st.caption(f"Record #{tid}")
            if audio_filepath and os.path.exists(audio_filepath):
                st.audio(audio_filepath, format="audio/*", start_time=0)
            else:
                st.warning(f"Audio file not found: {filename}")
            if original and original != (edited or original):
                with st.expander("View original (unedited)"): st.text(original)
            edited_val = st.text_area("Pashto Transcription (editable)", value=edited or original or "", key=f"edit_{tid}", height=120)
            new_ref = st.text_input("Reference text for WER", value=reference or "", key=f"ref_{tid}", placeholder="Paste correct Pashto text...")
            if wer_score is not None: wer_badge(wer_score)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Save Edits", key=f"save_{tid}", use_container_width=True, type="primary"):
                    new_wer = compute_wer(new_ref, edited_val) if new_ref.strip() else None
                    update_transcription(tid, edited_val, new_ref.strip() or None, new_wer)
                    st.success("Saved!")
                    st.rerun()
            with col2:
                dk = f"confirm_delete_{tid}"
                if st.session_state.get(dk, False):
                    st.warning("Are you sure?")
                    if st.button("Confirm Delete", key=f"confirm_del_{tid}", use_container_width=True, type="primary"):
                        delete_transcription(tid, audio_filepath)
                        st.session_state[dk] = False
                        st.rerun()
                    if st.button("Cancel", key=f"cancel_del_{tid}", use_container_width=True):
                        st.session_state[dk] = False
                        st.rerun()
                else:
                    if st.button("Remove", key=f"remove_{tid}", use_container_width=True, type="primary"):
                        st.session_state[dk] = True
                        st.rerun()
            with col3:
                st.download_button("Download TXT", data=edited_val.encode("utf-8"), file_name=f"{filename}.txt", mime="text/plain; charset=utf-8", key=f"dl_t_{tid}", use_container_width=True)
            with col4:
                sh = ["Filename", "Pashto", "English", "Urdu", "WER (%)", "Date"]
                sr = [[filename, edited_val, en_t or "", ur_t or "", wer_score or "", created_at]]
                st.download_button("Download CSV", data=make_csv_bytes(sr, sh), file_name=f"{filename}.csv", mime="text/csv; charset=utf-8", key=f"dl_c_{tid}", use_container_width=True)
            st.markdown("---")
            show_translation_box(tid, edited_val, en_t, ur_t)

def ai_assistant_page():
    st.header("AI Assistant")
    st.caption("Ask questions about your transcriptions using free inference engines.")
    rows = get_user_history(st.session_state.user_id)
    if not rows:
        st.info("No transcriptions available. Please transcribe speech files first.")
        return
    options = {f"#{r[0]} | {r[1]} ({r[9][:16].replace('T',' ')})": r for r in rows}
    sel_key = st.selectbox("Select transcription record to reference:", list(options.keys()))
    record = options[sel_key]
    tid, filename, original, edited, reference, wer_score, en_t, ur_t, audio_filepath, created_at = record
    
    st.markdown("### Context Reference Summary")
    st.info(f"**Pashto Text Context:**\n\n{edited or original}")
    if en_t: st.caption(f"**English Translation Reference:** {en_t}")
    
    st.markdown("---")
    user_query = st.text_area("Ask the Assistant anything about this context:", height=80, placeholder="Summarize this dialogue segment or help translate complex regional syntax structures...")
    
    if st.button("Submit Inquiry to AI", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("Please type a clear inquiry block first.")
            return
        sys_msg = (
            f"You are a professional linguistic research assistant analyzing automatic speech recognition datasets.\n"
            f"Context details:\n- Source File: {filename}\n- Transcription: {edited or original}\n"
            f"- English Translation: {en_t or 'N/A'}\n- Urdu Translation: {ur_t or 'N/A'}"
        )
        with st.spinner("Polling available open-access AI strategy channels..."):
            ans, err = ask_ai_free(user_query.strip(), sys_msg)
            if ans:
                st.markdown("### 🤖 Assistant Response")
                st.write(ans)
            else:
                st.error(err)

# ── Global App Execution Root Controller Route ────────────────────────────────
def main():
    st.markdown(
        '<style>div.block-container{padding-top:2rem!important;}</style>',
        unsafe_allow_html=True
    )
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        auth_page()
        return

    with st.sidebar:
        st.markdown("## 🎙️ Pashto Whisper")
        st.markdown(f"**User: {st.session_state.username}**")
        st.markdown("---")
        page = st.radio("Navigate", ["About", "Transcribe", "History", "AI Assistant"], label_visibility="collapsed")
        st.markdown("---")
        if not JIWER_AVAILABLE:
            st.warning("WER disabled. Run: pip install jiwer")
        device_label = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.caption(f"Device     : {device_label}")
        st.caption(f"FP16       : {'ON' if FP16_ENABLED else 'OFF'}")
        st.caption("Model      : Merged Whisper+LoRA")
        st.caption("AI         : HF Free + Pollinations")
        st.caption("Translation: NLLB-200 (local CPU)")
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            for k in ("logged_in", "user_id", "username"):
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    if page == "About": about_page()
    elif page == "Transcribe": transcribe_page()
    elif page == "History": history_page()
    elif page == "AI Assistant": ai_assistant_page()

if __name__ == "__main__":
    main()