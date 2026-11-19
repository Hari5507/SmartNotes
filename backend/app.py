from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from pypdf import PdfReader

from google import genai

from dotenv import load_dotenv

import os
import time
import base64
import re
import json
import subprocess
import urllib.request
import tempfile
import secrets
from datetime import datetime, timedelta

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    YOUTUBE_API_AVAILABLE = False
    print("WARNING: youtube-transcript-api not installed.")


# =========================================
# LOAD ENVIRONMENT VARIABLES
# =========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

try:
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except Exception as e:
    print(f"Notice: .env file skipped or inaccessible ({e})")


# =========================================
# FLASK APP
# =========================================

app = Flask(__name__)

CORS(app, supports_credentials=True)

# =========================================
# AUTH & DB CONFIGURATION
# =========================================

# Permanent stable secret key fallback to prevent invalidating user sessions across restarts / gunicorn workers
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or 'smartnotes-production-super-secret-key-32bytes-auth-hash-2026'
db_path = os.path.join(BASE_DIR, 'smartnotes.db').replace('\\', '/')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = '/login'
login_manager.login_message = None


# =========================================
# USER & DOCUMENT MODELS
# =========================================

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Document(db.Model):
    __tablename__ = 'documents'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(50), default='pdf')  # 'pdf', 'audio', 'youtube'
    pages = db.Column(db.Integer, default=1)
    extracted_text = db.Column(db.Text, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    summary = db.Column(db.Text, nullable=True)
    key_points = db.Column(db.Text, nullable=True)
    flashcards = db.Column(db.Text, nullable=True)
    practice_questions = db.Column(db.Text, nullable=True)
    mindmap = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('documents', lazy=True, cascade='all, delete-orphan'))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'filename': self.filename,
            'name': self.filename,
            'file_type': self.file_type,
            'type': self.file_type,
            'pages': self.pages,
            'extracted_text': self.extracted_text,
            'text': self.extracted_text,
            'notes': self.notes,
            'summary': self.summary,
            'key_points': self.key_points,
            'flashcards': self.flashcards,
            'practice_questions': self.practice_questions,
            'mindmap': self.mindmap,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401


# =========================================
# GEMINI AI CONFIGURATION
# =========================================

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("WARNING: GEMINI_API_KEY is not configured.")
else:
    print("Gemini API key loaded successfully.")


def get_fresh_client():
    if not gemini_api_key:
        return None
    return genai.Client(api_key=gemini_api_key)


# =========================================
# UPLOAD CONFIGURATION
# =========================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"pdf", "mp3", "wav", "m4a", "aac", "ogg", "webm"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_user_upload_folder(user_id):
    folder = os.path.join(UPLOAD_FOLDER, f'user_{user_id}')
    os.makedirs(folder, exist_ok=True)
    return folder


# =========================================
# HIGH-SPEED SSL-SAFE GEMINI GENERATOR
# Verified active Google Gemini 3 preview models
# =========================================

GEMINI_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
]


def gemini_generate(prompt):
    """
    High-speed SSL-safe generation using Gemini 3 Preview models.
    Tries Google GenAI SDK first, then cascades to direct REST with SSL bypass.
    Works reliably on Windows, macOS, and Linux/Render.
    """
    if not gemini_api_key:
        return None, "Gemini API key is not configured."

    # Method 1: Google GenAI Client (Fastest & direct)
    client = get_fresh_client()
    for model_name in GEMINI_MODELS:
        try:
            if client:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and response.text:
                    return response.text, None
        except Exception as e:
            print(f"GenAI SDK notice on {model_name}: {str(e)[:100]}")
            continue

    # Method 2: Direct REST fallback via urllib with unverified SSL context (works universally on all OS without curl.exe)
    import ssl
    ctx = ssl._create_unverified_context()
    for model_name in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"
            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                if "candidates" in data and data["candidates"]:
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"], None
        except Exception as err:
            print(f"REST fallback notice on {model_name}: {str(err)[:100]}")
            continue

    return None, "All AI models are temporarily busy. Please retry in a moment."


# =========================================
# PROMPTS FOR AI STUDY TOOLS (TUNED FOR SPEED)
# =========================================

def build_smart_notes_prompt(text):
    return f"""
You are SmartNotes, an elite AI study assistant.
Generate clean, organized Smart Notes for college students from this lecture:

Format with clear markdown:
# 📘 [Lecture Title]
## 🎯 Overview
(2-3 sentences summarizing the core topic)

## 📌 Core Concepts & Notes
(Topic-wise sections with clear subheadings, bold terms, and high-yield bullet points)

## 💡 Key Definitions & Formulas
(Bullet list of essential terms, definitions, and equations)

## ⚠️ Important Exam Tips
(Crucial takeaways for exams)

Keep the notes concise, clear, and student-friendly.

DOCUMENT CONTENT:
{text[:9000]}
"""

def build_summary_prompt(text):
    return f"""
You are SmartNotes, an AI study assistant.
Summarize this lecture content in 3-5 concise bullet points covering the most essential concepts for rapid revision:

DOCUMENT CONTENT:
{text[:8000]}
"""

def build_key_points_prompt(text):
    return f"""
You are SmartNotes, an AI study assistant.
Extract 8 to 12 high-impact KEY POINTS with emojis (🔑, ⚡, 📌, 💡):
- Bold keywords and formulas.
- High-yield, testable facts only.

DOCUMENT CONTENT:
{text[:8000]}
"""

def build_flashcards_prompt(text):
    return f"""
You are SmartNotes, an AI study assistant.
Generate 8 interactive revision FLASHCARDS in raw valid JSON format:
[
  {{
    "question": "Question here...",
    "answer": "Concise answer here...",
    "category": "Topic Name"
  }}
]
Return RAW JSON only. Do not wrap in markdown quotes.

DOCUMENT CONTENT:
{text[:8000]}
"""

def build_practice_questions_prompt(text):
    return f"""
You are SmartNotes, an elite AI study assistant.
Generate an interactive multiple-choice practice quiz (4 to 6 questions) based on this lecture content.

Format your output as a RAW JSON array of objects.
Each object must have the following exact schema:
[
  {{
    "id": 1,
    "question": "Clear multiple choice question text?",
    "options": [
      "First option",
      "Second option",
      "Third option",
      "Fourth option"
    ],
    "correct": 0,
    "correct_letter": "A",
    "explanation": "Clear explanation of why this option is correct."
  }}
]

RULES:
1. "correct" MUST be the 0-based integer index of the correct answer (0, 1, 2, or 3).
2. "correct_letter" MUST be the matching letter ("A", "B", "C", or "D").
3. "options" MUST be a list of exactly 4 strings.
4. "explanation" MUST be 1-2 sentences explaining why the correct answer is right.
5. Output ONLY the raw JSON array. Do not wrap in markdown code blocks, do not include any preamble or extra text.

DOCUMENT CONTENT:
{text[:8000]}
"""


def parse_practice_questions_json(result_text):
    """
    Safely extracts and validates practice quiz JSON array from AI output.
    Handles raw JSON, markdown-fenced json (```json ... ```), and string variants.
    """
    if not result_text:
        return []
    cleaned = result_text.strip()
    # Strip markdown code fences if model enclosed in ```json ... ```
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    # Find JSON array
    json_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            normalized = []
            for idx, q in enumerate(data):
                if isinstance(q, dict) and "question" in q and "options" in q:
                    options = q.get("options", [])
                    if isinstance(options, list) and len(options) >= 2:
                        correct = q.get("correct", 0)
                        if isinstance(correct, str):
                            if correct.upper() in ["A", "B", "C", "D"]:
                                correct = ord(correct.upper()) - ord("A")
                            else:
                                try:
                                    correct = int(correct)
                                except Exception:
                                    correct = 0
                        correct_letter = q.get("correct_letter")
                        if not correct_letter:
                            letters = ["A", "B", "C", "D"]
                            correct_letter = letters[correct] if 0 <= correct < len(letters) else "A"
                        normalized.append({
                            "id": q.get("id", idx + 1),
                            "question": q.get("question", ""),
                            "options": options[:4],
                            "correct": correct if 0 <= correct < len(options) else 0,
                            "correct_letter": correct_letter,
                            "explanation": q.get("explanation", "Correct answer based on the lecture material.")
                        })
            if normalized:
                return normalized
    except Exception as err:
        print(f"Notice: practice questions JSON parse note: {err}")
    return []

def build_mindmap_prompt(text):
    return f"""
You are SmartNotes, an AI study assistant.
Create a structured hierarchical MIND MAP outline from this lecture:
Central Topic -> 3 to 4 Main Branches -> 2 to 3 Sub-concepts each.
Also include a short Mermaid mindmap diagram:
```mermaid
mindmap
  root((Topic))
    Branch1
      Sub1
    Branch2
      Sub2
```

DOCUMENT CONTENT:
{text[:8000]}
"""


# =========================================
# AUTH ROUTES
# =========================================

@app.route('/login')
def login_page():
    if current_user.is_authenticated:
        return redirect('/')
    return send_from_directory(BASE_DIR, 'login.html')


@app.route('/register')
def register_page():
    if current_user.is_authenticated:
        return redirect('/')
    return send_from_directory(BASE_DIR, 'login.html')


@app.route('/api/auth/register', methods=['POST'])
def api_register():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    if not name or not email or not password:
        return jsonify({'error': 'All fields are required.'}), 400
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters.'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'An account with this email already exists.'}), 409
    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    session.permanent = True
    login_user(user, remember=True)
    return jsonify({'success': True, 'user': {'id': user.id, 'name': user.name, 'email': user.email}}), 201


@app.route('/api/auth/login', methods=['POST'])
def api_login():
    data = request.get_json(silent=True) or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    remember = data.get('remember', True)
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Incorrect email or password.'}), 401
    session.permanent = True
    login_user(user, remember=remember)
    return jsonify({'success': True, 'user': {'id': user.id, 'name': user.name, 'email': user.email}})


@app.route('/api/auth/logout', methods=['POST'])
def api_logout():
    logout_user()
    return jsonify({'success': True})


@app.route('/api/auth/me')
def api_me():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True, 'user': {'id': current_user.id, 'name': current_user.name, 'email': current_user.email}})
    return jsonify({'authenticated': False}), 401


# =========================================
# DOCUMENT MANAGEMENT ROUTES (USER ISOLATED)
# =========================================

@app.route('/api/documents', methods=['GET'])
@login_required
def get_user_documents():
    """Return all documents owned by current_user, newest first."""
    docs = Document.query.filter_by(user_id=current_user.id).order_by(Document.created_at.desc()).all()
    return jsonify({
        'success': True,
        'documents': [doc.to_dict() for doc in docs]
    })


@app.route('/api/documents/<int:doc_id>', methods=['GET'])
@login_required
def get_single_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first()
    if not doc:
        return jsonify({'error': 'Document not found or access denied.'}), 404
    return jsonify({'success': True, 'document': doc.to_dict()})


@app.route('/api/documents', methods=['POST'])
@login_required
def save_or_update_document():
    data = request.get_json(silent=True) or {}
    filename = data.get('filename') or data.get('name') or 'Untitled Lecture'
    file_type = data.get('file_type') or data.get('type') or 'pdf'
    pages = int(data.get('pages', 1))
    extracted_text = data.get('extracted_text') or data.get('text') or ''
    notes = data.get('notes') or ''
    summary = data.get('summary') or ''
    key_points = data.get('key_points') or ''
    flashcards = data.get('flashcards') or ''
    practice_questions = data.get('practice_questions') or ''
    mindmap = data.get('mindmap') or ''

    # Check if a document with this filename already exists for this user
    doc = Document.query.filter_by(user_id=current_user.id, filename=filename).first()
    if doc:
        if extracted_text: doc.extracted_text = extracted_text
        if pages: doc.pages = pages
        if notes: doc.notes = notes
        if summary: doc.summary = summary
        if key_points: doc.key_points = key_points
        if flashcards: doc.flashcards = flashcards
        if practice_questions: doc.practice_questions = practice_questions
        if mindmap: doc.mindmap = mindmap
    else:
        doc = Document(
            user_id=current_user.id,
            filename=filename,
            file_type=file_type,
            pages=pages,
            extracted_text=extracted_text,
            notes=notes,
            summary=summary,
            key_points=key_points,
            flashcards=flashcards,
            practice_questions=practice_questions,
            mindmap=mindmap
        )
        db.session.add(doc)

    db.session.commit()
    return jsonify({'success': True, 'document': doc.to_dict()}), 201


@app.route('/api/documents/<int:doc_id>', methods=['DELETE'])
@login_required
def delete_document(doc_id):
    doc = Document.query.filter_by(id=doc_id, user_id=current_user.id).first()
    if not doc:
        return jsonify({'error': 'Document not found.'}), 404
    db.session.delete(doc)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Document deleted.'})


@app.route('/api/documents/by-name/<filename>', methods=['DELETE'])
@login_required
def delete_document_by_name(filename):
    doc = Document.query.filter_by(filename=filename, user_id=current_user.id).first()
    if not doc:
        return jsonify({'error': 'Document not found.'}), 404
    db.session.delete(doc)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Document {filename} deleted.'})


# =========================================
# FRONTEND ROUTES
# =========================================

@app.route("/")
def home():
    if not current_user.is_authenticated:
        return redirect('/login')
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend(filename):
    # Protect document.html and other app pages
    protected_pages = {'document.html', 'index.html'}
    if filename in protected_pages and not current_user.is_authenticated:
        return redirect('/login')
    return send_from_directory(BASE_DIR, filename)


# =========================================
# PDF & GENERIC FILE UPLOAD ROUTE
# =========================================

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if ext == "pdf":
        user_folder = get_user_upload_folder(current_user.id)
        file_path = os.path.join(user_folder, filename)
        file.save(file_path)

        try:
            reader = PdfReader(file_path)
            extracted_text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted_text += t + "\n"

            # Persist document record in SQLite database isolated to current user
            doc = Document.query.filter_by(user_id=current_user.id, filename=filename).first()
            if not doc:
                doc = Document(
                    user_id=current_user.id,
                    filename=filename,
                    file_type="pdf",
                    pages=len(reader.pages),
                    extracted_text=extracted_text
                )
                db.session.add(doc)
            else:
                doc.pages = len(reader.pages)
                doc.extracted_text = extracted_text
            db.session.commit()

            return jsonify({
                "message": "PDF uploaded successfully!",
                "filename": filename,
                "pages": len(reader.pages),
                "text": extracted_text,
                "type": "pdf",
                "document_id": doc.id
            })
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {str(e)}"}), 500

    return jsonify({"error": "Only PDF files should be uploaded via /upload. Use /audio/transcribe for audio."}), 400


# =========================================
# AI STUDY TOOLS ENDPOINTS
# =========================================

@app.route("/ai/summarize", methods=["POST"])
@login_required
def ai_summarize():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_summary_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503
    return jsonify({"success": True, "result": result})


@app.route("/ai/smart-notes", methods=["POST"])
@login_required
def ai_smart_notes():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_smart_notes_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503
    return jsonify({"success": True, "result": result})


@app.route("/ai/key-points", methods=["POST"])
@login_required
def ai_key_points():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_key_points_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503
    return jsonify({"success": True, "result": result})


@app.route("/ai/flashcards", methods=["POST"])
@login_required
def ai_flashcards():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_flashcards_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503

    cleaned = result.strip().strip("`").replace("json\n", "", 1).strip()
    try:
        cards = json.loads(cleaned)
        return jsonify({"success": True, "flashcards": cards, "result": result})
    except Exception:
        return jsonify({"success": True, "result": result})


@app.route("/ai/practice-questions", methods=["POST"])
@login_required
def ai_practice_questions():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_practice_questions_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503

    questions = parse_practice_questions_json(result)
    return jsonify({"success": True, "questions": questions, "result": result})


@app.route("/ai/mindmap", methods=["POST"])
@login_required
def ai_mindmap():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_mindmap_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503
    return jsonify({"success": True, "result": result})


@app.route("/ai/tool", methods=["POST"])
@login_required
def ai_unified_tool():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    tool = data.get("tool", "smart_notes").strip()

    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompts = {
        "smart_notes": build_smart_notes_prompt,
        "summarize": build_summary_prompt,
        "key_points": build_key_points_prompt,
        "flashcards": build_flashcards_prompt,
        "practice_questions": build_practice_questions_prompt,
        "mindmap": build_mindmap_prompt
    }

    builder = prompts.get(tool, build_smart_notes_prompt)
    prompt = builder(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503

    response_data = {"success": True, "tool": tool, "result": result}
    if tool == "flashcards":
        cleaned = result.strip().strip("`").replace("json\n", "", 1).strip()
        try:
            response_data["flashcards"] = json.loads(cleaned)
        except Exception:
            pass
    elif tool == "practice_questions":
        response_data["questions"] = parse_practice_questions_json(result)

    return jsonify(response_data)


# =========================================
# AUDIO / MUSIC FILE TRANSCRIPTION ROUTE
# Supports: Microphone recordings & MP3, WAV, M4A, AAC, OGG
# =========================================

@app.route("/audio/transcribe", methods=["POST"])
@login_required
def transcribe_audio():
    if not gemini_api_key:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    if "audio" not in request.files:
        return jsonify({"error": "No audio file received."}), 400

    audio_file = request.files["audio"]
    raw_filename = audio_file.filename or "Lecture_Audio.mp3"
    filename = secure_filename(raw_filename) or "Lecture_Audio.mp3"

    content_type = audio_file.content_type or "audio/mpeg"
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if ext in ["mp3", "mpeg"] or "mp3" in content_type or "mpeg" in content_type:
        mime_type = "audio/mp3"
    elif ext in ["wav"] or "wav" in content_type:
        mime_type = "audio/wav"
    elif ext in ["m4a", "mp4", "aac"] or "m4a" in content_type:
        mime_type = "audio/mp4"
    elif ext in ["ogg"] or "ogg" in content_type:
        mime_type = "audio/ogg"
    else:
        mime_type = "audio/webm"

    audio_bytes = audio_file.read()
    if len(audio_bytes) < 100:
        return jsonify({"error": "Audio file is empty or too short."}), 400

    print(f"Received audio: {filename} ({len(audio_bytes)} bytes, MIME: {mime_type})")

    prompt_text = """
You are SmartNotes, an elite AI study assistant.
Transcribe this spoken lecture accurately, then generate high-yield study notes:

STEP 1 — TRANSCRIPT:
(Concise transcript of the lecture speech)

STEP 2 — ORGANIZED STUDY NOTES:
# 📘 Lecture Notes
## 🎯 Key Summary
## 📌 Core Concepts (bullet points & definitions)
## ❓ Revision Questions & Answers
"""

    last_error = None
    for model_name in GEMINI_MODELS:
        print(f"Trying audio transcription with: {model_name}")
        try:
            from google.genai import types
            client = get_fresh_client()
            audio_part = types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            )
            response = client.models.generate_content(
                model=model_name,
                contents=[audio_part, prompt_text]
            )

            full_response = response.text
            transcript_text = ""
            notes_text = full_response

            for sep in ["STEP 2", "ORGANIZED STUDY NOTES", "Study Notes", "NOTES:", "---"]:
                if sep in full_response:
                    parts = full_response.split(sep, 1)
                    transcript_text = parts[0].strip()
                    notes_text = sep + parts[1].strip()
                    break

            if not transcript_text:
                transcript_text = "(Audio transcript processed into study notes below)"

            # Persist document record in SQLite database isolated to current user
            doc = Document.query.filter_by(user_id=current_user.id, filename=filename).first()
            if not doc:
                doc = Document(
                    user_id=current_user.id,
                    filename=filename,
                    file_type="audio",
                    pages=1,
                    extracted_text=transcript_text,
                    notes=notes_text
                )
                db.session.add(doc)
            else:
                doc.extracted_text = transcript_text
                doc.notes = notes_text
            db.session.commit()

            return jsonify({
                "success": True,
                "filename": filename,
                "transcript": transcript_text,
                "notes": notes_text,
                "source": "audio",
                "document_id": doc.id
            })

        except Exception as e:
            last_error = e
            print(f"{model_name} audio note: {str(e)[:120]}")
            continue

    return jsonify({"error": f"Audio processing is busy. Please retry in a moment."}), 503


# =========================================
# YOUTUBE PROCESS ROUTE (FAST & SSL-SAFE)
# Universal support via Transcript API + Official oEmbed API
# =========================================

@app.route("/youtube/process", methods=["POST"])
@login_required
def process_youtube():
    if not gemini_api_key:
        return jsonify({"error": "Gemini API key is not configured."}), 500

    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "No YouTube URL provided."}), 400

    video_id = None
    patterns = [r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})"]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            video_id = m.group(1)
            break

    if not video_id:
        return jsonify({"error": "Could not extract a valid YouTube video ID from the URL."}), 400

    print(f"Processing YouTube video ID: {video_id}")
    transcript_text = ""
    video_title = f"YouTube Lecture ({video_id})"
    author = "YouTube Instructor"

    # --- Step 1: Fetch official YouTube metadata via oEmbed (Instant & zero bot check) ---
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read().decode("utf-8"))
            video_title = info.get("title", video_title)
            author = info.get("author_name", author)
            print(f"oEmbed Title: {video_title} by {author}")
    except Exception as e:
        print(f"oEmbed fetch note: {e}")

    # --- Step 2: Try fetching transcripts via YouTubeTranscriptApi ---
    if YOUTUBE_API_AVAILABLE:
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            else:
                yt_api = YouTubeTranscriptApi()
                transcript_list = yt_api.fetch(video_id)

            transcript_text = " ".join(
                getattr(entry, "text", entry["text"] if isinstance(entry, dict) else str(entry))
                for entry in transcript_list
            )
            print(f"Transcript fetched via YouTubeTranscriptApi: {len(transcript_text)} chars")
        except Exception as e:
            print(f"YouTubeTranscriptApi note: {e}")

    # Condense long transcripts to high-yield 7500 chars (intro + core middle + summary)
    if len(transcript_text) > 7500:
        first_part = transcript_text[:3500]
        mid_point = len(transcript_text) // 2
        mid_part = transcript_text[mid_point:mid_point + 2500]
        last_part = transcript_text[-1500:]
        content_for_ai = f"{first_part}\n\n[... lecture continues ...]\n\n{mid_part}\n\n[... lecture wrap-up ...]\n\n{last_part}"
    elif transcript_text.strip():
        content_for_ai = transcript_text
    else:
        content_for_ai = f"Lecture Title: {video_title}\nInstructor: {author}\nVideo Link: {url}"

    prompt = f"""
You are SmartNotes, an elite AI university study assistant.
Generate comprehensive, student-friendly lecture notes from this YouTube lecture:

TITLE: {video_title}
INSTRUCTOR: {author}
CONTENT / TRANSCRIPT:
{content_for_ai}

FORMAT WITH CLEAR MARKDOWN:
# 📘 {video_title}
## 🎯 Lecture Overview
(2-3 sentences summarizing the topic)

## 📌 Detailed Topic Notes
(Key concepts, explanations, and bullet points)

## 💡 Important Definitions & Formulas
(Essential testable terms)

## ❓ 3 Quick Revision Questions & Answers
"""

    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503

    # Persist document record in SQLite database isolated to current user
    doc = Document.query.filter_by(user_id=current_user.id, filename=video_title).first()
    if not doc:
        doc = Document(
            user_id=current_user.id,
            filename=video_title,
            file_type="youtube",
            pages=1,
            extracted_text=content_for_ai,
            notes=result
        )
        db.session.add(doc)
    else:
        doc.extracted_text = content_for_ai
        doc.notes = result
    db.session.commit()

    return jsonify({
        "success": True,
        "video_id": video_id,
        "title": video_title,
        "transcript": transcript_text[:3000] if transcript_text else f"Lecture: {video_title}\nInstructor: {author}",
        "notes": result,
        "source": "youtube",
        "document_id": doc.id
    })


# =========================================
# GLOBAL ERROR HANDLERS (ALWAYS RETURN JSON FOR APIS)
# =========================================

@app.errorhandler(500)
def handle_internal_server_error(e):
    if request.path.startswith('/api/') or request.path.startswith('/ai/') or request.path in ['/upload', '/audio/transcribe', '/youtube/process']:
        return jsonify({
            'error': 'The server encountered an error processing this request. Please retry.',
            'success': False
        }), 500
    return e


@app.errorhandler(Exception)
def handle_unhandled_exception(e):
    if request.path.startswith('/api/') or request.path.startswith('/ai/') or request.path in ['/upload', '/audio/transcribe', '/youtube/process']:
        print(f"Handled API error on {request.path}: {e}")
        return jsonify({
            'error': str(e) or 'An unexpected error occurred. Please retry.',
            'success': False
        }), 500
    raise e


# =========================================
# DB INITIALIZATION
# =========================================

with app.app_context():
    db.create_all()


# =========================================
# START SERVER
# =========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1")
    app.run(
        host="0.0.0.0",
        debug=debug_mode,
        port=port,
        use_reloader=False
    )
