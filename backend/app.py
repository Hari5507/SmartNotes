from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
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

CORS(app)


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


# =========================================
# HIGH-SPEED SSL-SAFE GEMINI GENERATOR
# Uses curl --ssl-no-revoke to bypass Windows CRL offline errors
# =========================================

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]


def gemini_generate(prompt):
    """
    High-speed SSL-safe generation using curl --ssl-no-revoke.
    Cascades across models automatically if one is busy.
    Falls back to Google GenAI SDK if needed.
    """
    if not gemini_api_key:
        return None, "Gemini API key is not configured."

    # Try high-speed SSL-safe transport first (eliminates UNEXPECTED_EOF on Windows)
    for model_name in GEMINI_MODELS:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"
            payload = json.dumps({"contents": [{"parts": [{"text": prompt}]}]})

            res = subprocess.run([
                "curl.exe", "--ssl-no-revoke", "-s", "-X", "POST", url,
                "-H", "Content-Type: application/json",
                "--data-binary", "@-"
            ], input=payload.encode("utf-8"), capture_output=True, timeout=12)

            if res.returncode == 0 and res.stdout:
                data = json.loads(res.stdout.decode("utf-8", errors="ignore"))
                if "candidates" in data and data["candidates"]:
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        print(f"Fast SSL generation success using: {model_name}")
                        return parts[0]["text"], None
                elif "error" in data:
                    print(f"Model {model_name} busy: {data['error'].get('message', '')[:70]}")
                    continue
        except Exception as err:
            print(f"Fast curl check on {model_name}: {err}")

    # Fallback to GenAI SDK if curl unavailable
    for model_name in GEMINI_MODELS:
        try:
            client = get_fresh_client()
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            print(f"SDK fallback success using: {model_name}")
            return response.text, None
        except Exception as e:
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
You are SmartNotes, an AI study assistant.
Create a fast practice quiz:
1. 4 Multiple Choice Questions (A, B, C, D) with Correct Answer & 1-line explanation.
2. 2 Conceptual Short-Answer Questions with model answer.

DOCUMENT CONTENT:
{text[:8000]}
"""

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
# FRONTEND ROUTES
# =========================================

@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend(filename):
    return send_from_directory(BASE_DIR, filename)


# =========================================
# PDF & GENERIC FILE UPLOAD ROUTE
# =========================================

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file was uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if ext == "pdf":
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        try:
            reader = PdfReader(file_path)
            extracted_text = ""
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted_text += t + "\n"

            return jsonify({
                "message": "PDF uploaded successfully!",
                "filename": filename,
                "pages": len(reader.pages),
                "text": extracted_text,
                "type": "pdf"
            })
        except Exception as e:
            return jsonify({"error": f"Could not read PDF: {str(e)}"}), 500

    return jsonify({"error": "Only PDF files should be uploaded via /upload. Use /audio/transcribe for audio."}), 400


# =========================================
# AI STUDY TOOLS ENDPOINTS
# =========================================

@app.route("/ai/summarize", methods=["POST"])
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
def ai_flashcards():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_flashcards_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503

    import json
    cleaned = result.strip().strip("`").replace("json\n", "", 1).strip()
    try:
        cards = json.loads(cleaned)
        return jsonify({"success": True, "flashcards": cards, "result": result})
    except Exception:
        return jsonify({"success": True, "result": result})


@app.route("/ai/practice-questions", methods=["POST"])
def ai_practice_questions():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No document text was provided."}), 400

    prompt = build_practice_questions_prompt(text)
    result, error = gemini_generate(prompt)
    if error:
        return jsonify({"error": error}), 503
    return jsonify({"success": True, "result": result})


@app.route("/ai/mindmap", methods=["POST"])
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
        import json
        cleaned = result.strip().strip("`").replace("json\n", "", 1).strip()
        try:
            response_data["flashcards"] = json.loads(cleaned)
        except Exception:
            pass

    return jsonify(response_data)


# =========================================
# AUDIO / MUSIC FILE TRANSCRIPTION ROUTE
# Supports: Microphone recordings & MP3, WAV, M4A, AAC, OGG
# =========================================

@app.route("/audio/transcribe", methods=["POST"])
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
        for attempt in range(2):
            try:
                from google.genai import types
                client = get_fresh_client()
                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=mime_type
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                audio_part,
                                types.Part.from_text(text=prompt_text)
                            ]
                        )
                    ]
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

                return jsonify({
                    "success": True,
                    "filename": filename,
                    "transcript": transcript_text,
                    "notes": notes_text,
                    "source": "audio"
                })

            except Exception as e:
                last_error = e
                error_text = str(e)
                print(f"{model_name} attempt {attempt + 1} audio note: {error_text[:120]}")
                if any(err in error_text for err in ["503", "UNAVAILABLE", "429", "EOF", "SSL", "RemoteDisconnected"]):
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                break

    return jsonify({"error": f"Audio processing is busy. Please retry in a moment."}), 503


# =========================================
# YOUTUBE PROCESS ROUTE (FAST & SSL-SAFE)
# Universal support via Transcript API + Official oEmbed API
# =========================================

@app.route("/youtube/process", methods=["POST"])
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

    return jsonify({
        "success": True,
        "video_id": video_id,
        "title": video_title,
        "transcript": transcript_text[:3000] if transcript_text else f"Lecture: {video_title}\nInstructor: {author}",
        "notes": result,
        "source": "youtube"
    })


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
