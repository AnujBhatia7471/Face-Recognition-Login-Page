import psutil, os
import cv2
import sqlite3
import numpy as np
import onnxruntime as ort
import urllib.request
import threading

from flask import Flask, request, jsonify
from flask_cors import CORS

# ================= MEMORY DEBUG =================
def mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

print(f"[MEM] BOOT: {mem():.2f} MB")
print("🔥 app.py started")

# ================= APP =================
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.debug = True

CORS(app, resources={r"/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
THRESHOLD = 0.75   # 🔥 SAFE THRESHOLD

# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn, conn.cursor()

conn, cur = get_db()
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    embedding BLOB NOT NULL
)
""")
conn.commit()
conn.close()

# ================= FACE DETECTOR =================
face_detector = cv2.dnn.readNetFromCaffe(
    os.path.join(BASE_DIR, "deploy.prototxt"),
    os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
)

# ================= ARC FACE (LOAD ONCE) =================
MODEL_URL = (
    "https://huggingface.co/FoivosPar/Arc2Face/resolve/"
    "da2f1e9aa3954dad093213acfc9ae75a68da6ffd/arcface.onnx"
)
MODEL_PATH = os.path.join(BASE_DIR, "arcface.onnx")

arcface = None
arc_input_name = None
arc_lock = threading.Lock()

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading ArcFace model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ ArcFace model downloaded")

def get_arcface():
    global arcface, arc_input_name
    with arc_lock:
        if arcface is None:
            ensure_model()
            arcface = ort.InferenceSession(
                MODEL_PATH,
                providers=["CPUExecutionProvider"]
            )
            arc_input_name = arcface.get_inputs()[0].name
    return arcface

# ================= UTILS =================
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def detect_face(img):
    if img is None:
        return None

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        img, 1.0, (300, 300), (104, 177, 123)
    )
    face_detector.setInput(blob)
    dets = face_detector.forward()

    for i in range(dets.shape[2]):
        if dets[0, 0, i, 2] > 0.9:
            box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face = img[y1:y2, x1:x2]
            if face.size:
                return face
    return None

def get_embedding(face):
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)

    session = get_arcface()
    emb = session.run(None, {arc_input_name: face})[0][0]
    return emb / np.linalg.norm(emb)

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():
    try:
        email = request.form.get("email", "").lower()
        password = request.form.get("password", "")
        image = request.files.get("image")

        if not email or not password or not image:
            return jsonify(success=False, msg="Missing data")

        conn, cur = get_db()

        cur.execute("SELECT COUNT(*) FROM embeddings WHERE email=?", (email,))
        if cur.fetchone()[0] >= 5:
            conn.close()
            return jsonify(success=False, msg="Already fully registered")

        cur.execute("SELECT email FROM users WHERE email=?", (email,))
        if not cur.fetchone():
            cur.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, password)
            )

        img = cv2.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        face = detect_face(img)
        if face is None:
            conn.close()
            return jsonify(success=False, msg="No face detected")

        emb = get_embedding(face)
        cur.execute(
            "INSERT INTO embeddings (email, embedding) VALUES (?, ?)",
            (email, emb.tobytes())
        )

        cur.execute("SELECT COUNT(*) FROM embeddings WHERE email=?", (email,))
        count = cur.fetchone()[0]

        conn.commit()
        conn.close()

        return jsonify(
            success=True,
            completed=(count == 5),
            msg="Registration completed" if count == 5 else f"Face saved ({count}/5)"
        )

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify(success=False, msg="Internal server error")

# ================= LOGIN (FACE) =================
@app.route("/login/face", methods=["POST"])
def face_login():
    try:
        email = request.form.get("email")
        image = request.files.get("image")

        if not email or not image:
            return jsonify(success=False, msg="Missing data")

        conn, cur = get_db()
        cur.execute("SELECT embedding FROM embeddings WHERE email=?", (email,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return jsonify(success=False, msg="User not registered")

        img = cv2.imdecode(
            np.frombuffer(image.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        face = detect_face(img)
        if face is None:
            return jsonify(success=False, msg="No face detected")

        emb = get_embedding(face)

        scores = []
        for r in rows:
            stored = np.frombuffer(r[0], dtype=np.float32)
            score = cosine_sim(emb, stored)
            scores.append(score)

        best_score = max(scores)
        print("BEST FACE SCORE:", best_score)

        if best_score >= THRESHOLD:
            return jsonify(success=True, msg="Login successful")

        return jsonify(success=False, msg="Face does not match")

    except Exception as e:
        print("FACE LOGIN ERROR:", e)
        return jsonify(success=False, msg="Internal server error")

# ================= LOGIN (PASSWORD) =================
@app.route("/login/password", methods=["POST"])
def password_login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify(success=False, msg="Missing credentials")

        conn, cur = get_db()
        cur.execute("SELECT password FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return jsonify(success=False, msg="User not found")

        if row[0] != password:
            return jsonify(success=False, msg="Invalid password")

        return jsonify(success=True, msg="Login successful")

    except Exception as e:
        print("PASSWORD LOGIN ERROR:", e)
        return jsonify(success=False, msg="Internal server error")

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
