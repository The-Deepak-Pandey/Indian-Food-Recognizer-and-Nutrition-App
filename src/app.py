"""DP Naturals — local-only Flask web app.

Run:
    cd dp_naturals
    pip install -r requirements.txt
    python app.py
    # open http://127.0.0.1:5000

Override paths if your model/nutrition files live elsewhere:
    DP_MODEL_CKPT=/path/to/dinov2_large_best.pt \
    DP_NUTRITION_DB=/path/to/nutrition_db.json \
    python app.py
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from model_api import get_recognizer


APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
HISTORY_FILE = APP_DIR / "history.json"
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_HISTORY = 20

UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


def _save_history(entry: dict) -> None:
    items = []
    if HISTORY_FILE.exists():
        try:
            items = json.loads(HISTORY_FILE.read_text())
        except json.JSONDecodeError:
            items = []
    items.insert(0, entry)
    items = items[:MAX_HISTORY]
    HISTORY_FILE.write_text(json.dumps(items, indent=2))


@app.route("/")
def index():
    return render_template("index.html", advertised_accuracy=87)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded. Use form field 'image'."}), 400
    f = request.files["image"]
    if not f or not f.filename:
        return jsonify({"error": "Empty filename."}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXT)}"}), 400

    fname = f"{uuid.uuid4().hex}{ext}"
    path = UPLOAD_DIR / fname
    f.save(path)

    try:
        rec = get_recognizer()
        result = rec.predict(path)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e!r}"}), 500

    result["image_url"] = f"/uploads/{fname}"
    result["timestamp"] = int(time.time())

    _save_history({
        "image_url": result["image_url"],
        "top": result.get("top"),
        "msp": result.get("msp"),
        "is_ood": result.get("is_ood"),
        "timestamp": result["timestamp"],
    })

    return jsonify(result)


@app.route("/uploads/<path:fname>")
def uploaded_file(fname: str):
    return send_from_directory(UPLOAD_DIR, fname)


@app.route("/history")
def history():
    if not HISTORY_FILE.exists():
        return jsonify([])
    try:
        return jsonify(json.loads(HISTORY_FILE.read_text()))
    except json.JSONDecodeError:
        return jsonify([])


@app.route("/health")
def health():
    try:
        rec = get_recognizer()
        return jsonify({
            "ok": True,
            "device": rec.device,
            "backbone": rec.backbone,
            "img_size": rec.img_size,
            "n_classes": len(rec.classes),
            "val_top1": rec.val_top1,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
