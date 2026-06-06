"""
photo_manager.html backend.

Wraps tools/image_pipeline/geometry.py behind a tiny Flask app so the
browser can ask 'where are the corners?' on every upload, parallel to
its Gemini call.

Run from the project root or from this dir:
    python -m tools.image_pipeline.server
    # or
    cd tools/image_pipeline && python server.py

CORS is hard-pinned to http://localhost:8000 (where http.server serves the page).
"""
from __future__ import annotations

import io  # noqa: F401  (kept for parity with plan; future binary handling may use it)
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

# When run as a script (`python server.py`), make the sibling module importable.
sys.path.insert(0, str(Path(__file__).parent))
from geometry import analyze  # noqa: E402

ALLOWED_ORIGIN = "http://localhost:8000"
MAX_BYTES = 50 * 1024 * 1024  # 50 MB, accommodates ~30 MP JPEG

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

log = logging.getLogger("pm-backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.after_request
def add_cors(resp):
    return _cors_headers(resp)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True), 200


@app.route("/analyze", methods=["POST", "OPTIONS"])
def analyze_endpoint():
    if request.method == "OPTIONS":
        return ("", 204)

    f = request.files.get("image")
    if f is None:
        # Allow raw body POST as a fallback (Content-Type: image/jpeg)
        raw = request.get_data()
        if not raw:
            return jsonify(error="missing image"), 400
        data = np.frombuffer(raw, dtype=np.uint8)
    else:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="could not decode image"), 400

    try:
        result = analyze(img)
    except Exception as e:  # detector exceptions shouldn't crash the server
        log.exception("analyze failed")
        return jsonify(error=str(e), source="python", reason="exception"), 500

    log.info("analyze: shape=%s result=%s", img.shape, {
        k: result[k] for k in ("needs_perspective_correction", "needs_crop", "reason")
    })
    return jsonify(result), 200


if __name__ == "__main__":
    # Bind only to loopback — this is a local sidecar, never exposed externally.
    app.run(host="127.0.0.1", port=8001, debug=False, threaded=True)
