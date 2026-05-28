"""Flask dashboard at http://127.0.0.1:7433 — shows live bot state and trades."""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template

LOG_DIR = Path(__file__).parent / "logs"
STATE_FILE = LOG_DIR / "state.json"
TRADES_FILE = LOG_DIR / "trades.jsonl"
TICKS_FILE = LOG_DIR / "ticks.jsonl"

app = Flask(__name__)


def read_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def read_jsonl(path: Path, tail: int = 100):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()[-tail:]
    out = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    return jsonify({
        "state": read_state(),
        "trades": list(reversed(read_jsonl(TRADES_FILE, 30))),
        "ticks": read_jsonl(TICKS_FILE, 240),
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7433, debug=False)
