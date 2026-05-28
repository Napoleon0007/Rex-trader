"""Combined entry point for Railway deployment.

Runs the trading bot loop in a background thread and serves the Flask
dashboard on the port Railway gives us via $PORT.
"""
import logging
import os
import threading

from bot import main as bot_main
from dashboard import app

log = logging.getLogger("serve")


def start_bot():
    try:
        bot_main()
    except Exception:
        log.exception("bot crashed")


if __name__ == "__main__":
    t = threading.Thread(target=start_bot, daemon=True, name="bot-loop")
    t.start()
    port = int(os.environ.get("PORT", 7433))
    app.run(host="0.0.0.0", port=port, debug=False)
