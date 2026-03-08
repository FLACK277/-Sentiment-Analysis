from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys


API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.append(str(API_DIR))

from _shared_py import ensure_model_ready, load_dataset_summary, send_json  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            model = ensure_model_ready()
            dataset = load_dataset_summary()
            send_json(self, 200, {"status": "ok", "dataset": dataset, "model": model})
        except Exception as error:
            send_json(self, 500, {"error": str(error)})

    def do_POST(self):
        send_json(self, 405, {"error": "Method not allowed"})

    def do_PUT(self):
        send_json(self, 405, {"error": "Method not allowed"})

    def do_DELETE(self):
        send_json(self, 405, {"error": "Method not allowed"})