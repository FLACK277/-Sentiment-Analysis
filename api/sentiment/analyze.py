from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.append(str(API_DIR))

from _shared_py import analyze_text, read_json_body, send_json  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            body = read_json_body(self)
            text = body.get("text") if isinstance(body.get("text"), str) else ""
            result = analyze_text(text)
            send_json(self, 200, result)
        except ValueError as error:
            send_json(self, 400, {"error": str(error)})
        except Exception as error:
            send_json(self, 500, {"error": str(error)})

    def do_GET(self):
        send_json(self, 405, {"error": "Method not allowed"})

    def do_PUT(self):
        send_json(self, 405, {"error": "Method not allowed"})

    def do_DELETE(self):
        send_json(self, 405, {"error": "Method not allowed"})