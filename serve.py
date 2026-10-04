"""Local development server for GripSense: python3 serve.py [port]

Like `python3 -m http.server`, plus two things the app needs:
- HTTP Range requests (206 Partial Content). Safari and every iPhone browser refuse
  to play MP4 video from a server without them, so the videos stay blank.
- "Cache-Control: no-cache", so the browser always checks for the latest files.
Vercel provides both automatically; this file is only for local testing.
"""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self.range_left = None
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        path = self.translate_path(self.path)
        if not match or not os.path.isfile(path):
            return super().send_head()

        size = os.path.getsize(path)
        first, last = match.groups()
        if first:
            start, end = int(first), int(last) if last else size - 1
        else:                              # "bytes=-N": the last N bytes
            start, end = max(size - int(last or 0), 0), size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        file = open(path, "rb")
        file.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self.range_left = end - start + 1
        return file

    def copyfile(self, source, outputfile):
        remaining = self.range_left
        if remaining is None:
            return super().copyfile(source, outputfile)
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"GripSense: http://localhost:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("", port), Handler).serve_forever()
