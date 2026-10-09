#!/usr/bin/env python3
"""Local preview server with sane cache headers.

`python3 -m http.server` sends no Cache-Control at all, so browsers fall back to
heuristic caching and will happily reuse a stale index.html or explore.html. That
is the one failure mode that looks like "my edits did nothing": the HTML is
served from cache, so it keeps pointing at the previous ?v= asset URLs and the
page runs old JavaScript against new markup.

This serves the same directory and fixes that:

  * HTML and JSON     -> no-store, so a reload always fetches the current file
  * ?v=-tagged assets -> cached hard, which is safe because the URL changes when
                         the file does

Usage:  python3 serve.py [port]        (default 8000)
"""
from __future__ import annotations

import functools
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        path = self.path.split("?", 1)[0]
        versioned = "?v=" in self.path
        if path.endswith((".html", ".json")) or path.endswith("/") or "." not in path.rsplit("/", 1)[-1]:
            # documents and data: never reuse without asking
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        elif versioned:
            # the URL carries the version, so the bytes behind it never change
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        # a no-store document must not be answered with 304 either
        path = self.path.split("?", 1)[0]
        if path.endswith((".html", ".json")) or path.endswith("/"):
            self.headers.replace_header("If-Modified-Since", "") \
                if "If-Modified-Since" in self.headers else None
            if "If-None-Match" in self.headers:
                del self.headers["If-None-Match"]
            if "If-Modified-Since" in self.headers:
                del self.headers["If-Modified-Since"]
        return super().send_head()

    def log_message(self, fmt, *args):
        code = args[1] if len(args) > 1 else ""
        if str(code).startswith(("4", "5")):      # only surface problems
            super().log_message(fmt, *args)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(Handler, directory=HERE)
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving {HERE}\n  http://localhost:{port}/\n"
              f"  HTML is sent no-store, so a plain reload always shows your edits.\n"
              f"Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


if __name__ == "__main__":
    main()
