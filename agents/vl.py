#!/usr/bin/env python3
"""Ask the local vision model about an image.

    python3 vl.py chart.png "What does this chart show?"
    python3 vl.py photo.jpg "Read every sign in this photo."
    PORT=8094 python3 vl.py img.png "..."

Talks to llama-server's OpenAI-compatible endpoint, so the same call works for
any mounting of this model. The image goes as base64 in the message, which is
the only form a local server understands.
"""
import base64
import json
import os
import sys
import time
import urllib.request

PORT = os.environ.get("PORT", "8093")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"


def ask(path: str, question: str, max_tokens: int = 1200) -> str:
    with open(path, "rb") as fh:
        payload = base64.b64encode(fh.read()).decode()
    body = json.dumps({
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{payload}"}},
                {"type": "text", "text": question},
            ],
        }],
        # The model thinks before it answers, and the thinking spends the same
        # budget: 400 tokens was all reasoning and an empty answer.
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"].get("content", "").strip()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = sys.argv[1]
    question = " ".join(sys.argv[2:]) or "Describe this image in one sentence."
    started = time.time()
    answer = ask(path, question)
    print(answer)
    print(f"\n[{time.time() - started:.1f}s · :{PORT}]", file=sys.stderr)


if __name__ == "__main__":
    main()
