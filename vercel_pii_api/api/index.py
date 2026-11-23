from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pii

app = FastAPI(title="PII redaction API", version="1.0")

class Req(BaseModel):
    text: str

PAGE = """<!doctype html><title>PII redaction demo</title><body style="font-family:sans-serif;max-width:720px;margin:40px auto">
<h2>PII redaction demo</h2><p>Synthetic-data-tested rule + validator detector. <b>Do not paste real personal data.</b></p>
<textarea id=t rows=6 style="width:100%">Hi, my name is Priya Nair. Mail priya@example.org or call +91 98765 43210. Card 4111 1111 1111 1111.</textarea>
<p><button onclick="go()">Redact</button></p><pre id=o></pre>
<script>async function go(){const r=await fetch('/redact',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({text:document.getElementById('t').value})});document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2)}</script></body>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/redact")
def redact(r: Req):
    spans = pii.detect(r.text[:20000])
    return {"redacted": pii.redact(r.text[:20000], spans),
            "entities": [{"start": s, "end": e, "type": l} for s, e, l, _ in spans]}
