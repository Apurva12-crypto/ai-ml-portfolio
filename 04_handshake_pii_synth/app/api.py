import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fastapi import FastAPI
from pydantic import BaseModel
import pii
app = FastAPI(title="PII redaction API", version="1.0")
class Req(BaseModel): text: str
@app.get("/health")
def health(): return {"ok": True}
@app.post("/redact")
def redact(r: Req):
    spans = pii.detect(r.text)
    return {"redacted": pii.redact(r.text, spans), "entities": [{"start": s, "end": e, "type": l} for s, e, l, _ in spans]}
