"""
SDR App Backend — Minimal API key vault with PIN auth.

Stores the user's LLM API key encrypted on disk. The app fetches the key
at startup by presenting the PIN, then calls the LLM directly from the device.
"""

import asyncio
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import bcrypt
from cryptography.fernet import Fernet
import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

load_dotenv()

app = FastAPI(title="SDR Key Vault")

# Allow Flutter web (and any local dev origin) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only, safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DATA_FILE = Path(os.getenv("DATA_FILE", "key_store.dat"))

# ── Schemas ─────────────────────────────────────────────────────────────
class SetupRequest(BaseModel):
    pin: str
    api_key: str

class KeyRequest(BaseModel):
    pin: str

class GenerateRequest(BaseModel):
    prompt: str
    api_key: str = ""
    llm_base_url: str = ""
    model: str = ""

class EnrichRequest(BaseModel):
    url: str


# ── Helpers ─────────────────────────────────────────────────────────────
def _derive_key(pin: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible key from a PIN + salt."""
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=600_000)
    return base64.urlsafe_b64encode(kdf.derive(pin.encode()))


def _load_store() -> dict | None:
    """Read and decrypt the store file. Returns None if missing or corrupt."""
    if not DATA_FILE.exists():
        return None
    try:
        raw = DATA_FILE.read_bytes()
        parts = raw.split(b"\n", 1)
        if len(parts) != 2:
            return None
        salt, encrypted = parts[0], base64.urlsafe_b64decode(parts[1])
        return {"salt": salt, "encrypted": parts[1].decode(), "raw_encrypted": encrypted}
    except Exception:
        return None


def _save_store(salt: bytes, encrypted: bytes) -> None:
    """Write the store file — salt line + base64 encrypted key."""
    DATA_FILE.write_bytes(salt + b"\n" + base64.urlsafe_b64encode(encrypted))


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    store = _load_store()
    return {
        "status": "ok",
        "has_key": store is not None,
    }


@app.post("/api/setup")
def setup(req: SetupRequest):
    if len(req.pin) < 4:
        raise HTTPException(status_code=400, detail="PIN must be at least 4 characters")
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty")

    salt = os.urandom(16)
    key = _derive_key(req.pin, salt)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(req.api_key.encode())

    _save_store(salt, encrypted)
    return {"status": "ok", "detail": "API key stored successfully"}


@app.post("/api/key")
def get_key(req: KeyRequest):
    store = _load_store()
    if store is None:
        raise HTTPException(status_code=404, detail="No key stored. Call /api/setup first.")

    key = _derive_key(req.pin, store["salt"])
    fernet = Fernet(key)
    try:
        decrypted = fernet.decrypt(store["raw_encrypted"])
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid PIN")

    return {"api_key": decrypted.decode()}


# ── Enrichment ──────────────────────────────────────────────────────────

import re as _re


def _strip_html(html: str) -> str:
    """Remove HTML tags and collapse whitespace.
    Also strips <script>, <style>, <svg> tag contents so embedded JSON-LD/CSS doesn't leak.
    """
    # Remove script, style, svg blocks and their contents
    text = _re.sub(r"<(?:script|style|svg)[^>]*>.*?</(?:script|style|svg)>", " ", html, flags=_re.DOTALL)
    # Remove remaining HTML tags
    text = _re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = _re.sub(r"\s+", " ", text)
    return text.strip()


async def _fetch_and_extract(url: str) -> str:
    """Fetch a URL, strip HTML, return the first ~4000 chars of readable text."""
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SDR-App/1.0)"},
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    text = _strip_html(resp.text)
    # Clamp to a reasonable chunk for the LLM
    return text[:4000]


@app.post("/api/enrich")
async def enrich(req: EnrichRequest):
    """Fetch a URL and extract structured prospect data via the LLM."""
    try:
        page_text = await _fetch_and_extract(req.url)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: HTTP {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")

    api_key = os.getenv("LLM_API_KEY", "")
    llm_base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
    if not api_key:
        raise HTTPException(status_code=400, detail="LLM_API_KEY not configured on backend")

    prompt = f"""\
URL: {req.url}

PAGE CONTENT:
{page_text}

Extract prospect/sales information from the page content above.

Return ONLY valid JSON with exactly these keys (all strings, use "" if unknown):
- prospect_first_name: the person's first name
- prospect_last_name: the person's last name
- prospect_title: their job title
- company_name: their company
- prospect_bio: 1-2 sentence bio
- recent_linkedin_activity: any recent posts/activity mentioned
- company_news: recent company news (funding, hiring, products)
- inferred_value_prop: what the company does/sells (1 sentence)

If this is a company page with no specific person, set person fields to "" and fill company_name, company_news, inferred_value_prop.
Do NOT copy any existing JSON or structured data from the page — synthesize your own from the text."""

    try:
        content = await _call_llm(prompt, api_key, llm_base_url, model, max_tokens=600, retries=2)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI enrichment failed: {e}")
    # Parse the LLM output — it should be JSON
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines.pop(0)
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # LLM didn't return valid JSON; return as raw text
        return {
            "prospect_first_name": "",
            "prospect_last_name": "",
            "prospect_title": "",
            "company_name": "",
            "prospect_bio": "",
            "recent_linkedin_activity": "",
            "company_news": "",
            "inferred_value_prop": "",
            "_raw": cleaned,
        }


# ── LLM helpers ─────────────────────────────────────────────────────────


def _resolve_config(req: GenerateRequest):
    """Resolve API key, base URL, and model from request or env."""
    api_key = req.api_key or os.getenv("LLM_API_KEY", "")
    if not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="API key is required — set LLM_API_KEY in backend .env",
        )
    llm_base_url = req.llm_base_url or os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = req.model or os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
    return api_key, llm_base_url, model


async def _call_llm(prompt: str, api_key: str, llm_base_url: str, model: str, max_tokens: int = 500, retries: int = 1):
    """Make a single LLM call with timeout and automatic retries on transient errors."""
    url = f"{llm_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }

    last_error = None
    for attempt in range(retries + 1):
        if attempt > 0:
            wait = 2 ** attempt  # exponential backoff: 2s, 4s
            await asyncio.sleep(wait)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(65.0, connect=10.0)) as client:
                resp = await client.post(url, headers=headers, json=body)

            if resp.status_code in (429, 502, 503, 504) and attempt < retries:
                last_error = resp.status_code
                continue

            if resp.status_code != 200:
                detail = "LLM request failed"
                try:
                    detail = resp.json().get("error", {}).get("message", detail)
                except Exception:
                    pass
                raise HTTPException(status_code=resp.status_code, detail=detail)

            data = resp.json()

            if "choices" not in data or not data["choices"]:
                err_msg = "LLM returned an empty response"
                if "error" in data:
                    err_msg = data["error"].get("message", err_msg)
                raise HTTPException(status_code=502, detail=err_msg)

            return data["choices"][0]["message"]["content"]

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = e
            if attempt < retries:
                continue
            detail = (
                "LLM timed out after 65 seconds. "
                "The free OpenRouter model can be slow on cold starts. "
                "Try again, or switch to a faster model (gpt-4o-mini, claude-haiku) in the backend .env file."
                if isinstance(e, httpx.TimeoutException)
                else f"LLM unreachable after {retries + 1} attempts. Check your LLM base URL in .env."
            )
            raise HTTPException(status_code=504, detail=detail)

    raise HTTPException(
        status_code=502,
        detail=f"LLM request failed after {retries} retries (last status: {last_error})",
    )


def _parse_email_response(content: str) -> dict:
    """Parse LLM output into {subject_line, email_body}."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines.pop(0)
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        return {
            "subject_line": parsed.get("subject_line", ""),
            "email_body": parsed.get("email_body", cleaned),
        }
    except json.JSONDecodeError:
        lines = cleaned.split("\n", 1)
        subject_line = lines[0].strip().replace("Subject:", "").strip() if len(lines) > 0 else "SDR Outreach"
        email_body = lines[1] if len(lines) > 1 else cleaned
        return {"subject_line": subject_line, "email_body": email_body}


# ── Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Proxy an LLM call server-to-server (avoids browser CORS blocks)."""
    api_key, llm_base_url, model = _resolve_config(req)
    content = await _call_llm(req.prompt, api_key, llm_base_url, model)
    return _parse_email_response(content)


@app.post("/api/generate-sequence")
async def generate_sequence(req: GenerateRequest):
    """Generate a 3-email follow-up sequence from one prompt.

    Makes 3 sequential LLM calls:
      1. First email — the original SDR prompt
      2. Follow-up — brief follow-up referencing the first
      3. Break-up — final short email
    """
    api_key, llm_base_url, model = _resolve_config(req)
    base_prompt = req.prompt

    # Step 1: First email
    first_content = await _call_llm(base_prompt, api_key, llm_base_url, model)
    first = _parse_email_response(first_content)

    # Step 2: Follow-up (3-day framing)
    follow_prompt = f"""Write a follow-up email to the same prospect as the email below.
Reference the previous email briefly. Keep it short. New subject line.
Make it sound like 3 days have passed since the first email.

Previous email subject: {first["subject_line"]}

Return ONLY valid JSON with "subject_line" and "email_body"."""
    follow_content = await _call_llm(follow_prompt, api_key, llm_base_url, model, max_tokens=400)
    follow = _parse_email_response(follow_content)

    # Step 3: Break-up (5-day framing)
    break_prompt = f"""Write a short break-up email to the same prospect.
Let them know you will stop following up unless they reply, but leave the door open.
Make it sound like 5 days have passed since the first email.
New subject line.

Previous email subject: {first["subject_line"]}

Return ONLY valid JSON with "subject_line" and "email_body"."""
    break_content = await _call_llm(break_prompt, api_key, llm_base_url, model, max_tokens=300)
    break_ = _parse_email_response(break_content)

    return {
        "emails": [
            {"step": 1, "label": "First Email", "subject_line": first["subject_line"], "email_body": first["email_body"]},
            {"step": 2, "label": "Follow-Up", "subject_line": follow["subject_line"], "email_body": follow["email_body"]},
            {"step": 3, "label": "Break-Up", "subject_line": break_["subject_line"], "email_body": break_["email_body"]},
        ]
    }


# ── Runner ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
