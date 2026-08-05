"""
SDR App Backend — Minimal API key vault with PIN auth.

Stores the user's LLM API key encrypted on disk. The app fetches the key
at startup by presenting the PIN, then calls the LLM directly from the device.
"""

import asyncio
import collections
import ipaddress
import os
import json
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cryptography.fernet import Fernet
import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

load_dotenv()

app = FastAPI(title="SDR Key Vault")

# Allow the Flutter app to call the API.
# allow_credentials=True + "*" is not allowed by CORS, and this API is
# token- (not cookie-) based, so credentials aren't needed here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ──────────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
DATA_FILE = Path(os.getenv("DATA_FILE", "key_store.dat"))

# Client gate: the Flutter app must send this header or requests are rejected.
# Set in backend .env (and on SnapDeploy). Empty string disables the check
# for local convenience — leave disabled only on a local-only backend.
CLIENT_SECRET = os.getenv("SDR_CLIENT_SECRET", "").strip()

# Rate limiting / PIN brute-force protection.
# A small fixed window is used so a rotating IP can't brute-force the PIN.
LOGIN_WINDOW_SEC = float(os.getenv("LOGIN_WINDOW_SEC", "60"))
LOGIN_MAX_ATTEMPTS = int(os.getenv("LOGIN_MAX_ATTEMPTS", "5"))
PIN_LOCKOUT_SEC = int(os.getenv("PIN_LOCKOUT_SEC", "300"))  # 5 min after too many failures
RATE_LIMIT_WINDOW_SEC = float(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_MAX", "20"))

# In-memory tracking (per-process; fine for a single container).
_login_attempts: dict[str, list[float]] = {}   # ip -> timestamps of key/setup calls
_invalid_pin: dict[str, tuple[list[float], float]] = {}  # ip -> (failure times, lockout_until)
_api_hits: dict[str, collections.deque] = {}   # ip -> sliding window of timestamps
_rate_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    """Best-effort client IP (handles common configured proxy headers)."""
    for header in ("x-forwarded-for", "x-real-ip"):
        val = request.headers.get(header)
        if val:
            return val.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _locked_out(ip: str, now: float) -> bool:
    with _rate_lock:
        entry = _invalid_pin.get(ip)
        if not entry:
            return False
        failures, until = entry
        if now < until:
            return True
        # Lockout expired — clear it.
        del _invalid_pin[ip]
        return False


def _enforce_login_limit(ip: str, now: float):
    """Rate-limit /api/setup and /api/key per IP."""
    with _rate_lock:
        times = _login_attempts.setdefault(ip, [])
        times[:] = [t for t in times if now - t < LOGIN_WINDOW_SEC]
        if len(times) >= LOGIN_MAX_ATTEMPTS:
            raise HTTPException(
                status_code=429,
                detail="Too many attempts. Try again later.",
            )
        times.append(now)


def _record_failed_pin(ip: str, now: float):
    """Track consecutive PIN failures; lock the IP after too many."""
    with _rate_lock:
        failures, _until = _invalid_pin.get(ip, ([], 0.0))
        failures[:] = [t for t in failures if now - t < PIN_LOCKOUT_SEC]
        failures.append(now)
        # Lock if 5 failures within the window.
        if len(failures) >= 5:
            _invalid_pin[ip] = (failures, now + PIN_LOCKOUT_SEC)


def _enforce_rate_limit(ip: str, now: float):
    """Generic sliding-window limiter for the LLM-priced endpoints."""
    with _rate_lock:
        dq = _api_hits.setdefault(ip, collections.deque())
        while dq and now - dq[0] > RATE_LIMIT_WINDOW_SEC:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later.",
            )
        dq.append(now)

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
def setup(req: SetupRequest, request: Request):
    ip = _client_ip(request)
    _enforce_login_limit(ip, time.time())
    if _locked_out(ip, time.time()):
        raise HTTPException(status_code=429, detail="Too many failures. Locked out.")
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
def get_key(req: KeyRequest, request: Request):
    ip = _client_ip(request)
    _enforce_login_limit(ip, time.time())
    if _locked_out(ip, time.time()):
        raise HTTPException(status_code=429, detail="Too many failures. Locked out.")
    store = _load_store()
    if store is None:
        raise HTTPException(status_code=404, detail="No key stored. Call /api/setup first.")

    key = _derive_key(req.pin, store["salt"])
    fernet = Fernet(key)
    try:
        decrypted = fernet.decrypt(store["raw_encrypted"])
    except Exception:
        _record_failed_pin(ip, time.time())
        raise HTTPException(status_code=403, detail="Invalid PIN")

    return {"api_key": decrypted.decode()}


# ── Security guards ─────────────────────────────────────────────────────


def _require_client_secret(request: Request):
    """Reject requests that don't carry the shared client secret header.

    Only enforces when a secret is configured (local dev without one is allowed).
    """
    if CLIENT_SECRET and request.headers.get("X-SDR-Client-Secret", "") != CLIENT_SECRET:
        raise HTTPException(status_code=401, detail="Missing or invalid client secret")


def _is_ssrf_blocked(url: str) -> bool:
    """Block SSRF: non-http(s) schemes, and any host resolving to a non-public address.

    Looks at the literal host and (for hostnames) resolves via the system resolver,
    so a name like 'localhost', 'metadata.google.internal', or '0.0.0.0' is caught
    even if it isn't a raw IP. 172.16/12, 10/8, 192.168/16, loopback, link-local,
    CGNAT, and metadata subnets are all treated as internal.
    """
    try:
        parsed = httpx.URL(url)
    except Exception:
        return True
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.host or ""
    if not host:
        return True

    def _blocked(ip_str: str) -> bool:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or (ip.version == 6 and ip.is_site_local)
            or (ip.version == 6 and ip.ipv4_mapped is not None and _blocked(str(ip.ipv4_mapped)))
        )

    # Literal IP host — check directly.
    try:
        if _blocked(host):
            return True
        return False
    except ValueError:
        pass

    # Hostname — resolve every A/AAAA record and block if any is internal.
    import socket

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except Exception:
        return True  # failed resolution → treat as blocked
    for info in infos:
        addr = info[4][0]
        try:
            if _blocked(addr):
                return True
        except ValueError:
            return True
    return False


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


# Max bytes we're willing to download from an arbitrary URL before clipping.
_MAX_DOWNLOAD_BYTES = 512 * 1024  # 512 KB


async def _fetch_and_extract(url: str) -> str:
    """Fetch a URL, strip HTML, return the first ~4000 chars of readable text."""
    if _is_ssrf_blocked(url):
        raise HTTPException(status_code=400, detail="URL must be a public http(s) address")
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(15.0, connect=10.0),
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; SDR-App/1.0)"},
    ) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            raw = b""
            async for chunk in resp.aiter_bytes():
                raw += chunk
                if len(raw) > _MAX_DOWNLOAD_BYTES:
                    break
    text = _strip_html(raw.decode("utf-8", errors="ignore"))
    # Clamp to a reasonable chunk for the LLM
    return text[:4000]


@app.post("/api/enrich")
async def enrich(req: EnrichRequest, request: Request):
    """Fetch a URL and extract structured prospect data via the LLM."""
    _require_client_secret(request)
    _enforce_rate_limit(_client_ip(request), time.time())
    try:
        page_text = await _fetch_and_extract(req.url)
    except HTTPException as e:
        raise  # pass through SSRF block (400) as-is
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: HTTP {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {e}")

    api_key, llm_base_url, model = _resolve_config(req)

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
    """Resolve API key, base URL, and model from env ONLY.

    Caller-supplied values (api_key / llm_base_url / model) are intentionally
    ignored so the backend can't be pointed at an attacker-controlled endpoint.
    """
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="API key is required — set LLM_API_KEY in backend .env",
        )
    llm_base_url = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("LLM_MODEL", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
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
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
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
async def generate(req: GenerateRequest, request: Request):
    """Proxy an LLM call server-to-server (avoids browser CORS blocks)."""
    _require_client_secret(request)
    _enforce_rate_limit(_client_ip(request), time.time())
    api_key, llm_base_url, model = _resolve_config(req)
    content = await _call_llm(req.prompt, api_key, llm_base_url, model)
    return _parse_email_response(content)


@app.post("/api/generate-sequence")
async def generate_sequence(req: GenerateRequest, request: Request):
    """Generate a 3-email follow-up sequence from one prompt.

    Makes 3 sequential LLM calls:
      1. First email — the original SDR prompt
      2. Follow-up — brief follow-up referencing the first
      3. Break-up — final short email
    """
    _require_client_secret(request)
    _enforce_rate_limit(_client_ip(request), time.time())
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
