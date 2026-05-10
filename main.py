import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from nemoguardrails import LLMRails, RailsConfig
from database import conn, cursor


os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SEP = "=" * 62

NEMO_BLOCK_MESSAGES = [
    "I'm sorry, I can't respond to that.",
    "I'm sorry, an internal error has occurred.",
]


def log(section: str, msg: str):
    print(f"\n{SEP}\n  [{section}]\n  {msg}\n{SEP}")


def log_box(section: str, lines: list):
    print(f"\n{SEP}\n  [{section}]")
    for line in lines:
        print(f"  {line}")
    print(SEP)


def detect_blocked(text: str) -> bool:
    if not text:
        return True

    t = text.strip()

    if t.startswith("🚫"):
        return True

    if t in NEMO_BLOCK_MESSAGES:
        return True

    if "can't respond" in t.lower():
        return True

    if "internal error" in t.lower():
        return True

    return False


def load_html(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()

    return f"<h1>{filename} tidak ditemukan</h1>"


def load_nemo():
    try:
        config = RailsConfig.from_path("./config")
        rails = LLMRails(config)

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "nemo_config",
            "./config/config.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.init(rails)

        from config.actions import rag_action
        rails.register_action(rag_action, name="rag_action")

        log(
            "STARTUP | NEMO",
            (
                "NeMo aktif\n"
                "  input  : guardrails\n"
                "  output : self check facts\n"
                "  action : rag_action"
            ),
        )

        return rails

    except Exception as e:
        log("STARTUP | ERROR", str(e))
        return None


rails = load_nemo()
log("STARTUP | READY", "Server ok")

class ChatRequest(BaseModel):
    message: str
    kelas: str
    mapel: str


class UserRequest(BaseModel):
    username: str
    password: str


@app.get("/", response_class=HTMLResponse)
async def login_page():
    return load_html("login.html")


@app.get("/register-page", response_class=HTMLResponse)
async def register_page():
    return load_html("register.html")


@app.get("/chatbot", response_class=HTMLResponse)
async def chatbot_page():
    return load_html("index.html")


@app.post("/register")
async def register(req: UserRequest):
    username = req.username.strip()
    password = req.password.strip()

    if not(username):
        raise HTTPException(
            status_code=400,
            detail="Username tidak boleh kosong"
        )

    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password minimal 6 karakter"
        )

    cursor.execute(
        "SELECT id FROM users WHERE username=?",
        (username,)
    )

    if cursor.fetchone():
        raise HTTPException(
            status_code=400,
            detail="Username sudah dipakai"
        )

    cursor.execute(
        "INSERT INTO users(username,password) VALUES(?,?)",
        (username, password)
    )
    conn.commit()

    log("REGISTER", f"✅ user baru: {username}")

    return {
        "success": True,
        "message": "Register berhasil"
    }


@app.post("/login")
async def login(req: UserRequest):
    username = req.username.strip()
    password = req.password.strip()

    cursor.execute(
        "SELECT id FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Username / password salah"
        )

    log("LOGIN", f"✅ {username}")

    return {
        "success": True,
        "message": "Login berhasil",
        "username": username
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    t_total = time.perf_counter()

    user_input = req.message.strip()
    kelas = req.kelas.lower().strip()
    mapel = req.mapel.lower().strip()

    log_box("REQUEST MASUK", [
        f"message : '{user_input}'",
        f"kelas   : '{kelas}'",
        f"mapel   : '{mapel}'",
    ])

    if not rails:
        return JSONResponse({
            "reply": "🚫 Sistem belum siap.",
            "blocked": True,
            "blocked_at": "startup",
            "source_documents": [],
        })

    log("NEMO", "▶ generate_async dimulai...")
    t0 = time.perf_counter()

    try:
        result = await rails.generate_async(
            messages=[
                {
                    "role": "context",
                    "content": {
                        "kelas": kelas,
                        "mapel": mapel,
                        "user_message": user_input,
                    },
                },
                {
                    "role": "user",
                    "content": user_input,
                },
            ]
        )

    except Exception as e:
        log("NEMO ERROR", str(e))

        return JSONResponse({
            "reply": "🚫 Sistem error.",
            "blocked": True,
            "blocked_at": "nemo",
            "source_documents": [],
        })

    elapsed = time.perf_counter() - t0

    content = result.get("content", "").strip()
    is_blocked = detect_blocked(content)

    log_box("NEMO RESULT", [
        f"waktu      : {elapsed:.2f}s",
        f"content    : '{content[:200]}'",
        f"is_blocked : {is_blocked}",
    ])

    ctx = result.get("context", {})
    source_documents = ctx.get("source_documents", [])

    total = time.perf_counter() - t_total

    log_box("SELESAI", [
        f"total waktu : {total:.2f}s",
        f"blocked     : {is_blocked}",
        f"reply chars : {len(content)}",
    ])

    return JSONResponse({
        "reply": content,
        "blocked": is_blocked,
        "blocked_at": "output" if is_blocked else None,
        "source_documents": source_documents,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)