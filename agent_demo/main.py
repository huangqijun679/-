import sqlite3
import uuid
import numpy as np
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import os
import sys
import subprocess
import json
from bs4 import BeautifulSoup
import urllib.parse
# from sentence_transformers import SentenceTransformer
from skills import SKILLS_REGISTRY
# 在本地加载一个轻量级的开源向量模型（第一次运行会自动下载，几十MB左右）
# local_embedder = SentenceTransformer('shibing624/text2vec-base-chinese')

DB_DIR = Path("data")
DB_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Agent Chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _envelope(data, request: Request | None = None):
    payload = {"ok": True, "data": data}
    if request is not None:
        payload["path"] = request.url.path
    return payload


def get_db():
    conn = sqlite3.connect(str(DB_DIR / "agent.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            provider TEXT PRIMARY KEY,
            key_value TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS model_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            provider TEXT NOT NULL DEFAULT 'openai',
            model TEXT NOT NULL DEFAULT 'gpt-4o-mini'
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            updated_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime')),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS rag_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB,
            FOREIGN KEY (doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS custom_providers (
            name TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            models TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE TABLE IF NOT EXISTS personas (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            prompt TEXT NOT NULL DEFAULT '',
            skills TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        );
        CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(doc_id);
    """)
    conn.execute(
        "INSERT OR IGNORE INTO model_config (id, provider, model) VALUES (1, 'openai', 'gpt-4o-mini')"
    )
    conn.commit()
    conn.close()


init_db()

PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "embedding_model": "text-embedding-ada-002",
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-chat", "deepseek-reasoner", "deepseek-flash", "deepseek-pro"],
        "default_model": "deepseek-chat",
        "embedding_model": "deepseek-embedding",
    },
}

# 模型别名映射，用于API调用时的模型名称转换
MODEL_ALIASES = {
    "deepseek-flash": "deepseek-chat",
    "deepseek-pro": "deepseek-reasoner",
}







class ChatRequest(BaseModel):
    session_id: str
    message: str
    use_rag: bool = True
    thinking_mode: bool = False
    skills: list[str] = []
    persona_id: str = ""


class APIKeyRequest(BaseModel):
    key: str
    provider: str = "openai"


class ModelConfigRequest(BaseModel):
    provider: str
    model: str


class CustomProviderRequest(BaseModel):
    name: str
    display_name: str
    base_url: str
    models: str


class PersonaCreateRequest(BaseModel):
    name: str
    prompt: str = ""
    skills: str = ""


class PersonaUpdateRequest(BaseModel):
    name: str
    prompt: str = ""
    skills: str = ""


def get_api_key(provider: str = "openai"):
    conn = get_db()
    row = conn.execute("SELECT key_value FROM api_keys WHERE provider=?", (provider,)).fetchone()
    conn.close()
    return row["key_value"] if row else None


def embed_texts(texts: list[str], api_key: str, base_url: str = None, model: str = None) -> list[list[float]]:
    if base_url is None:
        base_url = "https://api.openai.com/v1"
    if model is None:
        model = "text-embedding-ada-002"
    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in resp.data]


def get_embedding_config():
    """获取当前供应商的嵌入配置"""
    config = get_active_model_config()
    provider = config["provider"]
    
    # 获取供应商信息
    all_providers = get_all_providers()
    if provider in all_providers:
        provider_info = all_providers[provider]
        embedding_model = provider_info.get("embedding_model")
        base_url = provider_info.get("base_url")
        
        # 获取API密钥
        key_provider = provider if provider.startswith("custom_") else provider
        api_key = get_api_key(key_provider)
        
        if embedding_model and api_key:
            return {
                "api_key": api_key,
                "base_url": base_url,
                "model": embedding_model,
                "provider": provider
            }
    
    # 尝试使用任何已配置密钥的供应商
    # 优先使用有embedding_model的供应商
    for provider_name, provider_info in all_providers.items():
        embedding_model = provider_info.get("embedding_model")
        if not embedding_model:
            continue
        
        key_provider = provider_name if provider_name.startswith("custom_") else provider_name
        api_key = get_api_key(key_provider)
        
        if api_key:
            return {
                "api_key": api_key,
                "base_url": provider_info.get("base_url"),
                "model": embedding_model,
                "provider": provider_name
            }
    
    # 回退到OpenAI
    openai_key = get_api_key("openai")
    if openai_key:
        return {
            "api_key": openai_key,
            "base_url": "https://api.openai.com/v1",
            "model": "text-embedding-ada-002",
            "provider": "openai"
        }
    
    return None


def cosine_sim(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_custom_providers():
    conn = get_db()
    rows = conn.execute("SELECT * FROM custom_providers").fetchall()
    conn.close()
    result = {}
    for r in rows:
        models = [m.strip() for m in r["models"].split(",") if m.strip()]
        result[f"custom_{r['name']}"] = {"name": r["display_name"], "base_url": r["base_url"], "models": models}
    return result


def get_provider_base_url(provider: str) -> str | None:
    if provider in PROVIDERS:
        return PROVIDERS[provider]["base_url"]
    custom = get_custom_providers()
    if provider in custom:
        return custom[provider]["base_url"]
    return None


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def get_all_providers():
    merged = dict(PROVIDERS)
    merged.update(get_custom_providers())
    return merged


def get_active_model_config():
    conn = get_db()
    row = conn.execute("SELECT provider, model FROM model_config WHERE id=1").fetchone()
    conn.close()
    if row:
        return {"provider": row["provider"], "model": row["model"]}
    return {"provider": "openai", "model": "gpt-4o-mini"}


def set_active_model_config(provider: str, model: str):
    all_providers = get_all_providers()
    if provider not in all_providers:
        raise HTTPException(400, f"Unknown provider: {provider}")
    provider_info = all_providers[provider]
    if model not in provider_info["models"]:
        raise HTTPException(400, f"Model {model} not available for {provider}")
    conn = get_db()
    conn.execute("UPDATE model_config SET provider=?, model=? WHERE id=1", (provider, model))
    conn.commit()
    conn.close()


def resolve_persona(persona_id: str) -> tuple[str, list[str]]:
    if not persona_id:
        return "", []
    conn = get_db()
    persona = conn.execute("SELECT * FROM personas WHERE id=?", (persona_id,)).fetchone()
    conn.close()
    if persona and persona["prompt"].strip():
        prompt = persona["prompt"].strip()
        skills = [s.strip() for s in persona["skills"].split(",") if s.strip()]
        return prompt, skills
    return "", []


def build_agent(tools: list, system_prompt: str, api_key: str, base_url: str, model: str):
    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0.7)
    return create_react_agent(llm, tools, prompt=system_prompt)


@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# --- API Keys ---
@app.post("/api/keys/clear-all")
async def clear_all_keys():
    """一键清除所有API密钥"""
    conn = get_db()
    cursor = conn.execute("DELETE FROM api_keys")
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return {"status": "ok", "deleted": deleted_count}


@app.get("/api/keys/{provider}")
async def check_key(provider: str):
    return {"configured": get_api_key(provider) is not None}


@app.post("/api/keys/{provider}")
async def save_key(provider: str, req: APIKeyRequest):
    is_custom = provider.startswith("custom_")
    if not is_custom and not req.key.startswith("sk-"):
        raise HTTPException(400, "Invalid API key format (must start with sk-)")
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO api_keys (provider, key_value) VALUES (?, ?)",
        (provider, req.key),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/keys/{provider}")
async def delete_key(provider: str):
    conn = get_db()
    conn.execute("DELETE FROM api_keys WHERE provider=?", (provider,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# --- Skills ---
@app.get("/api/skills")
async def list_skills():
    return [{"name": name, "description": tool.description} for name, tool in SKILLS_REGISTRY.items()]


# --- Providers & Model Config ---
@app.get("/api/models")
async def list_models():
    return get_all_providers()


@app.get("/api/active-model")
async def get_active_model():
    return get_active_model_config()


@app.post("/api/active-model")
async def set_active_model(req: ModelConfigRequest):
    set_active_model_config(req.provider, req.model)
    return {"status": "ok"}


# --- Custom Providers ---
@app.get("/api/custom-providers")
async def list_custom_providers():
    return get_custom_providers()


@app.post("/api/custom-providers")
async def create_custom_provider(req: CustomProviderRequest):
    if not req.name or not req.base_url or not req.models:
        raise HTTPException(400, "name, base_url, and models are required")
    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO custom_providers (name, display_name, base_url, models) VALUES (?, ?, ?, ?)",
            (req.name, req.display_name, req.base_url, req.models),
        )
        conn.commit()
    except Exception as e:
        raise HTTPException(400, f"Failed to create provider: {e}")
    finally:
        conn.close()
    return {"status": "ok"}


@app.put("/api/custom-providers/{name}")
async def update_custom_provider(name: str, req: CustomProviderRequest):
    conn = get_db()
    row = conn.execute("SELECT 1 FROM custom_providers WHERE name=?", (name,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Provider not found")
    conn.execute(
        "UPDATE custom_providers SET display_name=?, base_url=?, models=? WHERE name=?",
        (req.display_name, req.base_url, req.models, name),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/custom-providers/{name}")
async def delete_custom_provider(name: str):
    conn = get_db()
    # 先检查是否存在
    row = conn.execute("SELECT 1 FROM custom_providers WHERE name=?", (name,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Provider not found")
    
    # 删除供应商和对应的API密钥
    conn.execute("DELETE FROM custom_providers WHERE name=?", (name,))
    conn.execute("DELETE FROM api_keys WHERE provider=?", (f"custom_{name}",))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# --- Personas ---
@app.get("/api/personas")
async def list_personas():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, name, prompt, skills, created_at FROM personas ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/personas")
async def create_persona(req: PersonaCreateRequest):
    if not req.name.strip():
        raise HTTPException(400, "Name is required")
    pid = uuid.uuid4().hex[:8]
    conn = get_db()
    conn.execute(
        "INSERT INTO personas (id, name, prompt, skills) VALUES (?, ?, ?, ?)",
        (pid, req.name.strip(), req.prompt.strip(), req.skills.strip()),
    )
    conn.commit()
    conn.close()
    return {"id": pid, "name": req.name.strip(), "prompt": req.prompt.strip(), "skills": req.skills.strip()}


@app.put("/api/personas/{pid}")
async def update_persona(pid: str, req: PersonaUpdateRequest):
    if not req.name.strip():
        raise HTTPException(400, "Name is required")
    conn = get_db()
    row = conn.execute("SELECT 1 FROM personas WHERE id=?", (pid,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Persona not found")
    conn.execute(
        "UPDATE personas SET name=?, prompt=?, skills=? WHERE id=?",
        (req.name.strip(), req.prompt.strip(), req.skills.strip(), pid),
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/api/personas/{pid}")
async def delete_persona(pid: str):
    conn = get_db()
    conn.execute("DELETE FROM personas WHERE id=?", (pid,))
    affected = conn.total_changes
    conn.commit()
    conn.close()
    if not affected:
        raise HTTPException(404, "Persona not found")
    return {"status": "ok"}


# --- Sessions ---
@app.get("/api/sessions")
async def list_sessions():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/api/sessions")
async def create_session():
    session_id = uuid.uuid4().hex[:8]
    conn = get_db()
    conn.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
    conn.commit()
    conn.close()
    return {"id": session_id, "title": "New Chat"}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Chat ---
@app.post("/api/chat")
async def chat(req: ChatRequest):
    conn = get_db()

    config_row = conn.execute("SELECT provider, model FROM model_config WHERE id=1").fetchone()
    provider = config_row["provider"] if config_row else "openai"
    model = config_row["model"] if config_row else "gpt-4o-mini"

    key_provider = provider if provider.startswith("custom_") else (f"custom_{provider}" if provider not in PROVIDERS else provider)
    api_key = get_api_key(key_provider)
    if not api_key:
        pname = PROVIDERS.get(provider, {}).get("name", provider)
        if not pname:
            custom = get_custom_providers()
            pname = custom.get(provider, {}).get("name", provider)
        raise HTTPException(400, f"Please configure your {pname} API key first")

    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)",
        (req.session_id, req.message),
    )

    msg_count = conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE session_id=?", (req.session_id,)
    ).fetchone()["cnt"]

    if msg_count == 1:
        title = (req.message[:50] + "...") if len(req.message) > 50 else req.message
        conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, req.session_id))

    conn.commit()

    rag_context = ""
    if req.use_rag:
        embedding_config = get_embedding_config()
        doc_count = conn.execute("SELECT COUNT(*) as cnt FROM rag_documents").fetchone()["cnt"]
        if doc_count > 0 and embedding_config:
            try:
                query_emb = embed_texts(
                    [req.message],
                    embedding_config["api_key"],
                    embedding_config["base_url"],
                    embedding_config["model"]
                )[0]
                chunks = conn.execute("SELECT content, embedding FROM rag_chunks").fetchall()
                scored = []
                for ch in chunks:
                    emb = np.frombuffer(ch["embedding"], dtype=np.float32)
                    sim = cosine_sim(query_emb, emb)
                    scored.append((sim, ch["content"]))
                scored.sort(key=lambda x: -x[0])
                top = scored[:3]
                if top and top[0][0] > 0.3:
                    rag_context = "\n\nRelevant context:\n" + "\n---\n".join(t[1] for t in top)
            except Exception:
                rag_context = ""

    conn.close()

    persona_prompt, persona_skill_names = resolve_persona(req.persona_id)

    all_skill_names = list(set(req.skills + persona_skill_names))
    enabled_tools = [SKILLS_REGISTRY[name] for name in all_skill_names if name in SKILLS_REGISTRY]

    try:
        # 获取实际的API模型名称（处理别名）
        actual_model = MODEL_ALIASES.get(model, model)
        
        if req.thinking_mode:
            ds_key = get_api_key("deepseek")
            if not ds_key:
                raise HTTPException(400, "Thinking mode requires a DeepSeek API key")
            client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
            model_kwargs = {"model": "deepseek-reasoner", "max_tokens": 8192}
        else:
            base_url = get_provider_base_url(provider)
            if not base_url:
                raise HTTPException(400, f"Unknown provider: {provider}")
            client = OpenAI(api_key=api_key, base_url=base_url)
            model_kwargs = {"model": actual_model, "temperature": 0.7}

        system_prompt = (persona_prompt if persona_prompt else "You are a helpful AI assistant.") + rag_context

        if enabled_tools and not req.thinking_mode:
            agent = build_agent(enabled_tools, system_prompt, api_key, base_url, actual_model)
            result = agent.invoke({"messages": [HumanMessage(content=req.message)]})
            last_msg = result["messages"][-1]
            reply = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            reasoning = None
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.message},
            ]
            resp = client.chat.completions.create(**model_kwargs, messages=messages)
            reply = resp.choices[0].message.content
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None)

        conn = get_db()
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)",
            (req.session_id, reply),
        )
        conn.execute("UPDATE sessions SET updated_at=datetime('now','localtime') WHERE id=?", (req.session_id,))
        conn.commit()
        conn.close()
        return {"reply": reply, "reasoning": reasoning}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# --- RAG ---
@app.post("/api/rag/upload")
async def upload_document(file: UploadFile = File(...)):
    embedding_config = get_embedding_config()
    if not embedding_config:
        raise HTTPException(400, "需要配置API密钥用于生成嵌入向量")

    content = (await file.read()).decode("utf-8", errors="replace")
    if not content.strip():
        raise HTTPException(400, "File is empty")

    title = file.filename or "Untitled"
    chunks = split_text(content)

    conn = get_db()
    cursor = conn.execute("INSERT INTO rag_documents (title) VALUES (?)", (title,))
    doc_id = cursor.lastrowid

    if chunks:
        embeddings = embed_texts(
            chunks,
            embedding_config["api_key"],
            embedding_config["base_url"],
            embedding_config["model"]
        )
        for chunk, emb in zip(chunks, embeddings):
            conn.execute(
                "INSERT INTO rag_chunks (doc_id, content, embedding) VALUES (?, ?, ?)",
                (doc_id, chunk, np.array(emb, dtype=np.float32).tobytes()),
            )

    conn.commit()
    conn.close()
    return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}


@app.get("/api/rag/documents")
async def list_documents():
    conn = get_db()
    rows = conn.execute("""
        SELECT d.id, d.title, d.created_at, COUNT(c.id) as chunks
        FROM rag_documents d LEFT JOIN rag_chunks c ON d.id=c.doc_id
        GROUP BY d.id ORDER BY d.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.delete("/api/rag/documents/{doc_id}")
async def delete_document(doc_id: int):
    conn = get_db()
    conn.execute("DELETE FROM rag_chunks WHERE doc_id=?", (doc_id,))
    conn.execute("DELETE FROM rag_documents WHERE id=?", (doc_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# --- Memory ---
class MemorySaveRequest(BaseModel):
    session_ids: list[str]


@app.post("/api/memory/save")
async def save_memory(req: MemorySaveRequest):
    embedding_config = get_embedding_config()
    if not embedding_config:
        raise HTTPException(400, "需要配置API密钥用于生成嵌入向量")
    
    conn = get_db()
    total_chunks = 0
    total_messages = 0
    
    for session_id in req.session_ids:
        # 获取会话信息
        session = conn.execute("SELECT title FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            continue
        
        # 获取会话的所有消息
        messages = conn.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
            (session_id,)
        ).fetchall()
        
        if not messages:
            continue
        
        total_messages += len(messages)
        
        # 将对话内容格式化为文本
        conversation_text = f"对话标题: {session['title']}\n\n"
        for msg in messages:
            role = "用户" if msg["role"] == "user" else "助手"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        # 分块
        chunks = split_text(conversation_text)
        
        # 生成嵌入向量并存储
        if chunks:
            try:
                embeddings = embed_texts(
                    chunks,
                    embedding_config["api_key"],
                    embedding_config["base_url"],
                    embedding_config["model"]
                )
                
                # 创建记忆文档
                cursor = conn.execute(
                    "INSERT INTO rag_documents (title) VALUES (?)",
                    (f"[记忆] {session['title']}",)
                )
                doc_id = cursor.lastrowid
                
                # 存储分块和嵌入向量
                for chunk, emb in zip(chunks, embeddings):
                    conn.execute(
                        "INSERT INTO rag_chunks (doc_id, content, embedding) VALUES (?, ?, ?)",
                        (doc_id, chunk, np.array(emb, dtype=np.float32).tobytes())
                    )
                
                total_chunks += len(chunks)
            except Exception as e:
                print(f"Error processing session {session_id}: {e}")
                continue
    
    conn.commit()
    conn.close()
    
    return {"status": "ok", "chunks": total_chunks, "messages": total_messages}


@app.get("/api/memory/documents")
async def list_memory_documents():
    conn = get_db()
    rows = conn.execute("""
        SELECT d.id, d.title, d.created_at, COUNT(c.id) as chunks
        FROM rag_documents d LEFT JOIN rag_chunks c ON d.id=c.doc_id
        WHERE d.title LIKE '[记忆]%'
        GROUP BY d.id ORDER BY d.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]