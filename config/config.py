from functools import wraps
import hashlib
import os
import pickle

import httpx
from nemoguardrails import LLMRails
from nemoguardrails.embeddings.providers import register_embedding_provider
from nemoguardrails.embeddings.providers.base import EmbeddingModel

CACHE_DIR = "./cache/nemo_embeddings"


class NomicV2OllamaEmbedding(EmbeddingModel):
    engine_name = "nomic_v2_ollama"

    def __init__(self, embedding_model: str):
        self.model = embedding_model
        self.base_url = "http://localhost:11434/api/embed"
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _cache_key(self, texts):
        content = str(sorted(texts)).encode()
        return hashlib.md5(content).hexdigest()

    def _load_cache(self, key):
        path = os.path.join(CACHE_DIR, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key, embeddings):
        path = os.path.join(CACHE_DIR, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(embeddings, f)

    def _embed(self, texts):
        key = self._cache_key(texts)

        cached = self._load_cache(key)
        if cached is not None:
            print(f"⚡ Cache hit — skip embedding {len(texts)} texts")
            return cached

        print(f"🔄 Embedding {len(texts)} texts via Ollama...")

        results = []
        for text in texts:
            r = httpx.post(
                self.base_url,
                json={
                    "model": self.model,
                    "input": text,
                },
                timeout=30,
            )
            r.raise_for_status()

            data = r.json()
            results.append(data["embeddings"][0])

        self._save_cache(key, results)
        print(f"✅ Cache disimpan: {CACHE_DIR}/{key}.pkl")

        return results

    async def encode_async(self, documents):
        prefixed = [f"search_document: {doc}" for doc in documents]
        return self._embed(prefixed)

    def encode(self, documents):
        prefixed = [f"search_document: {doc}" for doc in documents]
        return self._embed(prefixed)


def init(app: LLMRails):
    # ── Register embedding provider ────────────────────────────
    try:
        register_embedding_provider(NomicV2OllamaEmbedding)
        print("✅ NomicV2OllamaEmbedding didaftarkan")

    except Exception as e:
        if "already exists" in str(e):
            print("⚠️ NomicV2OllamaEmbedding sudah ada — skip")
        else:
            raise e

    # ── Patch self_check_facts (debug mode) ────────────────────
    from nemoguardrails.library.self_check.facts.actions import (
        self_check_facts as original_self_check_facts,
    )

    @wraps(original_self_check_facts)
    async def patched_check_facts(*args, **kwargs):
        print("\n" + "=" * 62)
        print("  [SELF CHECK FACTS | DIPANGGIL]")

        context = kwargs.get("context", {})
        last_bot_message = context.get("last_bot_message", "")

        if last_bot_message.strip().startswith("🚫"):
            print("  [SELF CHECK FACTS | SKIP] pesan diawali 🚫, skip check")
            print("=" * 62)
            return 1.0

        # log positional args
        print(f"  args len    : {len(args)}")
        for i, arg in enumerate(args):
            preview = str(arg)
            preview = preview.replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  arg[{i}]      : {preview}")

        # log kwargs
        print(f"  kwargs keys : {list(kwargs.keys())}")
        for k, v in kwargs.items():
            preview = str(v)
            preview = preview.replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"  {k:<12}: {preview}")

        try:
            # forward semua dependency internal NeMo
            result = await original_self_check_facts(*args, **kwargs)

            print(f"  hasil check : '{result}'")
            print("=" * 62)

            return result

        except Exception as e:
            print(f"  ERROR       : {e}")
            print("=" * 62)
            raise

    app.register_action(
        action=patched_check_facts,
        name="self_check_facts",
    )

    print("✅ self_check_facts patched (debug mode)")