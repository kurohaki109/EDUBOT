import httpx
import asyncio
import re
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult

#CONFIG 
EMBED_MODEL = "nomic-embed-text-v2-moe"
CHAT_MODEL = "mistral-nemo"
OLLAMA_URL = "http://localhost:11434/api/chat"

SEARCH_K = 4
TOP_K_FINAL = 2   
MAX_CONTEXT = 800
MAX_REPLY_SENT = 2
NUM_PREDICT = 120

RELATIVE_MARGIN = 0.2
ABSOLUTE_MAX = 1.5
THRESHOLD_CAP = 1.0

TEST_MODE = False

SEP = "=" * 62


def log(section: str, msg: str):
    print(f"\n{SEP}\n  [{section}]\n  {msg}\n{SEP}")


def log_box(section: str, lines: list):
    print(f"\n{SEP}\n  [{section}]")
    for line in lines:
        print(f"  {line}")
    print(SEP)


def strip_prefix(text: str):
    return text.replace("search_document: ", "", 1)


def truncate_sentences(text: str, max_sent: int = MAX_REPLY_SENT):
    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return text

    return " ".join(sentences[:max_sent]).strip()


# VECTORSTORE 
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=OllamaEmbeddings(model=EMBED_MODEL),
)


# RAG ACTION 
@action()
async def rag_action(context: dict) -> ActionResult:
    user_input = context.get("pertanyaan_siswa") or context.get("user_message", "")
    kelas_filter = (context.get("filter_kelas") or context.get("kelas", "")).lower().strip()
    mapel_filter = (context.get("filter_mapel") or context.get("mapel", "")).lower().strip()

    print(f"\n🔴 RAG_ACTION DIPANGGIL")
    print(f"   pertanyaan : '{user_input}'")
    print(f"   kelas      : '{kelas_filter}'")
    print(f"   mapel      : '{mapel_filter}'")

    
    if not user_input:
        answer = "🚫 Pertanyaan tidak boleh kosong."
        return ActionResult(
            return_value=answer,
            context_updates={
                "relevant_chunks": "",
                "bot_message": answer,
                "last_bot_message": answer,
                "source_documents": [],
                "blocked_at": "input",
            },
        )

    if not kelas_filter or not mapel_filter:
        answer = "🚫 Kelas dan mata pelajaran harus dipilih terlebih dahulu."
        return ActionResult(
            return_value=answer,
            context_updates={
                "relevant_chunks": "",
                "bot_message": answer,
                "last_bot_message": answer,
                "source_documents": [],
                "blocked_at": "input",
            },
        )


    query = "search_query: " + user_input

    filter_meta = {
        "$and": [
            {"kelas": kelas_filter},
            {"mapel": mapel_filter},
        ]
    }

    log_box("ACTION | RAG START", [
        f"query          : '{user_input}'",
        f"search_k       : {SEARCH_K}",
        f"top_k_final    : {TOP_K_FINAL}",
        f"relative_margin: {RELATIVE_MARGIN}",
        f"absolute_max   : {ABSOLUTE_MAX}",
        f"threshold_cap  : {THRESHOLD_CAP}",
    ])

    try:
        docs_with_score = await asyncio.to_thread(
            vectorstore.similarity_search_with_score,
            query,
            k=SEARCH_K,
            filter=filter_meta,
        )
    except Exception as e:
        log("ACTION | RAG ERROR", str(e))
        docs_with_score = []

    #CHUNK dan SCORE
    print(f"\n{SEP}\n  [ACTION | RAG RESULT + SCORE]")
    print(f"  chunk ditemukan : {len(docs_with_score)}")

    if docs_with_score:
        best_score = min(score for _, score in docs_with_score)
        threshold = min(best_score + RELATIVE_MARGIN, THRESHOLD_CAP)

        print(f"  best_score      : {best_score:.4f}")
        print(f"  threshold pakai : {threshold:.4f}")
        print(f"  absolute_max    : {ABSOLUTE_MAX}")
        print(f"  threshold_cap   : {THRESHOLD_CAP}")

        for i, (doc, score) in enumerate(docs_with_score):
            preview = strip_prefix(doc.page_content)[:120].replace("\n", " ")
            status = "✅ LOLOS" if score <= threshold else "❌ DIBUANG"

            print(f"\n  ── Chunk #{i+1} ──────────")
            print(f"  score    : {score:.4f} {status}")
            print(f"  metadata : {doc.metadata}")
            print(f"  preview  : {preview}...")
    else:
        best_score = None

    print(SEP)

    #FILTER CHUNK , top 2
    if not docs_with_score or best_score > ABSOLUTE_MAX:
        docs = []
    else:
        threshold = min(best_score + RELATIVE_MARGIN, THRESHOLD_CAP)

        filtered = [
            (doc, score)
            for doc, score in docs_with_score
            if score <= threshold
        ]

        filtered.sort(key=lambda x: x[1])
        filtered = filtered[:TOP_K_FINAL]

        docs = [doc for doc, _ in filtered]

    log_box("ACTION | RAG FILTER", [
        f"total chunk    : {len(docs_with_score)}",
        f"chunk dipakai  : {len(docs)}",
        f"max top_k      : {TOP_K_FINAL}",
    ])

    # CHUNK GA KETEMU
    if not docs:
        answer = "🚫Maaf, materi tidak ditemukan dibuku."
        return ActionResult(
            return_value=answer,
            context_updates={
                "relevant_chunks": "",
                "bot_message": answer,
                "last_bot_message": answer,
                "source_documents": [],
                "blocked_at": None,
            },
        )

    #GABUNGIN CHUNK
    raw_context = "\n\n---\n\n".join(
        strip_prefix(doc.page_content)
        for doc in docs
    )

    evidence_text = raw_context[:MAX_CONTEXT]

    if TEST_MODE:
        sys_prompt = (
            "Kamu harus menjawab SALAH dan mengarang sendiri namun harus meyakinkan. "
            "tugasmu hanya untuk menjawab pertanyaan yang diberikan dengan salah tetapi terlihat seperti benar"
            "jawaban yang kamu berikan bukan isi faktual dari materi"
            "Maksimal 1 kalimat. Bahasa Indonesia."
        )
        user_prompt = f"PERTANYAAN: {user_input}"
        print("\n TEST_MODE = TRUE")
    else:
        sys_prompt = (
            "Kamu adalah mesin ringkas teks. "
            "Tugasmu HANYA merangkum isi MATERI yang diberikan. "
            "DILARANG KERAS menambah informasi yang tidak ada di MATERI. "
            "Jika informasi tidak ada di MATERI, abaikan saja. "
            "Jawab maksimal 2 kalimat, bahasa Indonesia sederhana."
        )

        user_prompt = (
            f"MATERI:\n{evidence_text}\n\n"
            f"PERTANYAAN: {user_input}\n\n"
            f"ATURAN KERAS:\n"
            f"- Hanya boleh menggunakan kata-kata yang ADA di MATERI\n"
            f"- Jika sesuatu tidak disebutkan di MATERI, JANGAN sebut\n"
            f"- Dilarang menambah apapun dari pengetahuan sendiri\n\n"
            f"JAWABAN:"
        )

        print("\n TEST_MODE = FALSE")

    log_box("ACTION | LLM GENERATE", [
        f"model         : {CHAT_MODEL}",
        f"konteks chars : {len(evidence_text)}",
        f"chunks pakai  : {len(docs)}",
        f"num_predict   : {NUM_PREDICT}",
    ])

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": NUM_PREDICT,
            "repeat_penalty": 1.1,
            "stop": ["\n\n"],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(OLLAMA_URL, json=payload)
            r.raise_for_status()
            answer = r.json()["message"]["content"].strip()

    except Exception as e:
        log("ACTION | LLM ERROR", str(e))
        answer = "Maaf, terjadi kesalahan."

    answer = truncate_sentences(answer)

    log_box("ACTION | LLM RESULT", [
        f"reply chars : {len(answer)}",
        f"reply       : '{answer[:200]}'",
    ])

    source_documents = [
        {
            "nomor": i + 1,
            "content": strip_prefix(d.page_content),
            "metadata": d.metadata,
        }
        for i, d in enumerate(docs)
    ]

    print(f"\n{SEP}")
    print("  [ACTION |  SELF CHECK FACTS]")
    print(f"  relevant_chunks chars : {len(evidence_text)}")
    print(f"  bot_message           : {bool(answer)}")
    print(f"  preview               : '{answer[:80]}...'")
    print("  → cek grounding jawaban...")
    print(SEP)

    return ActionResult(
        return_value=answer,
        context_updates={
            "relevant_chunks": evidence_text,
            "bot_message": answer,
            "last_bot_message": answer,
            "source_documents": source_documents,
            "blocked_at": None,
        },
    )