import os
os.environ["RAGAS_MAX_WORKERS"] = "1"
os.environ["RAGAS_RUN_CONFIG_TIMEOUT"] = "600"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from datasets import Dataset
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from typing import List
import time

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_excel("hasil_jawaban.xlsx")

# ✅ Baca 2 chunk (chunk_1 dan chunk_2)
def build_contexts(row):
    contexts = []
    for col in ["chunk_1", "chunk_2"]:
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text:
                contexts.append(text)
    return contexts

df["contexts"] = df.apply(build_contexts, axis=1)

# Hapus baris yang konteksnya kosong
df = df[df["contexts"].apply(len) > 0].reset_index(drop=True)

print(f"Data loaded: {len(df)} baris")
print(f"Contoh jumlah chunk per baris: {df['contexts'].apply(len).value_counts().to_dict()}")

# ============================================================
# EMBEDDINGS — Nomic dengan prefix
# ============================================================
class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.model = OllamaEmbeddings(
            model="nomic-embed-text-v2-moe:latest",
            base_url="http://localhost:11434"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"search_document: {t}" for t in texts]
        return self.model.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"search_query: {text}"
        return self.model.embed_query(prefixed)

print("Embeddings ok")

# ============================================================
# LLM — Mistral Nemo dengan system prompt Bahasa Indonesia
# ✅ FIX: paksa output Bahasa Indonesia agar tidak terjadi
#    language mismatch saat RAGAS generate pertanyaan balik
# ============================================================
SYSTEM_PROMPT_ID = (
    "Kamu adalah asisten AI yang selalu berkomunikasi dalam Bahasa Indonesia. "
    "Saat diminta membuat pertanyaan, buat pertanyaan dalam Bahasa Indonesia. "
    "Saat diminta menganalisis teks, jawab dalam Bahasa Indonesia."
)

ollama_llm = OllamaLLM(
    model="mistral-nemo:latest",
    base_url="http://localhost:11434",
    temperature=0,
    timeout=600,
    system=SYSTEM_PROMPT_ID,   # ✅ Fix language mismatch
)
llm = LangchainLLMWrapper(ollama_llm)

print("LLM Mistral Nemo ok (dengan system prompt Bahasa Indonesia)")

# ============================================================
# SETUP METRIK RAGAS
# ============================================================
embeddings = NomicEmbeddings()

faithfulness.llm        = llm
answer_relevancy.llm    = llm
answer_relevancy.embeddings = embeddings   # wajib untuk cosine similarity
context_precision.llm   = llm
context_recall.llm      = llm

run_config = RunConfig(
    timeout=600,
    max_retries=3,
    max_workers=1
)

# ============================================================
# EVALUASI PER BARIS
# ============================================================
print(f"\nMemulai evaluasi RAGAS (2 chunk)")
print(f"Mulai: {time.strftime('%H:%M:%S')}")
print("-" * 90)
print(f"{'No':<5} {'Kelas':<10} {'Faith':>7} {'AnsRel':>7} {'CtxPre':>7} {'CtxRec':>7} {'Chunk':>5} {'Waktu':>7}")
print("-" * 90)

results = []
start_total = time.time()

for i, row in df.iterrows():
    start_row = time.time()

    n_chunks = len(row["contexts"])

    single_data = Dataset.from_dict({
        "question":     [str(row["pertanyaan"])],
        "answer":       [str(row["jawaban_llm"])],
        "contexts":     [row["contexts"]],
        "ground_truth": [str(row["kunci"])]
    })

    try:
        result = evaluate(
            single_data,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config
        )
        result_row = result.to_pandas().iloc[0]

        faith  = result_row.get("faithfulness",      float("nan"))
        ansrel = result_row.get("answer_relevancy",  float("nan"))
        ctxpre = result_row.get("context_precision", float("nan"))
        ctxrec = result_row.get("context_recall",    float("nan"))

    except Exception as e:
        print(f"  ⚠ Error baris {row['no']}: {e}")
        faith = ansrel = ctxpre = ctxrec = float("nan")

    elapsed_row = time.time() - start_row

    results.append({
        "no":                row["no"],
        "kelas":             row["kelas"],
        "pertanyaan":        row["pertanyaan"],
        "jawaban_llm":       row["jawaban_llm"],
        "n_chunks":          n_chunks,
        "faithfulness":      faith,
        "answer_relevancy":  ansrel,
        "context_precision": ctxpre,
        "context_recall":    ctxrec,
    })

    faith_str  = f"{faith:.3f}"  if faith  == faith  else " NaN"
    ansrel_str = f"{ansrel:.3f}" if ansrel == ansrel else " NaN"
    ctxpre_str = f"{ctxpre:.3f}" if ctxpre == ctxpre else " NaN"
    ctxrec_str = f"{ctxrec:.3f}" if ctxrec == ctxrec else " NaN"

    print(
        f"{row['no']:<5} {str(row['kelas']):<10} "
        f"{faith_str:>7} {ansrel_str:>7} {ctxpre_str:>7} {ctxrec_str:>7} "
        f"{n_chunks:>5} {elapsed_row:>5.0f}s"
    )

# ============================================================
# SIMPAN HASIL
# ============================================================
result_df = pd.DataFrame(results)
output_file = "hasil_evaluasi_ragas_2chunk.xlsx"
result_df.to_excel(output_file, index=False)

elapsed_total = time.time() - start_total
print("-" * 90)
print(f"Total waktu: {elapsed_total/3600:.1f} jam ({elapsed_total/60:.0f} menit)")

# ============================================================
# RINGKASAN STATISTIK
# ============================================================
metrics_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

print(f"\n{'='*50}")
print("RATA-RATA PER METRIK")
print(f"{'='*50}")
means = result_df[metrics_cols].mean()
for col, val in means.items():
    print(f"  {col:<22}: {val:.4f}")

ragas_score = means.mean()
print(f"\n  {'RAGAS Score':<22}: {ragas_score:.4f}")

print(f"\n{'='*50}")
print("RATA-RATA PER KELAS")
print(f"{'='*50}")
per_kelas = result_df.groupby("kelas")[metrics_cols].mean()
print(per_kelas.round(4).to_string())

print(f"\n{'='*50}")
print("STATISTIK DESKRIPTIF")
print(f"{'='*50}")
print(result_df[metrics_cols].describe().round(4).to_string())

print(f"\n✅ Hasil tersimpan di: {output_file}")