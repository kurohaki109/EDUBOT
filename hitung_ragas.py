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

df = pd.read_excel("hasil_jawaban.xlsx")

df["contexts"] = df.apply(
    lambda row: [
        str(row["chunk_1"]) if pd.notna(row["chunk_1"]) else "",
        str(row["chunk_2"]) if pd.notna(row["chunk_2"]) else ""
    ], axis=1
)
df["contexts"] = df["contexts"].apply(lambda x: [c for c in x if c.strip()])

print(f"Data loaded: {len(df)} baris")

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

ollama_llm = OllamaLLM(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0,
    timeout=600
)
llm = LangchainLLMWrapper(ollama_llm)

print("LLM qwen ok")


embeddings = NomicEmbeddings()

faithfulness.llm = llm
answer_relevancy.llm = llm
answer_relevancy.embeddings = embeddings
context_precision.llm = llm
context_recall.llm = llm


run_config = RunConfig(
    timeout=600,
    max_retries=3,
    max_workers=1
)


print(f"\n Memulai evaluasi")
print(f"Mulai: {time.strftime('%H:%M:%S')}")
print("-" * 80)
print(f"{'No':<5} {'Kelas':<10} {'Faith':>7} {'AnsRel':>7} {'CtxPre':>7} {'CtxRec':>7} {'Waktu':>7}")
print("-" * 80)

results = []
start_total = time.time()

for i, row in df.iterrows():
    start_row = time.time()

    single_data = Dataset.from_dict({
        "question":     [row["pertanyaan"]],
        "answer":       [row["jawaban_llm"]],
        "contexts":     [row["contexts"]],
        "ground_truth": [row["kunci"]]
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

        faith  = result_row.get("faithfulness", float("nan"))
        ansrel = result_row.get("answer_relevancy", float("nan"))
        ctxpre = result_row.get("context_precision", float("nan"))
        ctxrec = result_row.get("context_recall", float("nan"))

    except Exception as e:
        print(f"  Error baris {row['no']}: {e}")
        faith = ansrel = ctxpre = ctxrec = float("nan")

    elapsed_row = time.time() - start_row

    results.append({
        "no":                row["no"],
        "kelas":             row["kelas"],
        "pertanyaan":        row["pertanyaan"],
        "faithfulness":      faith,
        "answer_relevancy":  ansrel,
        "context_precision": ctxpre,
        "context_recall":    ctxrec
    })

    print(f"{row['no']:<5} {str(row['kelas']):<10} {faith:>7.3f} {ansrel:>7.3f} {ctxpre:>7.3f} {ctxrec:>7.3f} {elapsed_row:>5.0f}s")


result_df = pd.DataFrame(results)
result_df.to_excel("hasil_evaluasi_ragas.xlsx", index=False)

elapsed_total = time.time() - start_total
print("-" * 80)
print(f"Total waktu: {elapsed_total/3600:.1f} jam ({elapsed_total/60:.0f} menit)")

print(f"\nRATA-RATA PER METRIK")
print(result_df[["faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]].mean())

print(f"\nRATA-RATA PER KELAS")
print(result_df.groupby("kelas")[["faithfulness", "answer_relevancy",
                                   "context_precision", "context_recall"]].mean())

print("\nHasil tersimpan di: hasil_evaluasi_ragas.xlsx")