import os
import re
import shutil
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

CHROMA_DIR  = "./chroma_db"
DATASET_DIR = "./dataset"
PREVIEW_DIR = "./chunk_preview"

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80
BATCH_SIZE    = 500

MAPEL_MAP = {
    "IPAS"      : "ipas",
    "PANCASILA" : "pendidikan pancasila",
    "PJOK"      : "pjok",
}


def detect_mapel(filename: str) -> str:
    name = Path(filename).stem.upper()
    for key, value in MAPEL_MAP.items():
        if key in name:
            return value
    return name.lower()


def clean_text(text: str) -> str:
    """Cleaning PDF text tanpa merusak struktur newline."""
    text = re.sub(r'Halaman\s+\d+', '', text, flags=re.I)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'ISBN[:\s0-9\-Xx]+', '', text, flags=re.I)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name.lower()


def export_chunks_per_file(kelas: str, mapel: str, source_name: str, chunks):
    """Export chunk preview per PDF — isi chunk tanpa prefix embedding."""
    kelas_dir = os.path.join(PREVIEW_DIR, safe_filename(kelas))
    os.makedirs(kelas_dir, exist_ok=True)

    pdf_name = Path(source_name).stem
    out_name = f"{safe_filename(pdf_name)}_chunks.txt"
    out_path = os.path.join(kelas_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.metadata

            display_content = chunk.page_content
            if display_content.startswith("search_document: "):
                display_content = display_content[len("search_document: "):]

            f.write("=" * 100 + "\n")
            f.write(f"CHUNK #{i}\n")
            f.write(f"kelas     : {meta['kelas']}\n")
            f.write(f"mapel     : {meta['mapel']}\n")
            f.write(f"source    : {meta['source']}\n")
            f.write(f"page      : {meta['page']}\n")
            f.write(f"chunk_id  : {meta['chunk_id']}\n")
            f.write(f"doc_id    : {meta['doc_id']}\n")
            f.write("-" * 100 + "\n")
            f.write(display_content)
            f.write("\n\n")

    print(f"Preview  → {out_path}")


def ingest_data():
    print(
        f"PROSES INGEST "
        f"(Fixed Chunk={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})\n"
        f"   Model : nomic-embed-text-v2-moe\n"
        f"   Prefix: 'search_document:' ditambahkan ke setiap chunk"
    )

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("hapus Chroma lama")

    if os.path.exists(PREVIEW_DIR):
        shutil.rmtree(PREVIEW_DIR)
        print("Preview lama hapus.")

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text-v2-moe")
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks  = []
    all_ids     = []
    seen_hashes = set()

    for root, _, files in os.walk(DATASET_DIR):
        for filename in files:
            if not filename.lower().endswith(".pdf"):
                continue

            filepath   = os.path.join(root, filename)
            rel_path   = os.path.relpath(filepath, DATASET_DIR)
            path_parts = rel_path.split(os.sep)

            current_kelas = (
                path_parts[0].strip().lower()
                if len(path_parts) > 1
                else "unknown"
            )
            current_mapel = detect_mapel(filename)

            print(f"\nMemproses: {current_kelas} | {current_mapel} | {filename}")

            try:
                loader = PyMuPDFLoader(filepath)
                pages  = loader.load()

                file_chunks = []

                for page_idx, page in enumerate(pages):
                    text = clean_text(page.page_content)

                    if len(text) < 50:
                        continue

                    chunks = text_splitter.create_documents([text])

                    for chunk_idx, chunk in enumerate(chunks):
                        raw_content = chunk.page_content.strip()

                        if len(raw_content) < 30:
                            continue

                        h = text_hash(raw_content)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        
                        chunk.page_content = "search_document: " + raw_content

                        doc_id = (
                            f"{current_kelas}_"
                            f"{current_mapel}_"
                            f"{Path(filename).stem}_"
                            f"p{page_idx+1}_"
                            f"c{chunk_idx}"
                        )

                        chunk.metadata.update({
                            "kelas"    : current_kelas,
                            "mapel"    : current_mapel,
                            "source"   : filename,
                            "page"     : page_idx + 1,
                            "chunk_id" : chunk_idx,
                            "doc_id"   : doc_id,
                        })

                        file_chunks.append(chunk)
                        all_chunks.append(chunk)
                        all_ids.append(doc_id)

                print(f"   → {len(file_chunks)} chunks dari {len(pages)} halaman")

                if file_chunks:
                    export_chunks_per_file(
                        current_kelas,
                        current_mapel,
                        filename,
                        file_chunks,
                    )

            except Exception as e:
                print(f"Gagal membaca {filename}: {e}")

    if not all_chunks:
        print("data gaketemu.")
        return

    print(f"\nMenyimpan {len(all_chunks)} chunks")

    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_docs = all_chunks[i : i + BATCH_SIZE]
        batch_ids  = all_ids[i : i + BATCH_SIZE]

        db.add_documents(
            documents=batch_docs,
            ids=batch_ids,
        )

        print(f"batch {i // BATCH_SIZE + 1} ({len(batch_docs)} docs)")

    print("\nINGEST SELESAI")
    print(f"Total chunk : {len(all_chunks)}")
    print(f"Preview dir : {PREVIEW_DIR}")
    print(f"Chroma DB   : {CHROMA_DIR}")


if __name__ == "__main__":
    ingest_data()