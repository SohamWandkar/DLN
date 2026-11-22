"""
Data Processor and Chunking Engine for LearnMate
Preprocesses syllabus documents into structured, metadata-indexed passages.
"""
import os
import sys
import re
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import CORPUS_DIR, PROCESSED_DIR, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS

def process_corpus():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    chunks = []
    chunk_counter = 0

    if not os.path.exists(CORPUS_DIR):
        raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

    files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")])
    for fname in files:
        fpath = os.path.join(CORPUS_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse Module Header
        mod_match = re.search(r"MODULE:\s*(.+)", content)
        module_name = mod_match.group(1).strip() if mod_match else fname

        # Split into sections
        sections = re.split(r"(SECTION:\s*.+)", content)
        current_section = "General Overview"
        
        for part in sections:
            part = part.strip()
            if not part:
                continue
            if part.startswith("SECTION:"):
                current_section = part.replace("SECTION:", "").strip()
                continue

            words = part.split()
            start = 0
            while start < len(words):
                end = min(start + CHUNK_SIZE_WORDS, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"CHUNK_{chunk_counter:04d}",
                    "source_file": fname,
                    "module": module_name,
                    "section": current_section,
                    "text": chunk_text,
                    "citation": f"{module_name} | {current_section}"
                })
                if end == len(words):
                    break
                start += (CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS)

    out_path = os.path.join(PROCESSED_DIR, "chunks_corpus.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"total_chunks": len(chunks), "chunks": chunks}, f, indent=2)

    print(f"Processed {len(files)} course modules into {len(chunks)} chunks.")
    print(f"Saved chunk corpus to {out_path}")
    return chunks

if __name__ == "__main__":
    process_corpus()
