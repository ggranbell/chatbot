import argparse
import re
import requests
import lancedb
from collections import Counter
from pathlib import Path
import fitz
import pytesseract
from pytesseract import TesseractNotFoundError
from PIL import Image

class MolaProcessor:
    def __init__(self, target_tokens=500, overlap_tokens=50):
        self.target_len = target_tokens * 4  # Approx 4 chars per token
        self.overlap_len = overlap_tokens * 4
        self.stop_words = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "this", "that"
        }

    def clean_text(self, text: str) -> str:
        """My Tokenizer-friendly normalization."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(char for char in text if char.isprintable() or char in "\n\t")
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        return text.strip()

    def get_keywords(self, text: str) -> str:
        """Extracts significant terms for hybrid search."""
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        meaningful = [w for w in words if w not in self.stop_words]
        return ", ".join([item for item, count in Counter(meaningful).most_common(5)])

    def slugify(self, text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')

    def derive_label(self, stem: str, first_line: str) -> str:
        """Generates the 'breadcrumb' label."""
        section = self.slugify(first_line[:50])
        return f"{self.slugify(stem)}:{section}" if len(section) > 5 else self.slugify(stem)

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _hard_split(self, text: str) -> list[str]:
        if not text:
            return []
        step = max(self.target_len - self.overlap_len, 1)
        chunks = []
        for start in range(0, len(text), step):
            end = start + self.target_len
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
        return chunks

    def _split_oversized_unit(self, text: str) -> list[str]:
        if len(text) <= self.target_len:
            return [text]

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._hard_split(text)

        chunks = []
        current = ""
        for sentence in sentences:
            if not current:
                current = sentence
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= self.target_len:
                current = candidate
            else:
                chunks.append(current.strip())
                current = sentence

        if current.strip():
            chunks.append(current.strip())

        flattened = []
        for chunk in chunks:
            if len(chunk) <= self.target_len:
                flattened.append(chunk)
            else:
                flattened.extend(self._hard_split(chunk))
        return flattened

    def split_semantic(self, text: str) -> list[str]:
        """Semantic-aware chunking that preserves context boundaries and order."""
        if not text:
            return []

        if len(text) <= self.target_len:
            return [text]

        paragraphs = [paragraph.strip() for paragraph in re.split(r'\n{2,}', text) if paragraph.strip()]
        if not paragraphs:
            return self._hard_split(text)

        chunks = []
        current_chunk = ""
        min_chunk_len = int(self.target_len * 0.35)

        def flush_chunk(force: bool = False):
            nonlocal current_chunk
            if not current_chunk.strip():
                return
            if force or len(current_chunk) >= min_chunk_len or not chunks:
                chunks.append(current_chunk.strip())
                current_chunk = ""

        for paragraph in paragraphs:
            paragraph_units = self._split_oversized_unit(paragraph)

            for unit in paragraph_units:
                if not current_chunk:
                    current_chunk = unit
                    continue

                candidate = f"{current_chunk}\n\n{unit}".strip()
                if len(candidate) <= self.target_len:
                    current_chunk = candidate
                    continue

                flush_chunk(force=True)
                overlap = ""
                if chunks and self.overlap_len > 0:
                    overlap = chunks[-1][-self.overlap_len:].strip()
                current_chunk = f"{overlap}\n\n{unit}".strip() if overlap else unit

        flush_chunk(force=True)

        normalized = []
        for chunk in chunks:
            if normalized and len(chunk) < min_chunk_len:
                merged = f"{normalized[-1]}\n\n{chunk}".strip()
                if len(merged) <= self.target_len + self.overlap_len:
                    normalized[-1] = merged
                    continue
            normalized.append(chunk)

        return normalized or self._hard_split(text)

def fetch_embeddings(texts: list[str], model: str, host: str) -> list[list[float]]:
    """Batched embedding generation via Ollama /api/embed for better throughput."""
    try:
        if not texts:
            return []
        url = f"{host.rstrip('/')}/api/embed"
        res = requests.post(url, json={"model": model, "input": texts}, timeout=120)
        res.raise_for_status()
        embeddings = res.json().get("embeddings", [])
        if not isinstance(embeddings, list):
            return []
        return embeddings
    except Exception as e:
        print(f"Embedding error (batch size {len(texts)}): {e}")
        return [[] for _ in texts]

def get_page_text(page, matrix: fitz.Matrix, ocr_enabled: bool, lang: str) -> tuple[str, bool]:
    native_text = page.get_text("text") or ""

    if not ocr_enabled:
        return native_text, False

    try:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(image, lang=lang) or ""
        if ocr_text.strip():
            return ocr_text, True
        return native_text, False
    except TesseractNotFoundError:
        return native_text, False

def is_ocr_available() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except TesseractNotFoundError:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--db-path", default="data/vector-db")
    parser.add_argument("--table", default="knowledge_base")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ocr-dpi", type=int, default=300)
    parser.add_argument("--ocr-lang", default="eng")
    parser.add_argument("--tesseract-cmd", default="")
    parser.add_argument("--require-ocr", action="store_true")
    args = parser.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    ocr_available = is_ocr_available()
    if not ocr_available:
        warning = "Tesseract OCR is not installed or not in PATH. Falling back to native PDF text extraction."
        print(warning)
        if args.require_ocr:
            raise SystemExit("OCR is required but Tesseract is not available.")

    processor = MolaProcessor(target_tokens=400, overlap_tokens=50)
    db = lancedb.connect(args.db_path)
    
    if args.overwrite and args.table in db.table_names():
        db.drop_table(args.table)

    path = Path(args.pdf_path)
    files = list(path.rglob("*.pdf")) if path.is_dir() else ([path] if path.exists() else [])
    
    all_data = []
    uid = 1

    for f in files:
        print(f"-> Indexing: {f.name}")
        zoom = max(args.ocr_dpi, 72) / 72
        matrix = fitz.Matrix(zoom, zoom)

        try:
            with fitz.open(f) as document:
                for page_num, page in enumerate(document, start=1):
                    raw_text, used_ocr = get_page_text(page, matrix=matrix, ocr_enabled=ocr_available, lang=args.ocr_lang)
                    text = processor.clean_text(raw_text)
                    if not text:
                        continue

                    chunks = processor.split_semantic(text)
                    if not chunks:
                        continue

                    embed_inputs = [
                        f"Source: {f.stem} | Page: {page_num} | Chunk: {chunk_idx + 1}/{len(chunks)} | {chunk}"
                        for chunk_idx, chunk in enumerate(chunks)
                    ]
                    vectors = fetch_embeddings(embed_inputs, args.embedding_model, args.ollama_host)

                    for chunk_idx, chunk in enumerate(chunks, start=1):
                        vector = vectors[chunk_idx - 1] if chunk_idx - 1 < len(vectors) else []
                        if not vector:
                            continue

                        all_data.append({
                            "id": uid,
                            "vector": vector,
                            "text": chunk,
                            "label": processor.derive_label(f.stem, chunk),
                            "metadata": {
                                "source": f.name,
                                "page": page_num,
                                "chunk_index": chunk_idx,
                                "chunk_count": len(chunks),
                                "keywords": processor.get_keywords(chunk),
                                "extractor": "ocr" if used_ocr else "native"
                            }
                        })
                        uid += 1
        except Exception as err:
            print(f"Extraction error on {f.name}: {err}")
            continue

    if all_data:
        if args.table in db.table_names():
            db.open_table(args.table).add(all_data)
        else:
            db.create_table(args.table, data=all_data)
        print(f"Added {len(all_data)} rows")
    else:
        raise SystemExit("Added 0 rows. No extractable text was found in the uploaded PDF(s).")

if __name__ == "__main__":
    main()