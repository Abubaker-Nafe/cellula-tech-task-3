import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    chunks_path = output_dir / "chunks.json"
    if not chunks_path.is_file():
        raise FileNotFoundError("Run chunk_documents.py first to create output/chunks.json")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not chunks:
        raise ValueError("No chunks found in chunks.json")
    texts = [chunk["page_content"] for chunk in chunks]
    if any(not text.strip() for text in texts):
        raise ValueError("An empty chunk was found in chunks.json")

    print(f"Loading model on CPU: {MODEL_NAME}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    # Character-based chunks do not guarantee a particular token count.
    # Check the model's token limit to avoid silently truncating CV information.
    token_ids = model.tokenizer(texts, truncation=False, add_special_tokens=True)["input_ids"]
    token_lengths = [len(ids) for ids in token_ids]
    longest_chunk = max(token_lengths)
    if longest_chunk > model.max_seq_length:
        raise ValueError(
            f"A chunk has {longest_chunk} tokens; model limit is {model.max_seq_length}. "
            "Reduce chunk_size in chunk_documents.py, then rerun chunking and embeddings."
        )

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding output does not match the chunks")
    if not np.isfinite(vectors).all():
        raise ValueError("Embedding output contains invalid values")

    np.save(output_dir / "embeddings.npy", vectors)

    # Row i of embeddings.npy corresponds to item i in this saved chunk list.
    metadata = {
        "model_name": MODEL_NAME,
        "normalize_embeddings": True,
        "embedding_dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    (output_dir / "embeddings_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Chunks embedded: {len(chunks)}")
    print(f"Embedding shape: {vectors.shape}")
    print(f"Numbers per chunk: {vectors.shape[1]}")
    print(f"Longest chunk: {longest_chunk} tokens | Model limit: {model.max_seq_length}")
    print("Saved: output/embeddings.npy")
    print("Saved: output/embeddings_metadata.json")


if __name__ == "__main__":
    main()
