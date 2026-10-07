import json
from pathlib import Path

import faiss
import numpy as np


def main():
    project_dir = Path(__file__).resolve().parent
    output_dir = project_dir / "output"
    vectors_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "embeddings_metadata.json"
    if not vectors_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run create_embeddings.py first to create the embedding files")

    vectors = np.ascontiguousarray(
        np.load(vectors_path, allow_pickle=False), dtype=np.float32
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata["chunks"]

    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Expected a non-empty matrix of embeddings")
    if vectors.shape != (len(chunks), metadata["embedding_dimension"]):
        raise ValueError("The vectors do not match their chunk metadata")
    if not np.isfinite(vectors).all():
        raise ValueError("The vectors contain invalid values")
    if not metadata.get("normalize_embeddings") or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5
    ):
        raise ValueError("Expected normalized vectors; rerun create_embeddings.py")

    # Inner product equals cosine similarity when both vectors have unit length.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    store_dir = output_dir / "vector_store"
    store_dir.mkdir(exist_ok=True)
    faiss.write_index(index, str(store_dir / "index.faiss"))

    # FAISS returns row numbers. Row i maps to item i in this chunk list.
    metadata["similarity_metric"] = "cosine"
    (store_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Vectors stored: {index.ntotal}")
    print(f"Vector dimension: {index.d}")
    print("Index type: IndexFlatIP")
    print("Similarity: cosine (normalized vectors)")
    print("Saved: output/vector_store/index.faiss")
    print("Saved: output/vector_store/metadata.json")


if __name__ == "__main__":
    main()
