import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np


def search_vectors(index, chunks, query_vector, top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    vector = np.ascontiguousarray(
        np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    )
    if vector.shape[1] != index.d or not np.isfinite(vector).all():
        raise ValueError("The question vector does not match the index")
    if np.linalg.norm(vector) == 0:
        raise ValueError("The question vector has zero length")
    faiss.normalize_L2(vector)

    scores, row_ids = index.search(vector, min(top_k, index.ntotal))
    results = []
    for rank, (score, row_id) in enumerate(zip(scores[0], row_ids[0]), start=1):
        chunk = chunks[int(row_id)]
        results.append({
            "reference": f"[{rank}]",
            "score": float(score),
            "page_content": chunk["page_content"],
            "metadata": chunk["metadata"],
        })
    return results


def retrieve_documents(question, top_k=3):
    if not question.strip():
        raise ValueError("Enter a non-empty question")
    project_dir = Path(__file__).resolve().parent
    store_dir = project_dir / "output" / "vector_store"
    index_path = store_dir / "index.faiss"
    metadata_path = store_dir / "metadata.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Run build_vector_store.py first")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks = metadata["chunks"]
    if index.ntotal == 0 or index.ntotal != len(chunks):
        raise ValueError("The index does not match the saved chunks")
    if index.d != metadata["embedding_dimension"]:
        raise ValueError("The index dimension does not match the metadata")
    if index.metric_type != faiss.METRIC_INNER_PRODUCT or not metadata.get("normalize_embeddings"):
        raise ValueError("Expected an inner-product index with normalized embeddings")

    # Use exactly the same pretrained model used to embed the CV chunks.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(metadata["model_name"], device="cpu")
    question_tokens = model.tokenizer(question, truncation=False)["input_ids"]
    if len(question_tokens) > model.max_seq_length:
        raise ValueError("The question exceeds the embedding model's token limit; shorten it")
    query_vector = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    return search_vectors(index, chunks, query_vector, top_k)


def main():
    parser = argparse.ArgumentParser(description="Retrieve CV passages using semantic similarity")
    parser.add_argument("question", help="Put the question in quotation marks")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve (default: 3)")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    # Keep CV punctuation printable on Windows terminals with older encodings.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    results = retrieve_documents(args.question, args.top_k)
    print(f"Question: {args.question}")
    print(f"Retrieved passages: {len(results)}")
    for result in results:
        metadata = result["metadata"]
        print(f"\n{result['reference']} Similarity: {result['score']:.4f}")
        print(f"Source: {metadata['source']} | Page: {metadata['page']} | Chunk: {metadata['chunk_id']}")
        print(result["page_content"])

    output_path = Path(__file__).resolve().parent / "output" / "retrieval_results.json"
    output_path.write_text(
        json.dumps({"question": args.question, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSaved: output/retrieval_results.json")


if __name__ == "__main__":
    main()
