import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np


def rank_passages(chunks, semantic_scores, question=None, top_k=3):
    """Combine normalized semantic and TF-IDF scores for this small collection."""
    semantic_scores = np.asarray(semantic_scores, dtype=np.float32)
    if top_k < 1 or len(semantic_scores) != len(chunks):
        raise ValueError("Invalid retrieval count or chunk mapping")
    ranking_scores = semantic_scores.copy()
    keyword_scores = None
    if question:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        text_matrix = vectorizer.fit_transform([c["page_content"] for c in chunks])
        query_text_vector = vectorizer.transform([question])
        keyword_scores = (text_matrix @ query_text_vector.T).toarray().ravel()
        if keyword_scores.max(initial=0) > 0:
            positive_semantic = np.maximum(semantic_scores, 0)
            semantic_max = positive_semantic.max(initial=0)
            semantic_scaled = positive_semantic / semantic_max if semantic_max > 0 else positive_semantic
            keyword_scaled = keyword_scores / keyword_scores.max()
            ranking_scores = 0.4 * semantic_scaled + 0.6 * keyword_scaled

    selected_rows = np.argsort(-ranking_scores, kind="stable")[:min(top_k, len(chunks))]
    results = []
    for rank, row_id in enumerate(selected_rows, start=1):
        chunk = chunks[int(row_id)]
        result = {
            "reference": f"[{rank}]",
            "score": float(semantic_scores[row_id]),
            "page_content": chunk["page_content"],
            "metadata": chunk["metadata"],
        }
        if keyword_scores is not None:
            result["keyword_score"] = float(keyword_scores[row_id])
            result["retrieval_score"] = float(ranking_scores[row_id])
        results.append(result)
    return results


def format_source(result):
    metadata = result["metadata"]
    label = f"{result['reference']} {metadata['source']}"
    if metadata.get("page") is not None:
        label += f" | Page: {metadata['page']}"
    label += f" | Chunk: {metadata['chunk_id']} | Semantic similarity: {result['score']:.4f}"
    if "retrieval_score" in result:
        label += f" | Retrieval score: {result['retrieval_score']:.4f}"
    if metadata.get("url"):
        label += f"\nURL: {metadata['url']}"
    return label


def search_vectors(index, chunks, query_vector, top_k=3, question=None):
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

    if index.ntotal != len(chunks) or not chunks:
        raise ValueError("The index does not match the chunks")
    # Evaluate every stored vector in this small collection to recover dense misses.
    scores, row_ids = index.search(vector, index.ntotal)
    semantic_scores = np.empty(index.ntotal, dtype=np.float32)
    semantic_scores[row_ids[0]] = scores[0]
    return rank_passages(chunks, semantic_scores, question=question, top_k=top_k)


def load_retrieval_resources():
    """Load the saved store and embedding model; a UI can cache these resources."""
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

    # Use exactly the same pretrained model used to embed the source chunks.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(metadata["model_name"], device="cpu")
    return index, chunks, model


def retrieve_documents(question, top_k=3, resources=None):
    if not question.strip():
        raise ValueError("Enter a non-empty question")
    index, chunks, model = resources if resources is not None else load_retrieval_resources()
    question_tokens = model.tokenizer(question, truncation=False)["input_ids"]
    if len(question_tokens) > model.max_seq_length:
        raise ValueError("The question exceeds the embedding model's token limit; shorten it")
    query_vector = model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    return search_vectors(index, chunks, query_vector, top_k, question=question)


def main():
    parser = argparse.ArgumentParser(description="Retrieve passages using semantic and TF-IDF keyword matching")
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
        print("\n" + format_source(result))
        print(result["page_content"])

    output_path = Path(__file__).resolve().parent / "output" / "retrieval_results.json"
    output_path.write_text(
        json.dumps({"question": args.question, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSaved: output/retrieval_results.json")


if __name__ == "__main__":
    main()
