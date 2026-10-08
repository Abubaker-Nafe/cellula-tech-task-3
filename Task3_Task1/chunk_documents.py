import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_LIMIT = 256
CHUNK_TOKENS = 220
OVERLAP_TOKENS = 30


def main():
    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / "output" / "loaded_documents.json"
    if not input_path.is_file():
        raise FileNotFoundError("Run load_documents.py first to create output/loaded_documents.json")

    loaded_documents = json.loads(input_path.read_text(encoding="utf-8"))
    if not loaded_documents:
        raise ValueError("No documents found in loaded_documents.json")

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Count full documents without truncation or warnings. No model runs here;
    # each final chunk is checked against MiniLM's actual 256-token window below.
    tokenizer.model_max_length = 10**9
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
    )

    # Each document's metadata, including GitHub URLs, is copied to its chunks.
    chunks = splitter.create_documents(
        texts=[document["page_content"] for document in loaded_documents],
        metadatas=[document["metadata"] for document in loaded_documents],
    )

    saved_chunks = []
    token_counts = []
    for index, chunk in enumerate(chunks, start=1):
        token_count = len(tokenizer.encode(chunk.page_content, truncation=False))
        if token_count > MODEL_LIMIT:
            raise ValueError(
                f"Chunk {index} has {token_count} tokens, exceeding {MODEL_LIMIT}. "
                "Reduce CHUNK_TOKENS and rerun this script."
            )
        token_counts.append(token_count)
        metadata = dict(chunk.metadata)
        metadata["chunk_id"] = f"chunk_{index:03d}"
        saved_chunks.append({
            "page_content": chunk.page_content,
            "metadata": metadata,
        })

    output_path = project_dir / "output" / "chunks.json"
    output_path.write_text(
        json.dumps(saved_chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Loaded documents: {len(loaded_documents)}")
    print(f"Chunk size: {CHUNK_TOKENS} content tokens")
    print(f"Target overlap: {OVERLAP_TOKENS} tokens")
    print(f"Model limit: {MODEL_LIMIT} tokens, including special tokens")
    print(f"Longest chunk: {max(token_counts, default=0)} tokens, including special tokens")
    print(f"Created chunks: {len(saved_chunks)}")
    for chunk, token_count in zip(saved_chunks, token_counts):
        metadata = chunk["metadata"]
        page_label = f" | Page: {metadata['page']}" if metadata.get("page") is not None else ""
        print(
            f"{metadata['chunk_id']}: {token_count} tokens, {len(chunk['page_content'])} characters "
            f"| Source: {metadata['source']}{page_label}"
        )
    print("Saved: output/chunks.json")


if __name__ == "__main__":
    main()
