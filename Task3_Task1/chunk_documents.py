"""
The script reads output/loaded_documents.json, preserves each page's metadata, adds a chunk_id, and saves output/chunks.json
"""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    project_dir = Path(__file__).resolve().parent
    input_path = project_dir / "output" / "loaded_documents.json"
    if not input_path.is_file():
        raise FileNotFoundError("Run load_documents.py first to create output/loaded_documents.json")

    loaded_documents = json.loads(input_path.read_text(encoding="utf-8"))
    if not loaded_documents:
        raise ValueError("No documents found in loaded_documents.json")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
    )

    # Each page's metadata is copied to the chunks created from that page.
    chunks = splitter.create_documents(
        texts=[document["page_content"] for document in loaded_documents],
        metadatas=[document["metadata"] for document in loaded_documents],
    )

    saved_chunks = []
    for index, chunk in enumerate(chunks, start=1):
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

    print(f"Loaded pages: {len(loaded_documents)}")
    print("Chunk size: 800 characters")
    print("Target overlap: 120 characters")
    print(f"Created chunks: {len(saved_chunks)}")
    for chunk in saved_chunks:
        metadata = chunk["metadata"]
        print(
            f"{metadata['chunk_id']}: {len(chunk['page_content'])} characters "
            f"| Source: {metadata['source']} | Page: {metadata['page']}"
        )
    print("Saved: output/chunks.json")


if __name__ == "__main__":
    main()
