import json
import sys
from pathlib import Path

from pypdf import PdfReader


def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not text:
            raise ValueError(f"Page {page_number} has no extractable text.")

        # PDF hyperlinks are stored separately from the visible text.
        links = []
        for annotation in page.get("/Annots", []):
            action = annotation.get_object().get("/A")
            if action:
                action = action.get_object()
                url = action.get("/URI")
                if url and str(url) not in links:
                    links.append(str(url))

        documents.append({
            "page_content": text,
            "metadata": {
                "source": pdf_path.name,
                "page": page_number,
                "links": links,
            },
        })

    return documents


def main():
    project_dir = Path(__file__).resolve().parent
    pdf_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else project_dir / "data" / "Nafe_Abubaker_CV.pdf"
    )
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    documents = load_pdf(pdf_path)
    output_dir = project_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "loaded_documents.json"
    output_path.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Loaded: {pdf_path.name}")
    print(f"Pages: {len(documents)}")
    for document in documents:
        metadata = document["metadata"]
        print(
            f"Page {metadata['page']}: {len(document['page_content'])} characters, "
            f"{len(metadata['links'])} hyperlinks"
        )
    print("Saved: output/loaded_documents.json")


if __name__ == "__main__":
    main()
