import argparse
import base64
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_REPOSITORY = "Abubaker-Nafe/ibt-ggateway-capstone"


def fetch_readme(repository):
    request = Request(
        f"https://api.github.com/repos/{repository}/readme",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Cellula-Personal-RAG",
            "X-GitHub-Api-Version": "2026-03-10",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:  # urllib uses seconds.
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError("GitHub could not find a public repository README; check the repository name") from exc
        if exc.code in (403, 429):
            raise ValueError("GitHub denied or rate-limited the request; try again later") from exc
        raise ValueError(f"GitHub returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("Could not download the README; check your connection and try again") from exc
    if payload.get("encoding") != "base64":
        raise ValueError("Expected a Base64-encoded README from GitHub")
    text = base64.b64decode(payload["content"]).decode("utf-8-sig").strip()
    if not text:
        raise ValueError("The repository README is empty")
    return text, payload


def make_document(repository, text, payload):
    source_url = payload["html_url"]
    return {
        "page_content": f"GitHub repository: {repository}\n\n{text}",
        "metadata": {
            "source": f"GitHub: {repository}/{payload['path']}",
            "source_type": "github_readme",
            "repository": repository,
            "page": None,  # A README has no PDF page number.
            "url": source_url,
            "links": [source_url],
            "github_blob_sha": payload["sha"],
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Add a public GitHub README to the knowledge base")
    parser.add_argument("repository", nargs="?", default=DEFAULT_REPOSITORY, help="owner/repository")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("Use a repository name in owner/repository format")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    project_dir = Path(__file__).resolve().parent
    documents_path = project_dir / "output" / "loaded_documents.json"
    if not documents_path.is_file():
        raise FileNotFoundError("Run load_documents.py first to load the CV")
    documents = json.loads(documents_path.read_text(encoding="utf-8"))
    if not isinstance(documents, list) or not documents:
        raise ValueError("Expected an existing list of loaded documents")

    print(f"Downloading README: {args.repository}")
    text, payload = fetch_readme(args.repository)
    document = make_document(args.repository, text, payload)
    # Refresh this README on repeated runs without duplicating it.
    documents = [d for d in documents if not (
        d["metadata"].get("source_type") == "github_readme"
        and d["metadata"].get("repository", "").lower() == args.repository.lower()
    )]
    documents.append(document)

    data_dir = project_dir / "data"
    data_dir.mkdir(exist_ok=True)
    cache_path = data_dir / (args.repository.replace("/", "_") + "_README.md")
    cache_path.write_text(text + "\n", encoding="utf-8")
    documents_path.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded: {document['metadata']['source']}")
    print(f"README characters: {len(text)}")
    print(f"Source URL: {document['metadata']['url']}")
    print(f"Total loaded documents: {len(documents)}")
    print(f"Saved README: data/{cache_path.name}")
    print("Updated: output/loaded_documents.json")
    print("Next: rebuild chunks, embeddings, and the vector store to make this source searchable.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")
