import argparse
import json
import sys
from pathlib import Path

from retrieve_documents import retrieve_documents, format_source
from generate_answer import generate_answer


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the personal knowledge base")
    parser.add_argument("question", help="Put your question in quotation marks")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve (default: 3)")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Enter a non-empty question")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Question: {args.question}")
    print("Retrieving passages...")
    results = retrieve_documents(args.question, top_k=args.top_k)
    (output_dir / "retrieval_results.json").write_text(
        json.dumps({"question": args.question, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generating answer...")
    # Pass this question's passages directly to the generation function.
    answer, model_name = generate_answer(args.question, results)
    print(f"\nAnswer:\n{answer}")
    print("\nRetrieved sources:")
    for result in results:
        print(format_source(result))

    (output_dir / "answer.json").write_text(
        json.dumps({"question": args.question, "answer": answer, "model": model_name,
                    "retrieved_sources": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nSaved: output/retrieval_results.json and output/answer.json")


if __name__ == "__main__":
    main()
