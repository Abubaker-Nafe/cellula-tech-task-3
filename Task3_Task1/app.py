import json
import logging
import os
import threading
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from generate_answer import UNKNOWN_ANSWER, generate_answer
from retrieve_documents import load_retrieval_resources, retrieve_documents

PROJECT_DIR = Path(__file__).resolve().parent
STORE_DIR = PROJECT_DIR / "output" / "vector_store"
EXAMPLES = [
    "What backend development experience does Nafe have?",
    "What were Nafe's primary and secondary roles in the student dropout prediction capstone?",
    "Which AWS certifications does Nafe hold?",
]
LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def cached_retriever(index_modified, metadata_modified):
    # File timestamps invalidate the cache when the vector store is rebuilt.
    # The lock protects the shared embedding model during concurrent requests.
    return load_retrieval_resources(), threading.Lock()


def configure_api_key():
    load_dotenv(PROJECT_DIR / ".env")
    if os.getenv("OPENROUTER_API_KEY", "").strip():
        return True
    try:
        key = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        key = ""
    if key:
        os.environ["OPENROUTER_API_KEY"] = key
        return True
    return False


def show_sources(results):
    st.subheader("Retrieved sources")
    st.caption("Citation numbers in the answer refer to these passages.")
    for result in results:
        metadata = result["metadata"]
        label = f"{result['reference']} {metadata['source']}"
        if metadata.get("page") is not None:
            label += f" · Page {metadata['page']}"
        with st.expander(label):
            url = metadata.get("url", "")
            if url and urlparse(url).scheme in ("http", "https"):
                st.link_button("Open original source", url)
            st.text(result["page_content"])


def main():
    st.set_page_config(page_title="Nafe's Professional Knowledge Base", page_icon="📚")
    st.title("Nafe's Professional Knowledge Base")
    st.write("Ask about Nafe's experience, skills, and projects. Answers use the CV and capstone README.")

    example = st.selectbox("Example question", EXAMPLES)
    with st.form("question_form"):
        question = st.text_area("Your question", value=example, height=100, max_chars=2000)
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        # Do not display a previous answer as the result of a failed new request.
        st.session_state.pop("latest_answer", None)
        question = question.strip()
        if not question:
            st.warning("Enter a question first.")
        elif not configure_api_key():
            st.error("Answer generation is not configured. The app owner needs to add the API key.")
        else:
            try:
                with st.spinner("Finding relevant passages and preparing an answer..."):
                    index_path = STORE_DIR / "index.faiss"
                    metadata_path = STORE_DIR / "metadata.json"
                    resources, lock = cached_retriever(
                        index_path.stat().st_mtime_ns,
                        metadata_path.stat().st_mtime_ns,
                    )
                    with lock:
                        results = retrieve_documents(question, top_k=3, resources=resources)
                    answer, model_name = generate_answer(question, results)
                # Answers belong to this browser session, not a shared cache.
                st.session_state["latest_answer"] = {
                    "question": question, "answer": answer,
                    "model": model_name, "retrieved_sources": results,
                }
            except FileNotFoundError:
                LOGGER.exception("Required application file is missing")
                st.error("The knowledge base is unavailable. The app owner needs to finish setup.")
            except ValueError as exc:
                LOGGER.exception("Question or answer validation failed")
                if "question exceeds" in str(exc).lower():
                    st.warning("That question is too long. Please shorten it.")
                else:
                    st.error("The service could not produce a valid answer. Please try again.")
            except Exception:
                LOGGER.exception("RAG request failed")
                st.error("The service could not complete your request. Please try again later.")

    saved = st.session_state.get("latest_answer")
    if saved:
        st.divider()
        st.subheader("Answer")
        st.caption(f"Question: {saved['question']}")
        if saved["answer"] == UNKNOWN_ANSWER:
            st.info(saved["answer"])
        else:
            st.markdown(saved["answer"])
        show_sources(saved["retrieved_sources"])
        st.download_button(
            "Download answer",
            data=json.dumps(saved, ensure_ascii=False, indent=2),
            file_name="answer.json", mime="application/json",
        )


if __name__ == "__main__":
    main()
