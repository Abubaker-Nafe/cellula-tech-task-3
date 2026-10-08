# Task 3 — Personal RAG Knowledge Base

This project answers questions about Nafe Abubaker's professional background using a CV and a GitHub project README. It retrieves relevant passages, sends them to an answer-generation model, and displays citations and source details. When the retrieved passages do not answer the question, it returns an insufficient-information message.

**Live app:** [cellula-tech-task-3.streamlit.app](https://cellula-tech-task-3.streamlit.app/)

The application runs from the command line or a Streamlit interface. Python 3.11 on Windows was used for development.

## Try the deployed app

Open the [Streamlit app](https://cellula-tech-task-3.streamlit.app/), select an example question or enter your own, then click **Ask**. Expand the retrieved sources to read the passages behind the answer. You can also download the answer and its source details as JSON.

For example, ask about backend development experience, capstone responsibilities, or certifications. If the retrieved passages do not contain the requested information, the app explains that it has insufficient evidence.

## Sources

- `data/Nafe_Abubaker_CV.pdf`
- The README of [Abubaker-Nafe/ibt-ggateway-capstone](https://github.com/Abubaker-Nafe/ibt-ggateway-capstone)

The PDF loader preserves page numbers and hyperlinks. The GitHub loader preserves the repository name and README URL, and saves a local copy. A README has no PDF page number.

## Setup

Run commands from the `Task3_Task1` directory. Install the packages used by the scripts:

```cmd
python -m pip install streamlit pypdf langchain-text-splitters transformers sentence-transformers numpy faiss-cpu scikit-learn langchain-openrouter python-dotenv
```

Place the CV in the `data` directory. Create a `.env` file beside the scripts containing your own key:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
```

Keep `.env` out of version control. Model downloads, GitHub ingestion, and answer generation require an internet connection.

## Build the knowledge base

Run these commands in order:

```cmd
python load_documents.py
python load_github_readme.py
python chunk_documents.py
python create_embeddings.py
python build_vector_store.py
```

`load_documents.py` replaces the loaded-document file with the CV pages. When rebuilding both sources, run the GitHub loader after it. Repeating the GitHub loader refreshes that repository's README without adding a duplicate.

After a source or chunking setting changes, rerun chunking, embeddings, and vector-store construction. Retrieval uses the saved index; it does not automatically refresh source content.

## Ask a question

```cmd
python ask_question.py "What backend development experience does Nafe have?"
```

Another example:

```cmd
python ask_question.py "What were Nafe's primary and secondary roles in the student dropout prediction capstone?"
```

The default is three retrieved passages. To change that number:

```cmd
python ask_question.py "What backend development experience does Nafe have?" --top-k 5
```

Retrieval and generation can also run separately:

```cmd
python retrieve_documents.py "Which AWS certifications does Nafe hold?"
python generate_answer.py
```

`generate_answer.py` uses the question and passages most recently saved by retrieval. Its `--preview` option displays the prompt without sending an API request.

## Implementation

### Streamlit interface

Save `app.py` beside the other scripts and use the updated `retrieve_documents.py`. Once the vector store is built, install Streamlit and launch the interface:

```cmd
python -m pip install streamlit
python -m streamlit run app.py --server.fileWatcherType none
```

The interface displays the answer, retrieved source passages, and links to GitHub sources. Answers stay in the current browser session and can be downloaded as JSON. It reuses the existing generation function and caches the embedding model and vector store across questions. Rebuilding the store invalidates that cache on the next question.

For local use, the existing `.env` file supplies the API key.

### Pipeline settings

| Stage | Implementation |
| --- | --- |
| Ingestion | `pypdf` for CV text and PDF links; GitHub API for the README |
| Chunking | LangChain recursive splitter using the embedding model's tokenizer; target size 220 content tokens and overlap 30 tokens |
| Length validation | Each final chunk is checked against the 256-token limit, including special tokens |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, running on CPU; 384 numbers per chunk; normalized vectors |
| Vector storage | FAISS `IndexFlatIP`; inner product of normalized vectors gives cosine similarity |
| Retrieval | Hybrid ranking: 40% scaled positive semantic similarity and 60% scaled TF-IDF similarity; semantic-only fallback when no keyword matches exist |
| Generation | `ChatOpenRouter` with `apodex/apodex-1.1-mini:free`; JSON-object response mode; 60,000 ms timeout |
| Validation | Checks answer status, nonempty answered text, presence of citations, and whether citation labels belong to the retrieved passages |

The current two-source build produced 19 chunks: five from the CV and fourteen from the README. The longest chunk contained 221 tokens including special tokens. The embedding matrix had shape `(19, 384)`.

Hybrid retrieval was added because semantic retrieval missed the README's team-role table. Keyword matching brought that table into the top three for the primary/secondary-role question. The displayed retrieval score is a ranking value, not a probability that an answer is correct.

## Deployment

The web interface is deployed on **Streamlit Community Cloud** at:

[https://cellula-tech-task-3.streamlit.app/](https://cellula-tech-task-3.streamlit.app/)

The deployment uses the GitHub repository, with `app.py` as the application entry point. The hosted runtime needs the application scripts, a `requirements.txt` file, and these prebuilt knowledge-base files in their original relative locations:

- `output/vector_store/index.faiss`
- `output/vector_store/metadata.json`

Set the API key in Streamlit's **Secrets** settings using TOML syntax:

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key"
```

Keep `.env` and `.streamlit/secrets.toml` excluded from Git. The application supports the local `.env` file and hosted Streamlit Secrets; the deployed app does not need a committed `.env` file.

The app loads the saved knowledge base and caches the embedding model for subsequent questions. It does not run document ingestion or rebuild embeddings for each visitor. To publish updated source content, rebuild the vector store locally and push the updated index and metadata to the deployment repository. Intermediate embedding files and CLI question/answer outputs are not required for hosted inference.

## Output files

| File | Contents |
| --- | --- |
| `output/loaded_documents.json` | Extracted text and source metadata |
| `output/chunks.json` | Chunks, identifiers, and inherited metadata |
| `output/embeddings.npy` | Embedding matrix |
| `output/embeddings_metadata.json` | Embedding settings and matching chunks |
| `output/vector_store/index.faiss` | FAISS index |
| `output/vector_store/metadata.json` | Vector-to-chunk mapping and source metadata |
| `output/retrieval_results.json` | Latest question, passages, and retrieval scores |
| `output/answer.json` | Latest successful answer, model, and retrieved sources |
| `output/last_model_response.txt` | Latest received model text |
| `output/last_model_metadata.json` | Model/provider response details |
| `output/last_api_error.json` | HTTP error details when an API request fails |

Question and answer files are overwritten on subsequent runs. An unsuccessful generation can leave an older successful `answer.json`; check its question before using it. Diagnostic files describe their latest corresponding event and can also remain from earlier runs.

## Manual checks

During development, these behaviors were observed:

- A backend-experience question returned an answer with CV citations.
- A capstone-role question retrieved the team table and answered: primary roles **Project Coordinator and Data Lead**; secondary roles **EDA Lead and GitHub Lead**.
- An AWS-certifications question returned: **"I don't have enough information in the provided sources to answer this question."**

The backend and AWS checks were repeated successfully with the current generation model. These checks are examples, not a comprehensive accuracy evaluation.

## Current limits

Only the CV and this repository's README are ingested. LinkedIn, portfolio pages, other repository files, and linked PDF destinations are not automatically downloaded. Scanned PDFs without extractable text require an OCR step that is not implemented here.

Retrieval can miss relevant passages, and the model can still produce unsupported interpretations. Citation validation checks reference labels; it does not prove that each claim is supported by its cited text. An insufficient-evidence answer describes the retrieved context, not proof that a qualification or experience does not exist.

Free model availability and provider capabilities can change. This provider accepts `json_object` rather than `json_schema`. If generation fails, inspect the saved error details before changing the model or response format. The application does not automatically switch to a paid model.

BigBird is covered by the separate Task 0 research report. This RAG implementation uses MiniLM embeddings and does not train or run BigBird.
