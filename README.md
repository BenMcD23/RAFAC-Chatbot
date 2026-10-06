# RAFAC docs RAG chatbot (PoC)

Ask questions over the controlled-documents library. Answers come only from retrieved chunks and cite SharePoint links.

## Setup (WSL2 Ubuntu)

Don't install NVIDIA drivers or CUDA inside WSL; the Windows driver is enough.

```bash
cd chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA build (needed for RTX 50xx)
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"             # should print True

cp .env.example .env    # then set NVIDIA_API_KEY (and DOCS_DIR if the docs aren't in ./docs)
```

## Ingest

```bash
python cli.py ingest
```

- Indexes `.pdf` and `.docx` under `DOCS_DIR`. Other files are logged and counted, and `manifest.json` is ignored.
- `config.yaml` → `exclude_patterns` drops files by filename glob (case-insensitive), e.g. `*Form*`.
- Re-runs are incremental: only new or changed files are re-embedded (tracked in `data/state.json`), and deleted or newly excluded files are removed from the index.
- The summary lists failed files and files with no extractable text (scanned PDFs that need OCR).
- To rebuild from scratch, run `rm -rf data/`.
- `INDEX_VERSION` in `ingest.py` goes up when what's stored per chunk changes, and the next ingest then redoes every file. Version 2 added per-paragraph page/para positions and title-prefixed embeddings, so the first `./push_index.sh` after upgrading re-embeds everything.

## Ask

```bash
python cli.py ask "What is the minimum age for an adult volunteer?"
uvicorn app:app --host 0.0.0.0 --port 8000      # then open http://localhost:8000
```

Endpoints: `GET /`, `POST /ask {"question": "..."}`, `GET /health`.

### How an answer is built

1. The question is embedded and the 25 nearest chunks are pulled from Chroma. Each chunk is embedded with its document title and section on top, so questions that name a document find it.
2. A cross-encoder (`RERANK_MODEL`, default `cross-encoder/ms-marco-MiniLM-L6-v2`) re-scores those 25 against the question, and the best 6 go to the LLM. Fewer, better chunks give better answers and use fewer tokens of the provider's daily limit. If the reranker can't load, the embedding order is used.
3. The LLM replies in JSON: an answer with `[n]` citations, plus a verbatim quote from each source it cites (`cite.py`).
4. Each quote is checked against the chunk text. A quote that isn't really there is dropped, so a highlighted line is always one the document contains. The quote's character offsets are mapped back to the page and paragraph number it came from.
5. Only cited sources are returned, renumbered in order of use. If nothing answers the question, `related` lists the closest documents instead.

`POST /ask` returns:

```json
{
  "answer": "Cadets may stay overnight with Wing approval [1].",
  "found": true,
  "sources": [{
    "n": 1, "filename": "ACP 20.pdf", "location": "page 14, para 12",
    "url": "https://rafac.sharepoint.com/...ACP%2020.pdf#page=14",
    "snippet": "...", "score": 7.1, "text": "<the whole chunk>",
    "quotes": [{"start": 120, "end": 188, "text": "...", "location": "page 14, para 12", "page": 14}]
  }],
  "related": []
}
```

PDF links carry `#page=N`, which browser PDF viewers open at that page.

## Tests

```bash
pip install pytest ruff
pytest -q
ruff check --isolated --line-length 120 --select E,F,W,I .
```

The tests ingest a real Word document and PDF, then answer with a faked LLM. The models are faked in `conftest.py`, so the tests never download anything or call a provider. CI runs them on every PR and before every deploy.

## Eval

Put one question per line in `questions.txt`, then run `python eval.py`. For each question it prints the top-3 retrieved files with their scores, then the answer and sources.

## Switching to Ollama

Edit `.env` only:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.1:8b
```

Ollama on Windows is reachable from WSL at `localhost` if WSL networking is in mirrored mode. Otherwise, use the Windows host IP.

## Notes

- Chunks are about 500 tokens with 100 tokens of overlap, because `bge-base-en-v1.5` truncates at 512 tokens. If you switch `EMBED_MODEL` to a different model, delete `data/` and re-ingest.

## Deployment (k3s)

The service runs in the cluster as `chatbot` in namespace `chatbot`, on home. Argo CD deploys it from `deploy/`, using the Application in the k8s repo at `clusters/prod/chatbot.yaml`. A push to `main` builds `ghcr.io/benmcd23/chatbot` and bumps the tag in `deploy/kustomization.yaml`. The service is not public: the SMS site's Docs Assistant page calls the SMS API (`POST /chat/ask`, staff only), and the API forwards the question here. A NetworkPolicy only lets the SMS API in.

The image holds code and the embedding model, never documents. The index lives in a volume in the cluster and is built on this PC.

### Updating the documents

1. Pull the latest docs from SharePoint into `docs/` (manual for now).
2. Run `./push_index.sh`.

The script re-ingests locally on the GPU (only new or changed files), uploads `data/chroma` into the pod's volume, switches `/data/current` to the upload and restarts the pod. It keeps the previous upload, so you can roll back: `kubectl exec -n chatbot deploy/chatbot -- ls /data`, then point `current` at the older folder with `ln -sfn`, and restart the deployment.

The Groq key is a SealedSecret in `deploy/sealed-secret.yaml`. To change it:

```bash
kubectl create secret generic chatbot-secrets -n chatbot --from-literal=LLM_API_KEY=gsk_... --dry-run=client -o yaml \
  | kubeseal --controller-namespace kube-system --controller-name sealed-secrets-controller -o yaml > deploy/sealed-secret.yaml
```

The model and base URL are set in `deploy/kustomization.yaml`.
