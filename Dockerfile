FROM python:3.12-slim
WORKDIR /app

# CPU torch: the cluster only embeds one question per request, and the CUDA wheels are gigabytes.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding and reranking models into the image so the pod never calls Hugging Face.
ENV HF_HOME=/app/hf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY cite.py rag.py app.py index.html ./
USER 10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
