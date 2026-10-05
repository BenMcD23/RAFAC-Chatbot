#!/usr/bin/env bash
# Re-ingest the local docs folder (on this PC's GPU) and replace the cluster's
# index with the result. Usage: ./push_index.sh   (needs kubectl access)
#
# Each upload goes into its own timestamped folder in the pod's volume; the
# /data/current symlink is then switched to it and the pod restarted to load
# it. The previous upload is kept so a bad one can be rolled back by hand.
set -euo pipefail
cd "$(dirname "$0")"

.venv/bin/python cli.py ingest

ts=$(date +%Y%m%d-%H%M%S)
echo "Uploading index as /data/$ts ..."
tar -C data -cz chroma | kubectl exec -i -n chatbot deploy/chatbot -- sh -c "
  set -e
  mkdir -p /data/$ts && tar -xz -C /data/$ts
  # Before the first upload the app creates an empty real directory here.
  [ -L /data/current ] || rm -rf /data/current
  ln -sfn /data/$ts/chroma /data/current
  ls -1d /data/2* | head -n -2 | xargs -r rm -rf"

kubectl rollout restart -n chatbot deploy/chatbot
kubectl rollout status -n chatbot deploy/chatbot --timeout=5m
kubectl exec -n chatbot deploy/chatbot -- python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
