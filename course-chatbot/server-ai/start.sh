#!/bin/sh

echo "🔍 Checking vector DB..."

if [ ! -d "chroma_db2" ]; then
  echo "⚡ chroma_db2 not found, building..."
  python ingest.py
else
  echo "✅ chroma_db2 already exists, skip build"
fi

echo "🚀 Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0  --port 8000