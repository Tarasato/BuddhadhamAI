#!/bin/sh
# Start Ollama in background
ollama serve & 

sleep 1

ollama pull nomic-embed-text:v1.5 & ollama pull gpt-oss:20b &

# Start Python app
python main.py

# Wait for background jobs (optional)
wait