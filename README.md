# llm-evaluation

### Ollama
```
docker pull ollama:latest
docker network create llm-net
docker run -d --rm --name ollama -p 11434:11434 --network llm-net ollama/ollama:latest
docker exec -it ollama ollama pull qwen3:1.7b
```


### LLM Evaluation
```
docker build . -t llm-evaluation
docker run --rm -it --network llm-net llm-evaluation python /app/evaluate_llm.py \
-u http://ollama:11434 \
-m qwen3:1.7b
```
