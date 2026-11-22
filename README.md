# InternMemory

InternMemory is a local-first engineering knowledge engine with persistent
memory, search, retrieval, and agent capabilities.

## Development setup

InternMemory requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install the optional local semantic-search dependencies:

```bash
python -m pip install -e ".[dev,semantic]"
```

## Run the API

```bash
uvicorn internmemory.main:app --reload
```

Open <http://127.0.0.1:8000/health> to check the service or
<http://127.0.0.1:8000/docs> to explore the generated API documentation.

Search for grouped, cited evidence:

```bash
curl --get http://127.0.0.1:8000/search/evidence \
  --data-urlencode "q=circuit breaker retry budget"
```

Ingest an allowlisted public documentation page:

```bash
curl -X POST http://127.0.0.1:8000/sources/ingest-url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://docs.python.org/3/library/asyncio.html"}'
```

Run keyword, semantic, and hybrid evaluations:

```bash
python -m internmemory.evaluation --split evaluation \
  --strategy grouped_reranked_thresholded
python -m internmemory.semantic_evaluation --split evaluation
python -m internmemory.hybrid_evaluation --split evaluation
```

## Run tests

```bash
pytest
```

## Optional Groq provider

The agent harness can use Groq's free hosted API through the
model-independent provider interface. Create a Groq API key, enable Zero Data
Retention in Groq Data Controls, and export the key only in your shell:

```bash
export GROQ_API_KEY="your-key"
```

Do not commit the key or put it in source code. `.env` files are ignored, and
`.env.example` contains only a placeholder. The default adapter model is
`qwen/qwen3.8-27b`; live evaluation is intentionally separate from unit tests
so `pytest` never spends tokens or requires network access.

Run only the fictional development split:

```bash
python -m internmemory.groq_evaluation
```

By default it runs only the development split and reports component quality,
primary failures, tokens, and model/tool latency as JSON. The frozen
evaluation split requires an explicit confirmation and must run as one
aggregate:

```bash
python -m internmemory.groq_evaluation \
  --model openai/gpt-oss-120b \
  --split evaluation \
  --confirm-frozen
```

Run one development case while diagnosing provider connectivity:

```bash
python -m internmemory.groq_evaluation \
  --case-id batching-tail-latency
```
