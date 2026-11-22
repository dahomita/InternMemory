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

## Run tests

```bash
pytest
```
