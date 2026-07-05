# JobRadar AI

An agentic pipeline that scrapes job boards, uses an LLM to extract a typed
schema from each posting, indexes them in Qdrant and ranks openings against a CV,
filtering senior roles with EU/Canada visa sponsorship.

It is the third project in a three-repo story: it consumes the
[cloudscraper-go](https://github.com/maarkN/cloudscraper-go) server to reach
anti-bot protected sources.

> **Status: M1 (extraction core).** The part the whole project rests on is proven
> first: `HTML -> validated Job JSON`, with retry, a cheap/strong model cascade,
> a content-addressed cache and token cost accounting. Everything is tested
> offline with the LLM mocked. Embeddings, Qdrant, the LangGraph agent, the API
> and the visa/eval work follow in M2-M5.

## Why this design

- **Typed schema as the contract.** Every posting becomes a Pydantic `Job`. The
  LLM only fills what it can see (`ExtractedJob`); provenance (source, url,
  timestamps, dedup hash) is added by the pipeline, never invented.
- **Validation + retry + cascade.** A parse or validation failure retries on the
  cheap model, then escalates to the strong model once, then raises. Trusting LLM
  output without measuring it is the thing this project refuses to do.
- **Cost is visible.** Every call records tokens and an estimated USD cost.
- **Cache by input.** Identical posting text is never re-sent to the LLM.
- **Pluggable providers.** `LLM_PROVIDER=anthropic|openai` selects the backend;
  both use a fast tier for bulk work and a strong tier for repair.

## Quickstart

```bash
uv sync
make check          # ruff + mypy (strict) + pytest, all offline

# Live extraction needs an API key (copy .env.example -> .env and fill it in):
uv run jobradar extract --html-file tests/fixtures/sample_job.html \
  --source demo --url https://example.com/job/1
```

To fetch a real anti-bot protected page, start the cloudscraper-go server in the
sibling repo (`go run ./cmd/server`) and add `--anti-bot`.

## Layout

```
src/jobradar/
  models.py          # Job / ExtractedJob / RankedJob, dedup content_hash
  config.py          # env-driven settings + provider selection
  cost.py            # token + USD accounting
  cache.py           # content-addressed extraction cache
  llm/               # provider protocol + Anthropic/OpenAI + factory
  extract/           # html_to_text, prompt, Extractor (retry + cascade)
  fetch.py           # PlainFetcher + AntiBotFetcher (cloudscraper-go)
  cli.py             # `jobradar extract`
tests/               # deterministic, LLM mocked, no network
```

## Roadmap

- [x] **M1 — Ingest + Extract:** clean text, LLM extract, validate, retry,
  cascade, cache, cost; CLI `extract`.
- [ ] **M2 — Embed + Qdrant + match vs CV:** hosted embeddings, vector index,
  top-K ranking, CLI `rank`.
- [ ] **M3 — LangGraph agent + more sources:** stateful graph with retry and
  checkpointing, cross-source dedup.
- [ ] **M4 — API + Docker Compose:** FastAPI, `docker compose up`, CSV export.
- [ ] **M5 — Visa filter + eval + dashboard:** visa classifier, labelled eval set
  with field-level accuracy, mini dashboard.

MIT licensed.
