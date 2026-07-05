# JobRadar AI

An agentic pipeline that scrapes job boards, uses an LLM to extract a typed
schema from each posting, indexes them in Qdrant and ranks openings against a CV,
filtering senior roles with EU/Canada visa sponsorship.

It is the third project in a three-repo story: it consumes the
[cloudscraper-go](https://github.com/maarkN/cloudscraper-go) server to reach
anti-bot protected sources.

> **Status: M1 - M4.** `HTML -> validated Job JSON` (retry, cheap/strong model
> cascade, content-addressed cache, token cost) -> semantic ranking against a CV
> (hosted embeddings, Qdrant, explainable score with visa filtering) -> a
> **LangGraph** ingest pipeline (per-node failure isolation, retry, checkpointing,
> cross-source dedup) -> a **FastAPI** service + `docker compose up` + CSV export.
> Everything is tested offline: the LLM is mocked and Qdrant runs in its in-process
> local mode, so CI needs no network or Docker. The visa classifier, a labelled
> eval set and a dashboard follow in M5.

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

Run the full flow — ingest a set of sources through the LangGraph pipeline into a
job corpus, then rank it against a CV (both need an `OPENAI_API_KEY`; Qdrant runs
in local mode unless you set `QDRANT_URL`):

```bash
uv run jobradar ingest --sources-file examples/sources.json --out jobs.jsonl
uv run jobradar rank --jobs-file jobs.jsonl --cv-file cv.md --require-visa
```

## Layout

```
src/jobradar/
  models.py          # Job / ExtractedJob / RankedJob, dedup content_hash
  config.py          # env-driven settings + provider selection
  cost.py            # token + USD accounting
  cache.py           # content-addressed extraction cache
  llm/               # provider protocol + Anthropic/OpenAI + factory
  extract/           # html_to_text, prompt, Extractor (retry + cascade)
  embeddings.py      # hosted (OpenAI) embedder behind a protocol
  index.py           # QdrantJobIndex (local mode in tests, server in prod)
  rank.py            # Ranker: similarity + explainable rule boosts
  store.py           # JSONL job corpus + exact-key dedup
  sources.py         # Source config + per-source fetcher selection
  graph/             # LangGraph ingest pipeline (fetch->parse->extract->index)
  api/               # FastAPI app, deps (overridable), run registry
  export.py          # CSV export (applications-tracker format)
  fetch.py           # PlainFetcher + AntiBotFetcher (cloudscraper-go)
  cli.py             # extract / ingest / rank / export / serve
tests/               # deterministic: LLM mocked, Qdrant local, no network
```

## API

`jobradar serve` (or `docker compose up`) exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + indexed job count |
| `POST /runs` / `GET /runs/{id}` | trigger a background ingest and poll it |
| `GET /jobs` | list/filter indexed jobs (country, seniority, visa, remote) |
| `GET /jobs/{content_hash}` | one job |
| `POST /rank` | rank the corpus against a CV |
| `POST /export/csv` | top matches as a tracker-ready CSV |

## Roadmap

- [x] **M1 — Ingest + Extract:** clean text, LLM extract, validate, retry,
  cascade, cache, cost; CLI `extract`.
- [x] **M2 — Embed + Qdrant + match vs CV:** hosted embeddings, Qdrant index
  (local mode in tests), top-K ranking with explainable score + visa filter,
  CLI `rank`.
- [x] **M3 — LangGraph agent + more sources:** stateful ingest graph
  (fetch->parse->extract->index) with per-node failure isolation, fetch retry,
  a checkpointer and cross-source dedup (exact key + embedding similarity). CLI
  `ingest`.
- [x] **M4 — API + Docker Compose:** FastAPI (runs, jobs, rank, CSV export),
  `docker compose up` (app + Qdrant), CLI `export` / `serve`.
- [ ] **M5 — Visa filter + eval + dashboard:** visa classifier, labelled eval set
  with field-level accuracy, mini dashboard.

MIT licensed.
