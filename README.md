# Supply chain intelligence

A small HTTP service over three shipment CSVs: data-quality checks, a curated Postgres snapshot, a booking-time delay model, and a command-line assistant that calls tools.

## Getting started

```bash
docker compose up --build
```

The API listens on port 8000. Postgres is published on port 5432 (`sci` / `sci`, database `sci`). The API container does not need a model API key.

```bash
curl -s localhost:8000/health
curl -s localhost:8000/shipments/SHP-00623
curl -s localhost:8000/routes/Shanghai/Rotterdam/stats
curl -s -X POST localhost:8000/predict-delay -H 'content-type: application/json' -d '{"shipment_id":"SHP-00623"}'
curl -s localhost:8000/data-quality/report
```

Quality checks read the CSVs and do not need Postgres:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m app.dq --input data/raw
pytest
```

The assistant runs on the host. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL` to your ChatOpenAI-compatible endpoint.

```bash
export DATABASE_URL=postgresql+psycopg://sci:sci@localhost:5432/sci
export API_BASE_URL=http://localhost:8000
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
python -m app.assistant "Which routes have had the highest average delay this quarter?"
```

"This quarter" is the latest quarter of `booking_date` in the file (2025 Q4, bookings through 14 Oct 2025), not the wall clock. The tool returns that window, and the answer should name it.

## Architecture diagram

```mermaid
flowchart LR
  csvs[raw CSVs]
  dq[dq checks]
  pipe[pipeline]
  pg[(Postgres)]
  api[FastAPI]
  model[joblib model]
  assistant[LangGraph CLI]
  csvs --> dq
  csvs --> pipe
  pipe --> pg
  api --> pg
  api --> model
  assistant --> pg
  assistant -->|predict_delay only| api
```

Raw files are never modified. One command, `python -m app.pipeline`, replaces the curated tables and the quarantine tables in a single transaction. The API reads those tables. `query_shipments` and `get_route_stats` read the same tables. Only `predict_delay` calls the HTTP endpoint.

## Tech choices

Postgres is more machinery than 5,000 shipments need. It is here for foreign keys, check constraints, and one server shared by the API and the assistant. At 100 times the volume the same database still fits. At millions of events, partition `port_events` by month and stop wiping the tables on every boot. DuckDB would have been the smaller query engine. It would not have given the API a shared server with constraints.

Other choices:

- FastAPI, because the service is a handful of JSON endpoints.
- scikit-learn logistic regression with `class_weight="balanced"`. The held-out metrics are weak on purpose of the features, and they are reported as measured: precision 0.41, recall 0.49, F1 0.45, AUC 0.51 on the latest 20% of bookings. A fancier model would not change the method.
- ChatOpenAI pointed at `OPENAI_BASE_URL`. Cost per query is that endpoint's price for the tokens recorded in `logs/llm_calls.jsonl`. There is no fixed vendor price in this repo.
- pytest for unit tests and for HTTP tests against a real server. ruff for lint. GitHub Actions for CI.
- No UI, no auth, no Alembic, no Kafka, no vector store, no LangSmith, no Kubernetes. The curated schema is created from the SQLAlchemy models because this volume is rebuilt from the files.

`late_score` is a ranking score, not a calibrated probability. `class_weight="balanced"` reweights the loss, and the decision threshold stays at 0.5.

The model is trained only on shipments with a real arrival, so it does not predict cancellation. A cancelled shipment that survived quarantine can still be scored. Features are origin, destination, cargo, container count clipped at 40, weight, planned transit days, both ports' congestion scores, and booking month. `status` is excluded because on the canonical rows it matches the 24-hour label. Actual timestamps and port events are excluded because they are not known at booking.

In production, watch the share of scores above 0.5 against the training base rate, and precision and recall once `actual_arrival` is known. Retrain when that AUC drops by more than 0.05, on a schedule, or when a new port code appears.

## Data quality summary

`python -m app.dq --input data/raw` writes `data/quality/report.json`. `GET /data-quality/report` runs the same function. Percents use the file the check belongs to (5,028 shipments, 25,000 events, 25 ports).

| Check | Rows | Severity | Action |
| --- | ---: | --- | --- |
| Conflicting duplicate shipment id | 40 | critical | Quarantine both rows. |
| Unknown port (`XXTST`, `ZZZZZ`, `UNKNW`) | 95 | critical | Quarantine the shipment. |
| Conflicting duplicate event id | 206 | critical | Quarantine every row with that id. |
| Status outside `DELIVERED`, `DELAYED`, `CANCELLED` | 146 | warning | Map `delivered` to `DELIVERED`, `Complete`/`COMPLETED` to `COMPLETED`, blank and `N/A` to `UNKNOWN`. |
| Cargo case, whitespace, or blank | 276 | warning | Canonical label. Blank stays null. |
| Container count 0 | 23 | warning | Flag and keep. |
| Container count above 40 | 17 | warning | Flag and keep. The model clips the feature at 40. |
| Missing weight | 73 | warning | Leave null. |
| Negative weight | 36 | warning | Set to null. |
| Missing actual departure and arrival | 349 | warning | Leave null. Do not impute. |
| Actual departure before booking | 39 | warning | Flag and keep. |
| `BEANR` missing country | 1 | warning | Flag. Do not fill a country that is not in the file. |
| Blank event type | 140 | warning | Keep the row, type null. |
| Sentinel vessels `VSL-0000`, `VSL-9999` | 207 | warning | Flag and keep. |
| Blank event notes | 1,913 | info | Null is allowed. |
| Early arrival | 950 | info | None. An early ship is on time. |
| Identical duplicate shipment id | 16 | warning | Keep one copy. |

Curated shipments: 4,886. Quarantined shipment rows: 134. Curated events: 24,794. Quarantined event rows: 206.

Derived columns on `shipments`: `actual_delay_hours` (null if either arrival timestamp is missing), `on_time_flag` (true when delay is at most 24 hours, null when delay is null), `route_key` (`ORIGIN → DEST`, a label, not a join key), `transit_days_planned`, `transit_days_actual`.

Timestamps are `timestamp without time zone`. The files have no offset. Notes are `text`. `weight_tons >= 0`, `delay_minutes >= 0`, congestion is between 0 and 1. Curated shipment and event ports have foreign keys to `ports`. Quarantine tables do not: an unknown port is why many of those rows exist.

## Production support runbook

After a deployment, `GET /health` returns 200 with a version and `checks.database`. `ok` means the process should receive traffic. `down` means it should not. The HTTP status stays 200 so a database blip is not treated as a dead process.

Likely failures:

- The loader crashes and the container exits. The previous snapshot remains, because the load is one transaction. Start the container again. The usual cause is Postgres not ready or a check constraint rejecting a row the cleaner should have removed.
- `POST /predict-delay` returns 503. The artefact is missing or was saved with a different scikit-learn than the pinned `1.9.1`. The process still starts.
- The assistant times out on `OPENAI_BASE_URL` while `/health` stays up. The API does not call that endpoint. Check the key, the base URL, and `logs/llm_calls.jsonl`.

A bad local release is the previous git revision and `docker compose up --build`. A deployed rollback is the previous image tag. The loader is not part of that story once it is a separate job: a second API replica must not run it, because the reload takes an exclusive lock.

Next for a real deployment: a metrics endpoint and an alert on `/health` database `down` and on predict 503 rate, secrets in a secret store rather than an env file, and the loader moved out of the API process.

## CI/CD

`.github/workflows/ci.yml` installs pinned dependencies, runs ruff, runs the full pytest suite against a Postgres service container, and builds the image in a separate job. Integration tests fail if `DATABASE_URL` is unset. The image does not copy `.env`.

After the private repository exists, add this badge, with the real owner and name:

```markdown
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
```

Quality gates: ruff and pytest pass before merge. The image build passes before a release tag is cut. A change reaches a running service by building the image from the merged commit and replacing the running container. Rollback is the previous image tag, as in the runbook.

## If I had more time

Move the loader to a one-shot job that swaps staging tables in, so a restart of the API does not rewrite the data. Add a time-based backfill for events instead of a full wipe. Calibrate `late_score` so it can be read as a probability.

## Scaling to production

What breaks first is the full-table reload. Two API processes must not run it. The lock pauses reads, and at millions of event rows the pause becomes an outage. Replace it with a load into staging tables and a rename. Treat `port_events` as an append-only feed. `event_id` cannot be the idempotency key: 206 rows share an id with a different payload.

At 100 times the current volume, Postgres stays. Add the route and booking indexes only if a query shows up slow; those two indexes are already there. Offset pagination gets expensive; switch the list endpoint to a keyset on `shipment_id`.

At millions of shipments, 24 hours a day: partition `port_events` by month, keep scoring on a copy of the model loaded in the API process, and retrain offline. The model file is loaded at startup, not on each request. A new port code is ignored by the one-hot encoder and is a reason to retrain. The assistant stays a client of the API for predictions so a model failure is a 503, not a number invented in the prompt.
