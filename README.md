# Real-Time Transaction Scoring

[![tests](https://github.com/JAYANSHUBADLANI/realtime-transaction-scoring/actions/workflows/pytest.yml/badge.svg)](https://github.com/JAYANSHUBADLANI/realtime-transaction-scoring/actions/workflows/pytest.yml)

Score financial transactions for fraud as they happen, not in an overnight
batch. This repo is being built in stages: the batch scoring pipeline below
is complete and evaluated against real fraud labels; a streaming path
(Pub/Sub, a Cloud Run scorer, BigQuery, a live dashboard) is the next stage,
tracked in [Project status](#project-status).

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streaming](https://img.shields.io/badge/streaming-live%20on%20Cloud%20Run-brightgreen.svg)](#live-deployment)

## Problem statement

A small fraction of financial transactions are fraudulent, and reviewing
every transaction by hand does not scale. This project screens every
transaction with several complementary, interpretable methods, raises
alerts using both data driven detectors and hard business rules, and ranks
the resulting incidents by severity so the most suspicious cases are
reviewed first. Unlike the unlabeled dataset this project started from, the
data used here (PaySim, see [Dataset](#dataset)) carries a real fraud label,
so every detector below is evaluated by precision, recall, and lift, not
just by how many rows it flags.

The emphasis stays on clean, transparent statistics rather than heavy deep
learning. The statistical detectors are the core; Isolation Forest is a
complementary multivariate cross check.

## Methodology

Four detection methods are used, each interpretable and each catching a
different kind of anomaly:

- **Z-score.** `z = (x - mean) / std` on each screened numeric column;
  points beyond a threshold (default 3 standard deviations) are flagged.
- **IQR (Tukey fences).** Flags values outside `Q1 - 1.5*IQR` /
  `Q3 + 1.5*IQR`. Quantile based, so it does not assume a particular
  distribution, though see [What did not work](#what-did-not-work) for a
  real limitation of this found during evaluation.
- **Control chart (EWMA).** Transactions are aggregated into a daily time
  series (default: total amount per day). An EWMA center line plus control
  limits at a multiple of the residual standard deviation flag days that
  fall outside them.
- **Isolation Forest.** An unsupervised ensemble that isolates points by
  random partitioning; anomalies isolate in fewer splits. Unlike the
  column-by-column statistical methods, it can catch rows that are unusual
  only in combination across features.

On top of these, two engineered features and one business rule come from
double-entry bookkeeping logic on the raw ledger fields, not from the
label (see `src/load_data.py::engineer_features`):

- **`balance_error_dest`** = `oldbalanceDest + amount - newbalanceDest`.
  Nonzero when money left the sender but did not fully land in the
  recorded recipient balance.
- **`balance_error_orig`** = `oldbalanceOrg - amount - newbalanceOrig`.
  Ledger consistency on the sender's side. Computed but, as detailed below,
  deliberately *not* used by the z-score/IQR screens.
- **`orig_account_emptied`** rule: fires when the sender's balance was
  positive before the transaction and exactly zero after. The strongest
  single signal found during evaluation, see [Results](#results).

**A scope decision made from evidence, not intuition.** EDA on the full
6,362,620-row PaySim file found that fraud only ever occurs within the
`TRANSFER` and `CASH_OUT` transaction types: zero fraudulent rows among the
6,354,407 `PAYMENT` / `CASH_IN` / `DEBIT` rows. This pipeline scores only
`TRANSFER`/`CASH_OUT` transactions (2,770,409 of 6,362,620, 43.5% of the
file) by design. On this dataset that cuts real-time scoring volume by
56.5% with zero recall cost, which will matter directly for Pub/Sub message
counts and Cloud Run invocations once the streaming stage lands. This is a
property of *this* simulator's fraud injection model, confirmed empirically
here rather than assumed, and would need periodic revalidation against real
fraud patterns in a production system rather than being trusted forever.
That caveat is the point of saying "confirmed empirically" instead of
"fraud only happens in transfers."

The alerting engine combines every detector flag with the business rules
from `config.yaml` (amount over a threshold, the account-emptied rule).
Each alert records exactly which methods and rules fired. The
prioritization step blends normalized deviation, detector agreement, and
rule hits into a single 0-1 severity score, then buckets into High/Medium/Low.

## Results

Every detector below is fit once on a **reference window** (the first 96
simulated hours, ~4 days, 463,152 rows) and then scored against the
**held-out population that comes strictly after it** (the remaining ~27
simulated days, 2,307,257 rows), never the other way around; see
[Reference-based scoring](#reference-based-scoring-no-look-ahead) for why
that split exists. Reproducible with `python main.py` (about 50 seconds),
from `reports/evaluation_metrics.json` and `reports/run_summary.json`. The
label (`isFraud`) is used only for this evaluation table, never fed into
fitting or scoring.

| | |
|---|---:|
| Reference window rows (fit only, never scored) | 463,152 |
| Scored population rows | 2,307,257 |
| Real fraud rate in scored population | 0.306% (7,061 rows) |
| Total alerts raised (any signal fired) | 1,149,601 |
| High / Medium / Low severity | 41,929 / 87,776 / 1,019,896 |

**Per-detector precision / recall, unsupervised, no label ever seen by the detector:**

| Detector | Precision | Recall | F1 | PR-AUC | Fraud caught |
|---|---:|---:|---:|---:|---:|
| `orig_account_emptied` rule | 0.69% | **97.58%** | 0.0138 | n/a | 6,890 / 7,061 |
| IQR (amount, balance_error_dest) | 1.77% | 71.36% | 0.0346 | 0.0547 | 5,039 / 7,061 |
| Isolation Forest | 2.87% | 15.96% | 0.0487 | 0.0171 | 1,127 / 7,061 |
| Z-score (amount, balance_error_dest) | 2.51% | 27.94% | 0.0460 | 0.0201 | 1,973 / 7,061 |
| PaySim's own built-in `isFlaggedFraud` | 100.00% | 0.23% | 0.0045 | n/a | 16 / 7,061 |

**Ranked-queue lift** (combined score = max of the three normalized
detector scores, over the scored population):

| Top reviewed | Fraud caught | Precision at k | Lift over base rate |
|---|---:|---:|---:|
| 0.1% (2,307 rows) | 1.7% of all fraud | 0.52% | 1.7x |
| **1% (23,073 rows)** | **8.7% of all fraud** | **2.67%** | **8.7x** |
| 5% (115,363 rows) | 38.8% of all fraud | 2.38% | 7.8x |

The honest read: a single, cheap ledger-consistency rule
(`orig_account_emptied`, one comparison, no model) catches more fraud than
any of the three heavier statistical/model detectors individually, and
nearly ties the union of all three combined (see
[What did not work](#what-did-not-work), 71.5%). It is not high precision
on its own (99.3% of what it flags is not fraud), which is why it is one
signal feeding a blended severity score rather than an auto-block rule, but
as a recall-maximizing first pass it is the strongest single thing found
here.

### Reference-based scoring, no look-ahead

The batch pipeline this project started from fit z-score means/IQR fences
and the Isolation Forest on the exact same data being scored, which is
fine for a one-shot report but cannot work for streaming: a transaction
arriving right now cannot wait for the mean of transactions that have not
happened yet. `src/reference.py` splits this into a `fit_reference()` call
(reference window only) and a `score_with_reference()` call (scored
population, or a single incoming transaction, against the frozen result of
that fit). `main.py`'s batch run and a future streaming service call the
identical `score_with_reference()` function, which is what makes their
outputs provably identical rather than merely similar;
`tests/test_streaming_equivalence.py` proves this directly: it fits one
reference, scores a batch of 40 transactions all at once, scores the same
40 one row at a time through the same reference, and asserts the two are
bit-for-bit identical, then separately proves a detector's scores change
when it is fit against a *different* reference (ruling out the scorer
silently ignoring the reference and falling back to the old leaky
behavior) and that padding a small batch with 200 extra rows does not
change the small batch's own scores (ruling out any remaining dependence
on which other rows happen to be scored alongside a given one).

Splitting fit from scoring this way changed the actual numbers, not just
the code path, which is itself worth reporting: see the first point in
[What did not work](#what-did-not-work).

**A bug found and fixed while building this, not caught by any test at the
time:** `src/alerting.py::build_alerts` still listed the earlier
unlabeled-dataset scaffold's column names (`TransactionID`,
`TransactionAmount`, `Channel`, ...) as the context columns to carry onto
each alert. None of those exist in PaySim, and the lookup silently no-ops
when a column is absent rather than erroring, so every row in
`alerts.csv`/`incident_queue.csv` had detector flags and scores but zero
way to tell which real transaction was being flagged: no amount, no type,
no account IDs. The numbers in this README were never wrong (evaluation
reads `isFraud` and the detector columns directly, none of which touched
the missing context columns), but the incident queue this project's
problem statement promises an analyst could act on was, until this was
found and fixed, not actually actionable. Fixed by updating the context
column list to PaySim's real schema.

## What did not work

Three honestly-reported findings from evaluating against the real label,
not hidden or quietly retuned away:

1. **Leakage was not just theoretically wrong, it measurably inflated how
   noisy IQR looked.** An earlier version of this pipeline fit IQR fences
   on the same population being scored (fences pulled open by the very
   outliers, fraud included, that they were then asked to detect).
   `balance_error_dest` alone, under that leaky fit, flagged 42.1% of the
   population for 0.46% precision. Refitting the exact same fences on a
   clean historical reference window instead (the first 96 simulated
   hours, never scored) and applying them to the held-out population after
   it cut the flagged share to **5.0%** and nearly quadrupled precision to
   **3.08%**, with comparable recall (50.7%). This is one of the clearest
   pieces of evidence in this project that the reference-based refactor
   (see [Reference-based scoring](#reference-based-scoring-no-look-ahead))
   was not just a streaming-readiness formality: it changed what the
   detector appeared to do. A residual limitation remains even after the
   fix: Tukey fences assume roughly-normal data, and financial ledger
   differences are heavily right-skewed, so IQR is still the noisiest of
   the three statistical/model detectors in the table above. A quick
   diagnostic with `log1p(amount)` confirmed there is a real
   precision/recall tradeoff here rather than a free second win: a wider
   `k` on log-transformed amount cuts flagged rows sharply but also cuts
   recall, since the transform compresses exactly the large-amount tail
   fraud lives in.
2. **`balance_error_orig` looks discriminative in summary statistics but is
   useless to an extremity-based detector, confirmed under the same
   reference-window fit.** Fraudulent rows sit almost exactly at
   `balance_error_orig = 0` (median 0.0 within TRANSFER/CASH_OUT), inside
   the reference-fit fences (-636,036 to 299,249), while the bulk of
   legitimate rows sit further out (median -143,657, also technically
   inside those wide fences but nowhere near the center). Fraud is "too
   clean" on this column, not extreme, so z-score and IQR structurally
   cannot separate it here regardless of how the fences are fit: an
   earlier diagnostic with this column included in the screens added 7.8%
   of the population to the flagged set for 0.01% precision, pure noise.
   It was removed from `statistical.zscore.columns`/`statistical.iqr.columns`
   in `config.yaml` for that reason (kept only as an Isolation Forest
   feature, where multivariate combinations can still use it) rather than
   silently left in to inflate an ensemble count.
3. **The EWMA control chart never fires** (0 of 31 days flagged). PaySim's
   per-simulation-step transaction volume is extremely bursty (2 to 51,352
   rows in a single step), so daily aggregated totals swing by up to 90x
   day to day. The chart's single global residual standard deviation gets
   inflated by that burstiness, producing control limits wide enough that
   nothing ever breaches them across 31 days. A rolling (windowed) sigma
   instead of a single global one would likely fix this; that is a real
   next step, not implemented here, stated plainly rather than swapped in
   quietly to make the number look better.

## Streaming service (local, no GCP dependency yet)

`service/main.py` is a FastAPI app implementing Google Cloud Pub/Sub's
[push subscription contract](https://cloud.google.com/pubsub/docs/push):
`POST /push` accepts the exact JSON envelope Pub/Sub sends (a base64
encoded, batched payload), scores every transaction in it against the
frozen reference artifact loaded once at startup, applies the alerting
rules and severity scoring, and persists every scored transaction (alert
or not) to SQLite, deduplicating on a synthesized `transaction_id` since
Pub/Sub push delivery is at-least-once, not exactly-once.

**No real Pub/Sub topic or the official emulator is used yet, by
deliberate choice, not an oversight.** The emulator needs a JVM this
machine did not have installed, and installing one just to replay a
well-documented, stable JSON contract seemed like the wrong trade: instead,
`tests/test_service.py` uses FastAPI's `TestClient` to drive the exact same
ASGI request-handling code path a real deployment will run, with a tiny
synthetic reference fitted fresh per test so the suite stays fast (6 tests
covering scoring, the `orig_account_emptied` rule firing, deduplication on
message redelivery, and a malformed-payload request returning 400 rather
than crashing). `scripts/local_demo_replay.py` then exercises the same
service as a real running process over real HTTP, against the actual
463,152-row reference artifact `python main.py` produced, not a synthetic
one:

```
2,000 real held-out transactions scored across 200 requests in 3.2s (617.7 txn/s)
Alerts raised: 969 (48.4% of scored)
Per-request latency (batch of 10): mean=15.9ms  p50=15.9ms  p95=16.9ms  p99=17.7ms
```

The 48.4% alert rate looks high next to the 0.306% real fraud rate in
[Results](#results); that gap is expected and already explained there, not
a new finding: `orig_account_emptied` alone fires on 42.7% of *legitimate*
TRANSFER/CASH_OUT transactions too (see the Results table), so a small,
sequential, unfiltered sample of live-looking traffic naturally alerts
often. That is exactly why this project treats it as one signal feeding a
blended severity score, not an auto-block rule.

## Live deployment

Deployed for real on Google Cloud (`infra/setup.sh` is the exact,
re-runnable sequence used, parameterized by project ID rather than hard
coding one): a real Pub/Sub topic and push subscription, a private Cloud
Run service (`--no-allow-unauthenticated`, reachable only by a dedicated
push-auth service account, not the public internet, so no public URL is
published here, there would be nothing a reader could do with it),
and BigQuery. `publisher/replay.py` published real held-out PaySim
transactions to the actual topic; Pub/Sub push-delivered them to Cloud
Run; the service scored them against the same frozen reference as the
local demo and wrote them to BigQuery. Verified by querying BigQuery
directly, not by trusting the publisher's own success message:

```sql
SELECT COUNT(*) total, SUM(is_alert) alerts
FROM `<project>.realtime_scoring.scored_transactions`
-- 500 total, 334 alerts, all published rows accounted for
```

**A real, measured latency finding, not a flattering number:** a single,
non-concurrent push (one dedup check plus one BigQuery streaming insert)
takes about 0.92s end to end, already well above the 15.9ms this project's
local SQLite-backed demo measured. Under a realistic burst (50
near-simultaneous pushes, the same pattern `publisher/replay.py` sends),
latency degrades further: mean 9.17s, p50 9.23s, p95 11.08s. A second,
smaller batch of pushes that all turned out to be duplicates (dedup-only,
no insert) took about 0.6-0.85s each, isolating the actual bottleneck: it
is specifically BigQuery's classic streaming insert API
(`insert_rows_json`, what `service/sink_bigquery.py` uses) under
concurrent write contention on a fresh table/partition, not Pub/Sub
delivery, not the scoring logic, and not the dedup query alone. This is
exactly the kind of thing the local-only demo evidence could not have
shown, and exactly why this README does not present the 15.9ms SQLite
number as if it were what the live deployment achieves. The documented,
not-yet-implemented next step is the modern BigQuery Storage Write API,
which is built for sustained concurrent write throughput in a way the
streaming insert API this project uses is not.

## Dataset

**PaySim: Synthetic Financial Datasets For Fraud Detection**
(`ealaxi/paysim1` on Kaggle), 6,362,620 simulated mobile-money transactions
over 743 simulated hours, with a real `isFraud` label. Not committed to
this repo (~470 MB); see [`data/README.md`](data/README.md) for three
download options, including a `kagglehub` script confirmed to work with
zero Kaggle credential setup.

| Column | Description |
| --- | --- |
| `step` | Simulation hour index, 1-743. Not a wall-clock time; `load_data.py` derives a synthetic `event_time` from it, disclosed rather than presented as real. |
| `type` | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT`. Scoring is restricted to `TRANSFER`/`CASH_OUT`, see Methodology. |
| `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | Transaction amount and sender/recipient balances before and after. |
| `isFraud` | Ground truth label, used only for evaluation. |
| `isFlaggedFraud` | PaySim's own built-in rule (fires on 16 of 6.36M rows), kept as a weak baseline. |

## Project status

This repo is a work in progress, built in stages so each one can be
evaluated on its own before the next depends on it:

- [x] **Batch scoring pipeline**, evaluated against real fraud labels
  (this README's Results section).
- [x] **Reference-based refactor**: z-score/IQR/Isolation Forest now fit
  once on a frozen reference window and score any population (a 2.3M-row
  held-out batch or a single incoming transaction) against that same
  frozen state, with `tests/test_streaming_equivalence.py` proving batch
  and one-row-at-a-time scoring are bit-for-bit identical. See
  [Reference-based scoring](#reference-based-scoring-no-look-ahead). The
  frozen reference is persisted to `artifacts/` (`reference.json` +
  `isolation_forest.joblib`), exactly what a Cloud Run scorer will load at
  startup in the next stage.
- [x] **Streaming service logic**, proven locally with no GCP dependency:
  a FastAPI push subscriber (`service/main.py`) implementing Pub/Sub's real
  push contract, scoring against the frozen reference, deduplicating
  redelivered messages, with 6 passing tests and a real local demo run
  (617.7 txn/s, mean 15.9ms/batch) against the actual fitted artifact. See
  [Streaming service](#streaming-service-local-no-gcp-dependency-yet).
- [x] **Real GCP deployment**: an actual Pub/Sub topic + OIDC-authenticated
  push subscription, a private Cloud Run service, and BigQuery, all created
  by `infra/setup.sh` and verified end to end with `publisher/replay.py`
  publishing to the real topic. Found a real, honestly-reported latency
  regression under concurrent BigQuery writes in the process; see
  [Live deployment](#live-deployment).
- [ ] **Looker Studio dashboard**: the three views it would sit on top of
  are live in BigQuery (`infra/dashboard_views.sql`, verified against real
  data), but connecting Looker Studio itself is a genuinely manual, GUI
  step (Google sign-in, then point-and-click chart building) that cannot
  be scripted or handed off the way every other stage of this project was;
  see [Looker Studio dashboard](#looker-studio-dashboard) for the exact
  remaining steps.
- [ ] Refresh `app.py` (currently written for the batch scaffold's earlier,
  different bank-transactions dataset schema, not yet updated for PaySim)
  and `notebooks/01_eda.ipynb` (same issue) for the current schema.

## Project structure

```
realtime-transaction-scoring/
  main.py                    One command batch pipeline entry point
  config.yaml                All thresholds, weights, rules, scope filter
  requirements.txt
  scripts/
    download_data.py         kagglehub-based PaySim download
    local_demo_replay.py     Real-HTTP demo against a live service instance
  service/
    main.py                  FastAPI Pub/Sub push subscriber
    scoring.py               Batch scoring orchestration for one push
    sink.py                  SQLite sink with idempotent writes (local dev)
    sink_bigquery.py          BigQuery sink (deployed), same interface
    schemas.py                Pub/Sub push envelope + transaction contracts
    Dockerfile                linux/amd64 build for Cloud Run
    requirements.txt          Leaner, container-only dependency set
  publisher/
    replay.py                 Real google-cloud-pubsub publisher
  infra/
    setup.sh                  Every gcloud/docker/bq command used to
                              deploy this for real, re-runnable
  src/
    load_data.py             Load, filter, engineer features, validate
    eda.py                   Summary stats, data quality, figures
    statistical_methods.py   Z-score, IQR, EWMA control chart (batch-only;
                              periodic control chart, see project status)
    reference.py              Fit-once/score-many z-score, IQR, Isolation
                              Forest reference (streaming-safe path)
    models.py                Isolation Forest fit/score split, agreement
    alerting.py               Combine signals and apply business rules
    prioritization.py        Severity scoring and incident queue
    evaluation.py             Precision/recall/PR-AUC/lift vs real labels
  app.py                     Streamlit dashboard (pending schema refresh)
  notebooks/01_eda.ipynb     Exploratory analysis (pending schema refresh)
  tests/                     Pytest unit tests (34 passing), including
                              tests/test_streaming_equivalence.py (batch vs
                              row-by-row scoring proof) and
                              tests/test_service.py (FastAPI push endpoint)
  reports/                   Generated outputs (git ignored CSVs, figures)
  artifacts/                 Frozen reference: reference.json +
                              isolation_forest.joblib (git ignored,
                              regenerated by `python main.py`)
```

## Reproduce

```bash
# 1. Clone
git clone https://github.com/JAYANSHUBADLANI/realtime-transaction-scoring.git
cd realtime-transaction-scoring

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download PaySim into data/ (see data/README.md for two more options)
pip install kagglehub
python scripts/download_data.py

# 5. Run the full pipeline (writes outputs to reports/ and prints a summary,
#    including the labeled evaluation above; takes about a minute)
python main.py

# Run the tests
pytest -q

# Try the local streaming service (needs artifacts/ from `python main.py`
# above; run each command in a separate terminal)
uvicorn service.main:app --port 8000
python scripts/local_demo_replay.py

# Deploy for real (needs gcloud authenticated against a billing-enabled
# project, and a local Docker daemon running)
./infra/setup.sh
python publisher/replay.py --project <your-project-id> --n 500
```

## Limitations and next steps

- No Looker Studio dashboard yet; see [Project status](#project-status).
- The BigQuery sink uses the classic streaming insert API, not the modern
  Storage Write API, and the [Live deployment](#live-deployment) evidence
  shows exactly what that costs under concurrent writes (mean 9.17s per
  push under a 50-request burst). Switching is the clearest concrete next
  step this project has, not a vague "could be faster."
- `app.py` and `notebooks/01_eda.ipynb` still reference the earlier
  unlabeled bank-transactions scaffold this project started from and have
  not been updated for the PaySim schema; `main.py` is the verified path.
- Thresholds (rule cut points, IQR's `k`, z-score's threshold) are
  configured, not learned. With more time, the rules and weights could be
  tuned against precision/recall on a held-out split rather than fit and
  reported on the same run, which is what this evaluation does today.
- The deployed Cloud Run service's IAM policy also grants the developer's
  own Google account `roles/run.invoker`, alongside the dedicated
  push-auth service account, so the deployment could be verified directly
  during development. Disclosed here rather than left as an unstated
  exception to the "Pub/Sub-only" security posture described in
  [Live deployment](#live-deployment).
- See [What did not work](#what-did-not-work) for the control chart and
  IQR limitations found during evaluation, both left as open next steps
  rather than papered over.

## License

MIT, see [LICENSE](LICENSE).
