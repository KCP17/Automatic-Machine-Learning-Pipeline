# Automatic Machine Learning Pipeline

A minimal end-to-end MLOps loop: the dataset is versioned with **DVC**, a
**DistilBERT** text classifier is fine-tuned on it, and every successful run is
published to the **Hugging Face Hub** — automatically, on every push that
touches the data or the training code.

```
data/raw.csv (DVC)  ──►  training/train.py  ──►  models/model_<ts>/  ──►  HF Hub
        ▲                                                                    ▲
        └────────────── .github/workflows/train.yml orchestrates ────────────┘
```

Live model: **[Kiernan1410/auto-ml-model](https://huggingface.co/Kiernan1410/auto-ml-model)**
— currently ~0.85 accuracy / F1 on a held-out split of the dataset below.

## Layout

| Path | Purpose |
|---|---|
| `data/raw.csv` | Training data. Tracked by DVC in a GCS bucket, **not** stored in git. |
| `data/raw.csv.dvc` | The pointer git *does* track. |
| `training/prepare_data.py` | Regenerates `data/raw.csv` from the source dataset (manual, not CI). |
| `training/train.py` | Fine-tunes the model and exports a run directory. |
| `training/push_model.py` | Uploads the most recent run to the Hub. |
| `.github/workflows/train.yml` | Runs the whole thing in CI. |
| `models/` | Training output. Git-ignored — models live on the Hub. |

## Running it locally

```bash
pip install -r requirements.txt

dvc pull                        # fetch data/raw.csv (needs remote auth — see below)
python training/train.py        # writes models/model_<timestamp>/

export HF_TOKEN=hf_...          # a write token
python training/push_model.py   # publishes the newest run
```

A fresh clone can't `dvc pull` until the [data remote](#data-remote-google-cloud-storage)
is authenticated.

`train.py` writes a self-contained directory containing the weights, the
tokenizer, a `metrics.json`, and a generated model card. `push_model.py` reads
`models/latest.txt` to know which run to publish, so it never depends on
directory ordering.

### Configuration

`train.py` reads its settings from the environment, so CI can tune a run
without a code change:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_NAME` | `distilbert-base-uncased` | Base checkpoint to fine-tune. |
| `NUM_EPOCHS` | `2` | Training epochs. |
| `BATCH_SIZE` | `8` | Per-device batch size. |
| `LEARNING_RATE` | `5e-5` | Optimizer learning rate. |
| `TEST_SIZE` | `0.2` | Fraction held out for evaluation. |
| `MAX_LENGTH` | `256` | Token truncation length. CI sets `128` (the reviews are short). |
| `SEED` | `42` | Seed for the split and for training. |
| `LABEL_NAMES` | `negative,positive` | Comma-separated `id2label` names. |

`push_model.py` reads `HF_REPO_ID` (default `Kiernan1410/auto-ml-model`) and
requires `HF_TOKEN`.

## Data remote (Google Cloud Storage)

The dataset lives in a GCS bucket, tracked by DVC. `.dvc/config` holds the
bucket URL; credentials stay out of git.

**Local setup** (one time):

```bash
gcloud auth application-default login   # sign in with your own Google account
dvc pull                                # or `dvc push` after changing the data
```

`gcsfs` picks those credentials up automatically — no DVC credential config is
needed locally. To use a service-account key instead:
`dvc remote modify --local storage credentialpath /path/to/key.json`
(`--local` writes the git-ignored `.dvc/config.local`).

## Required repository secrets

Set these under **Settings → Secrets and variables → Actions**.

- **`GCP_SA_KEY`** — the full JSON of a Google Cloud service-account key for the
  bucket. CI only reads, so `roles/storage.objectViewer` is the minimum; this
  project's key has `roles/storage.objectAdmin` so the same credential can also
  `dvc push`. Without it CI has no dataset, because `data/raw.csv` is not in git.
- **`HF_TOKEN`** — a Hugging Face **write** token. Publishing is skipped rather
  than failed when this is absent.

The data must also exist in the bucket: run `dvc push` locally at least once.

The workflow trains on every push and pull request, but only publishes from
`main`.

## The dataset

`data/raw.csv` is a 4,000-row stratified sample (2,000 / 2,000) of the
**Rotten Tomatoes** movie-review sentiment corpus — short single-sentence
reviews, `label` 0 = negative, 1 = positive. The size keeps a CPU CI run to
~10 minutes while still producing meaningful accuracy/F1.

To change it, edit `training/prepare_data.py` (or its `SOURCE_DATASET` /
`SAMPLE_SIZE` env vars), then:

```bash
python training/prepare_data.py
dvc add data/raw.csv && dvc push
git add data/raw.csv.dvc && git commit -m "Refresh dataset" && git push
```

The final push retrains and republishes. `prepare_data.py` is a manual tool —
CI only ever consumes the committed `data/raw.csv.dvc` pointer.

## Proof: a data change alone retrains and republishes the model

The workflow reacts to Git, and the dataset's DVC pointer is a Git file:

```yaml
# .github/workflows/train.yml
on:
  push:
    paths:
      - data/**          # data/raw.csv.dvc lives here
      - training/**
      - requirements.txt
      - .github/workflows/**
```

`data/raw.csv` is git-ignored, but `data/raw.csv.dvc` — which DVC rewrites
whenever the data changes — sits under `data/`, so committing it fires a run.

### Worked example — commit [`5586213`](https://github.com/KCP17/Automatic-Machine-Learning-Pipeline/commit/5586213)

That commit changed **one file, no code** — it grew the sample from 3,000 to
4,000 rows:

```
 data/raw.csv.dvc | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)
```

What happened, with no manual step:

| Time (UTC) | Event |
|---|---|
| 13:49:11 | `git push` — commit `5586213`, only `data/raw.csv.dvc` |
| 13:49:21 | workflow run [#15](https://github.com/KCP17/Automatic-Machine-Learning-Pipeline/actions/runs/34485049591) starts automatically (`event: push`) |
| 13:59 | pulled the new data from GCS, trained (~10 min), evaluated |
| 14:01:17 | [new model `model_20260910_135054`](https://huggingface.co/Kiernan1410/auto-ml-model/commits/main) pushed to the Hub |

### The metrics move with the data

The [Hub commit history](https://huggingface.co/Kiernan1410/auto-ml-model/commits/main)
is one automated commit per run. Accuracy tracks the dataset — nothing else changed:

| Dataset | Triggering commit | Published model | Held-out accuracy |
|---|---|---|---|
| 4 rows (placeholder) | — | `model_20260910_125754` | 0.00 |
| 3,000 rows | [`361a9f9`](https://github.com/KCP17/Automatic-Machine-Learning-Pipeline/commit/361a9f9) | `model_20260910_131807` | 0.83 |
| 4,000 rows | [`5586213`](https://github.com/KCP17/Automatic-Machine-Learning-Pipeline/commit/5586213) | `model_20260910_135054` | 0.85 |

### Try it

```bash
SAMPLE_SIZE=2000 python training/prepare_data.py   # any change to the data
dvc add data/raw.csv && dvc push
git commit -am "Shrink the sample" && git push
# watch: github.com/KCP17/Automatic-Machine-Learning-Pipeline/actions
```
