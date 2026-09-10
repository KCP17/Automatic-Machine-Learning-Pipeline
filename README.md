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

## Layout

| Path | Purpose |
|---|---|
| `data/raw.csv` | Training data. Tracked by DVC, **not** stored in git. |
| `data/raw.csv.dvc` | The pointer git *does* track. |
| `training/train.py` | Fine-tunes the model and exports a run directory. |
| `training/push_model.py` | Uploads the most recent run to the Hub. |
| `.github/workflows/train.yml` | Runs the whole thing in CI. |
| `models/` | Training output. Git-ignored — models live on the Hub. |

## Running it locally

```bash
pip install -r requirements.txt

dvc pull                        # fetch data/raw.csv from the DVC remote
python training/train.py        # writes models/model_<timestamp>/

export HF_TOKEN=hf_...          # a write token
python training/push_model.py   # publishes the newest run
```

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
| `MAX_LENGTH` | `256` | Token truncation length. |
| `SEED` | `42` | Seed for the split and for training. |
| `LABEL_NAMES` | `negative,positive` | Comma-separated `id2label` names. |

`push_model.py` reads `HF_REPO_ID` (default `Kiernan1410/auto-ml-model`) and
requires `HF_TOKEN`.

## Required repository secrets

Set these under **Settings → Secrets and variables → Actions**.

- **`GDRIVE_CREDENTIALS_DATA`** — Google credentials that can *read* the Drive
  folder backing the DVC remote. Without it CI has no dataset, because
  `data/raw.csv` is intentionally not in git. The workflow accepts either a
  service-account key (recommended, does not expire) or a cached OAuth user
  token, and detects which one it was given.
- **`HF_TOKEN`** — a Hugging Face **write** token. Publishing is skipped rather
  than failed when this is absent.

The data must also exist in the remote: run `dvc push` locally at least once.

> Push from your own Google account, not the service account. Service accounts
> have no Drive storage quota of their own, so uploading into a personal My
> Drive folder fails with `storageQuotaExceeded`. Reading is unaffected, which
> is all CI does.

The workflow trains on every push and pull request, but only publishes from
`main`.

## Note on the sample dataset

`data/raw.csv` ships with four rows, which is enough to exercise the pipeline
but far too few to learn anything — expect a near-random model and an evaluation
set of a single example. Add real rows, then `dvc add data/raw.csv && dvc push`
and commit the updated `.dvc` pointer to trigger a retrain.
