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
| `data/raw.csv` | Training data. Tracked by DVC in a GCS bucket, **not** stored in git. |
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

- **`GCP_SA_KEY`** — the full JSON of a Google Cloud service-account key with
  read access to the bucket (`roles/storage.objectViewer`). Without it CI has
  no dataset, because `data/raw.csv` is intentionally not in git.
- **`HF_TOKEN`** — a Hugging Face **write** token. Publishing is skipped rather
  than failed when this is absent.

The data must also exist in the bucket: run `dvc push` locally at least once.

The workflow trains on every push and pull request, but only publishes from
`main`.

## Note on the sample dataset

`data/raw.csv` ships with four rows, which is enough to exercise the pipeline
but far too few to learn anything — expect a near-random model and an evaluation
set of a single example. Add real rows, then `dvc add data/raw.csv && dvc push`
and commit the updated `data/raw.csv.dvc` pointer to trigger a retrain.
