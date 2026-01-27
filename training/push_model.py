from huggingface_hub import HfApi
import os

MODELS_DIR = "models"
latest_model = sorted(os.listdir(MODELS_DIR))[-1]
model_path = os.path.join(MODELS_DIR, latest_model)

api = HfApi()
api.upload_folder(
    folder_path=model_path,
    repo_id="Kiernan1410/auto-ml-model",
    repo_type="model"
)