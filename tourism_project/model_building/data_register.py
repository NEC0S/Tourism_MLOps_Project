# Data registration: upload the raw dataset folder to a Hugging Face dataset repo
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_USER = os.getenv("HF_USER", "abhi261204")          # <- replace with your HF username
repo_id = f"{HF_USER}/tourism-package-prediction"           # dataset repo name on the Hub
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))                    # token comes from env / GitHub secret

# Create the dataset repo only if it does not exist yet
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload the local data folder (contains tourism.csv)
api.upload_folder(folder_path="tourism_project/data", repo_id=repo_id, repo_type=repo_type)
print("Raw dataset registered on Hugging Face Hub.")
