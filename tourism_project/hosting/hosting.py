# Hosting: push the deployment folder to a Hugging Face Streamlit (Docker) Space
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_USER = os.getenv("HF_USER", "abhi261204")
space_id = f"{HF_USER}/tourism-package-app"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=space_id, repo_type="space")
    print(f"Space '{space_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=space_id, repo_type="space", space_sdk="docker", private=False)
    print(f"Space '{space_id}' created.")

api.upload_folder(folder_path="tourism_project/deployment", repo_id=space_id, repo_type="space")
print("Deployment files pushed to the Hugging Face Space.")
