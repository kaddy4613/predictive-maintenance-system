from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

# Initialize the Hugging Face API
api = HfApi()

repo_id = "Kaddy4613/Predictive-Maintenance"
repo_type = "dataset"

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' does not exist. Creating a new space.")
    create_repo(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' created successfully.")

# Step 2 = upload dataset
api.upload_folder(
    folder_path="HardeepKadian_PredictiveMaintenance/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded successfully.")
