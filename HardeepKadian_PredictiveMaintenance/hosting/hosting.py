from huggingface_hub import HfApi
import os

# Initialize API
api = HfApi(
    token=os.getenv("HF_TOKEN")
)

# Upload deployment folder
api.upload_folder(
    folder_path="HardeepKadian_PredictiveMaintenance/deployment",
    repo_id="Kaddy4613/predictive-maintenance-system",
    repo_type="space",
    path_in_repo=""
)

print("Deployment files uploaded successfully.")
