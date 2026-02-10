from huggingface_hub import snapshot_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
TARGET_SUBDIR = "data/CC-MAIN-2025-26/**"
LOCAL_DIR = "/media/ayush/Rajtilak/fine_web_data/data"

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    allow_patterns=TARGET_SUBDIR,
    # local_dir=LOCAL_DIR,
    # local_dir_use_symlinks=False,  # IMPORTANT for mounted drives
)
