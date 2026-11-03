
import os
import sys
import time
import argparse
import logging

from huggingface_hub import snapshot_download

# ==========================================================
# CONFIGURATION
# ==========================================================

REPO_ID = "Huggingfacerepoid"

DOWNLOAD_DIR = os.path.expanduser(
    "path where to download"
)

WORKERS = 16

# Set HF_TOKEN in your environment for private repositories.
HF_TOKEN = "hugging face token"

# Download everything by default.
INCLUDE = None

# Examples:
# INCLUDE = ["*.jpg", "*.jpeg", "*.png"]
# INCLUDE = ["hi/*.jpg", "hi/*.png"]

EXCLUDE = [
    ".DS_Store",
    "Thumbs.db",
    "._*"
]

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)

lg = logging.getLogger("HF_DOWNLOAD")


# ==========================================================
# DOWNLOAD
# ==========================================================

def download_dataset(repo_id, output, workers):

    os.makedirs(output, exist_ok=True)

    lg.info("=" * 60)
    lg.info("HUGGING FACE BULK DOWNLOADER")
    lg.info("=" * 60)

    lg.info("Repository : %s", repo_id)
    lg.info("Destination: %s", output)
    lg.info("Workers    : %s", workers)

    start = time.time()

    try:

        downloaded_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output,
            token=HF_TOKEN,
            max_workers=workers,
            allow_patterns=INCLUDE,
            ignore_patterns=EXCLUDE,
        )

        elapsed = time.time() - start

        lg.info("=" * 60)
        lg.info("DOWNLOAD COMPLETED")
        lg.info("=" * 60)

        lg.info("Location: %s", downloaded_path)
        lg.info("Time: %.2f minutes", elapsed / 60)

    except Exception as e:

        lg.exception("DOWNLOAD FAILED: %s", e)
        sys.exit(1)


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Hugging Face Dataset Bulk Downloader"
    )

    parser.add_argument(
        "--repo-id",
        default=REPO_ID
    )

    parser.add_argument(
        "--output",
        default=DOWNLOAD_DIR
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS
    )

    args = parser.parse_args()

    download_dataset(
        repo_id=args.repo_id,
        output=args.output,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
