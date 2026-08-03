"""
download_data.py — Fetches ApacheJIT dataset CSVs from Zenodo

WHY THIS APPROACH:
We use the Zenodo REST API (https://zenodo.org/api/records/5907002/files) to get
the authoritative list of files rather than hardcoding URLs. This is important
because file paths can change between dataset versions, and we want the script
to always fetch the current, correct files.

The ApacheJIT dataset contains:
- apachejit_train.csv: Balanced training set (2003-2016)
- apachejit_total.csv: Full dataset (~106k commits, imbalanced ~26% buggy)
- apachejit_test_large.csv / apachejit_test_small.csv: Test sets (last 3 years)

USAGE:
    python scripts/download_data.py

The files will be saved to the data/ directory.
"""

import os
import sys
import requests
from pathlib import Path


# Zenodo record ID for ApacheJIT dataset
# Source: https://doi.org/10.5281/zenodo.5907002
ZENODO_RECORD_ID = "5907002"

# API endpoint to get file metadata for a Zenodo record
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


def get_project_root():
    """
    Returns the project root directory.
    This allows the script to work whether run from project root or scripts/ folder.
    """
    # This script is in scripts/, so go up one level
    script_dir = Path(__file__).parent.resolve()
    return script_dir.parent


def fetch_file_list():
    """
    Query the Zenodo API to get the list of files in the ApacheJIT dataset.
    
    WHY:
    Instead of hardcoding URLs (which can break if Zenodo changes paths),
    we ask Zenodo directly: "What files are in this record?" The API returns
    a JSON object with file names and their download links.
    
    Returns:
        list of dicts, each with 'filename' and 'download_url' keys
    """
    print(f"Querying Zenodo API for record {ZENODO_RECORD_ID}...")
    
    response = requests.get(ZENODO_API_URL)
    
    if response.status_code != 200:
        print(f"ERROR: Failed to fetch record metadata. Status code: {response.status_code}")
        sys.exit(1)
    
    record_data = response.json()
    
    # The 'files' key contains an array of file objects
    files = record_data.get("files", [])
    
    if not files:
        print("ERROR: No files found in the Zenodo record.")
        sys.exit(1)
    
    # Extract just the info we need: filename and download link
    file_info = []
    for f in files:
        file_info.append({
            "filename": f["key"],  # 'key' is the filename in Zenodo API
            "download_url": f["links"]["self"],  # direct download URL
            "size_mb": f["size"] / (1024 * 1024)  # convert bytes to MB for display
        })
    
    return file_info


def download_file(url, destination_path):
    """
    Download a file from a URL to the specified path.
    
    Uses streaming to handle large files without loading them entirely into memory.
    Shows a simple progress indicator.
    """
    print(f"  Downloading to {destination_path}...")
    
    # stream=True means we don't load the whole file into memory at once
    # This is important for large CSVs (the full dataset is ~50MB)
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"  ERROR: Failed to download. Status code: {response.status_code}")
        return False
    
    # Get total file size for progress display
    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0
    
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            # Simple progress indicator
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\r  Progress: {percent:.1f}%", end="", flush=True)
    
    print()  # newline after progress
    return True


def main():
    """
    Main function: fetch file list from Zenodo API, then download each CSV.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"
    
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ApacheJIT Dataset Downloader")
    print("=" * 60)
    print(f"Source: https://doi.org/10.5281/zenodo.{ZENODO_RECORD_ID}")
    print(f"License: CC-BY 4.0")
    print(f"Saving to: {data_dir}")
    print("=" * 60)
    print()
    
    # Step 1: Get the file list from Zenodo API
    files = fetch_file_list()
    
    print(f"\nFound {len(files)} files in the dataset:")
    for f in files:
        print(f"  - {f['filename']} ({f['size_mb']:.2f} MB)")
    print()
    
    # Step 2: Download each file
    # We only download CSV files (the dataset also has a README)
    csv_files = [f for f in files if f["filename"].endswith(".csv")]
    
    print(f"Downloading {len(csv_files)} CSV files...\n")
    
    for f in csv_files:
        dest_path = data_dir / f["filename"]
        
        # Skip if file already exists (don't re-download)
        if dest_path.exists():
            print(f"  {f['filename']} already exists, skipping.")
            continue
        
        print(f"Downloading {f['filename']} ({f['size_mb']:.2f} MB)...")
        success = download_file(f["download_url"], dest_path)
        
        if success:
            print(f"  Done!")
        else:
            print(f"  Failed to download {f['filename']}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    
    # Show what we downloaded
    print("\nFiles in data/ directory:")
    for f in sorted(data_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
