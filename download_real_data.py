import requests
import os
import json

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DDXPLUS_BASE_URL = "https://huggingface.co/datasets/aai530-group6/ddxplus/resolve/main"
CMS_SAMPLE_URL = "https://raw.githubusercontent.com/CMSgov/price-transparency-guide/master/examples/in-network-rates/in-network-rates-fee-for-service-single-plan-sample.json"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {path}")

def download_ddxplus_sample():
    # Download metadata
    download_file(f"{DDXPLUS_BASE_URL}/release_conditions.json?download=true", "release_conditions.json")
    download_file(f"{DDXPLUS_BASE_URL}/release_evidences.json?download=true", "release_evidences.json")
    
    # Stream first 1000 lines of train.csv to create a sample
    print("Downloading sample of train.csv...")
    response = requests.get(f"{DDXPLUS_BASE_URL}/train.csv?download=true", stream=True)
    response.raise_for_status()
    
    path = os.path.join(DATA_DIR, "train_sample.csv")
    with open(path, 'wb') as f:
        count = 0
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)
            count += 1
            if count > 500: # Approx 500KB
                break
    print(f"Saved sample to {path}")

def download_cms_sample():
    download_file(CMS_SAMPLE_URL, "cms_sample.json")

if __name__ == "__main__":
    try:
        download_ddxplus_sample()
        download_cms_sample()
        print("All downloads complete.")
    except Exception as e:
        print(f"Error downloading data: {e}")
