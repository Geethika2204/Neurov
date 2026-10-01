import re
import requests
from pathlib import Path

def parse_pointer(path):
    text = Path(path).read_text()
    oid = re.search(r"oid sha256:(\w+)", text)
    size = int(re.search(r"size (\d+)", text))
    return oid, size

def fetch_lfs_object(repo, pointer_path, out_path):
    oid, size = parse_pointer(pointer_path)
    batch_url = f"https://github.com/{repo}.git/info/lfs/objects/batch"
    headers = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}
    payload = {"operation": "download", "transfers": ["basic"], "objects": [{"oid": oid, "size": size}]}
    r = requests.post(batch_url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    href = r.json()["objects"][0]["actions"]["download"]["href"]
    with requests.get(href, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)

if __name__ == "__main__":
    fetch_lfs_object(
        repo="acoursey3/freeway-anomaly-data",
        pointer_path="data/nashville_freeway_anomaly.txt",
        out_path="data/nashville_freeway_anomaly.csv",
    )