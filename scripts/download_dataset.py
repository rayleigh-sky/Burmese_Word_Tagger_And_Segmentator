import urllib
import requests
from pathlib import Path

ROOT = Path.cwd().parent
DATA_DIR = Path("dataset")
DATA_DIR.mkdir(exist_ok=True)

BASE = "https://raw.githubusercontent.com/ye-kyaw-thu/myPOS/refs/heads/master/corpus-ver-3.0/corpus"

for name in ("mypos-ver.3.0.shuf.nopipe.txt", "otest.1k.nopipe.txt"):
    target = DATA_DIR / name
    if not target.exists():
        urllib.request.urlretrieve(f"{BASE}/{name}", target)
    print(f"{name:<34} {target.stat().st_size:>10,} bytes")