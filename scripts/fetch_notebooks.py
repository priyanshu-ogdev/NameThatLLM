import urllib.request
import json
import re

urls = [
    ("lightningv08", "https://www.kaggle.com/code/lightningv08/0-661-qwen3-8-27b-lora-spectral-ensemble"),
    ("omerfarukmerey", "https://www.kaggle.com/code/omerfarukmerey/0-663-0-660-v31-atlas-20")
]

for name, url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        print(f"{name}: HTML size {len(html)} bytes")
        m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
        if m:
            data = json.loads(m.group(1))
            with open(f"{name}_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"{name}: saved {name}_data.json")
        else:
            print(f"{name}: __NEXT_DATA__ not found")
    except Exception as e:
        print(f"{name} error: {e}")

