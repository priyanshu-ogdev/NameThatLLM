import urllib.request
import json
import re

url = "https://www.kaggle.com/code/alexchilton/name-that-llm-0-654-full-ensemble-what-failed"

req = urllib.request.Request(
    url,
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
)

try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    print(f"Downloaded HTML: {len(html)} bytes")
    
    # Check for embedded JSON blobs or metadata
    for m in re.finditer(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL):
        data = json.loads(m.group(1))
        print("Found __NEXT_DATA__ keys:", list(data.keys()))
        with open("next_data.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("Saved next_data.json")

    # Search for notebook content, markdown cells, code cells
    for m in re.finditer(r'\{"cell_type":', html):
        print("Found cell_type in HTML!")
        break
except Exception as e:
    print("Error:", e)

