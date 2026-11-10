import subprocess
import json
from collections import Counter

proc = subprocess.run(
    ["ruff", "check", "--output-format", "json", "notebooks/kaggle_tri_ensemble.py"],
    capture_output=True,
    text=True
)
items = json.loads(proc.stdout)
counts = Counter(item["code"] for item in items)
print("Error code counts:")
for code, count in counts.most_common():
    print(f"  {code}: {count}")

print("\nNon-F821 errors:")
for item in items:
    if item["code"] != "F821":
        print(f"  {item['code']} at {item['location']['row']}:{item['location']['column']} - {item['message']}")

