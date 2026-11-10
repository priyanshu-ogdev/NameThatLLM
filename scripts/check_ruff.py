import subprocess
import json

proc = subprocess.run(
    ["ruff", "check", "--select", "F821", "--output-format", "json", "notebooks/kaggle_tri_ensemble.py"],
    capture_output=True,
    text=True
)
items = json.loads(proc.stdout)
for item in items:
    print(f"{item['location']['row']}:{item['location']['column']} - {item['message']}")

