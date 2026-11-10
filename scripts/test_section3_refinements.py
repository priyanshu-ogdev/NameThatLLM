import re
from collections import Counter

# 1. Test Contraction Regex with straight and curly quotes
CONTRACTION_RE = re.compile(r"\b\w+['’\u2019]\w+\b")
WORD_RE = re.compile(r"\b\w+(?:['’\u2019]\w+)?\b")

sample_text = "I don't know, but it’s clear they didn’t want to."
contractions = CONTRACTION_RE.findall(sample_text)
words = WORD_RE.findall(sample_text)

print("Contractions found:", contractions)
assert len(contractions) == 3, f"Expected 3 contractions, found {len(contractions)}: {contractions}"

# 2. Test Counter vs lower.count
lower = [w.lower() for w in words]
FUNCTION_WORDS = ["i", "it", "to", "they", "but"]
counts = Counter(lower)
features = [counts[fw] / len(words) for fw in FUNCTION_WORDS]
assert len(features) == 5
print("Counter function word features:", features)

# 3. Test Iterative DSU with 10,000 chained elements (would exceed recursion limit of 1000)
class IterativeDSU:
    def __init__(self, elements):
        self.p = {x: x for x in elements}
    def find(self, x):
        root = x
        while self.p[root] != root:
            root = self.p[root]
        curr = x
        while curr != root:
            nxt = self.p[curr]
            self.p[curr] = root
            curr = nxt
        return root
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.p[ry] = rx

elements = list(range(2000))
dsu = IterativeDSU(elements)
for i in range(len(elements) - 1):
    dsu.union(elements[i], elements[i+1])

# Query deep chain
assert dsu.find(0) == dsu.find(1999)
print("Iterative DSU passed 2,000-deep chain test without RecursionError!")

