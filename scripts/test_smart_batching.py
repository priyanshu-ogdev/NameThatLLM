import numpy as np

# Simulate 2,159 real sequence lengths from the dataset
np.random.seed(42)
# Log-normal distribution mirroring real subword token lengths (mean ~230, min 40, max 352)
lengths = np.clip(np.random.lognormal(mean=5.4, sigma=0.35, size=2159).astype(int), 40, 352)

print(f"Total samples: {len(lengths)}")
print(f"Sample length min={lengths.min()}, median={np.median(lengths):.0f}, mean={np.mean(lengths):.1f}, max={lengths.max()}")

batch_size = 10

# Scenario A: Completely random permutation
perm_random = np.random.permutation(len(lengths))
random_max_lengths = []
for start in range(0, len(lengths), batch_size):
    batch = lengths[perm_random[start:start + batch_size]]
    random_max_lengths.append(batch.max())

avg_random_padded_len = np.mean(random_max_lengths)
random_attention_flops = np.sum([l**2 * len(lengths[perm_random[start:start + batch_size]]) for start, l in zip(range(0, len(lengths), batch_size), random_max_lengths)])

# Scenario B: Mega-batch bucketed sorting (mega_batch_size = 50 = 5 micro-batches)
mega_batch_size = 50
perm_mega = np.random.permutation(len(lengths))
smart_max_lengths = []
for m_start in range(0, len(lengths), mega_batch_size):
    m_idx = perm_mega[m_start:m_start + mega_batch_size]
    # Sort within mega-batch by length
    sorted_m_idx = m_idx[np.argsort(lengths[m_idx])]
    for start in range(0, len(sorted_m_idx), batch_size):
        batch = lengths[sorted_m_idx[start:start + batch_size]]
        smart_max_lengths.append(batch.max())

avg_smart_padded_len = np.mean(smart_max_lengths)
smart_attention_flops = np.sum([l**2 * len(lengths[sorted_m_idx[start:start + batch_size]]) for start, l in zip(range(0, len(sorted_m_idx), batch_size), smart_max_lengths)])

print("\n--- RESULTS ---")
print(f"Random Batching: Average Padded Length = {avg_random_padded_len:.1f} tokens")
print(f"Smart Mega-Batching: Average Padded Length = {avg_smart_padded_len:.1f} tokens")
print(f"Average Padded Length Reduction: {(1 - avg_smart_padded_len / avg_random_padded_len)*100:.1f}%")
print(f"Attention FLOPs (O(L^2)) Reduction: {(1 - smart_attention_flops / random_attention_flops)*100:.1f}%")
print(f"Speedup Factor on Attention: {random_attention_flops / smart_attention_flops:.2f}x")

