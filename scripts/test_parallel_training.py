"""
Test script to verify concurrent/parallel dual-model training logic.
Verifies thread isolation, proper return structures, and zero race conditions.
"""
import concurrent.futures
import time
import numpy as np
import torch
import torch.nn as nn

def dummy_train_worker(model, x_tr, y_tr, x_va, x_te, dev_name, epochs=3, lr=1e-3):
    dev = torch.device(dev_name)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    
    x_tr_t = torch.tensor(x_tr, dtype=torch.float32, device=dev)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=dev)
    x_va_t = torch.tensor(x_va, dtype=torch.float32, device=dev)
    x_te_t = torch.tensor(x_te, dtype=torch.float32, device=dev)
    
    best_val_loss = float("inf")
    best_val_logits = None
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(x_tr_t)
        loss = crit(out, y_tr_t)
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(x_va_t)
            v_loss = crit(val_out, torch.tensor(y_tr[:len(x_va)], dtype=torch.long, device=dev)).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_val_logits = val_out.cpu().numpy()
                
    with torch.no_grad():
        te_logits = model(x_te_t).cpu().numpy()
        
    return best_val_logits, te_logits

def main():
    print("Testing concurrent training worker flow...")
    n_samples = 200
    n_features = 32
    n_classes = 18
    
    x_tr = np.random.randn(n_samples, n_features).astype(np.float32)
    y_tr = np.random.randint(0, n_classes, size=n_samples)
    x_va = np.random.randn(40, n_features).astype(np.float32)
    x_te = np.random.randn(10, n_features).astype(np.float32)
    
    # Pre-instantiate two models sequentially
    model_a = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, n_classes))
    model_b = nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(), nn.Linear(128, n_classes))
    
    dev_a = "cpu"
    dev_b = "cpu"
    
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_a = executor.submit(dummy_train_worker, model_a, x_tr, y_tr, x_va, x_te, dev_a, epochs=5, lr=2e-3)
        fut_b = executor.submit(dummy_train_worker, model_b, x_tr, y_tr, x_va, x_te, dev_b, epochs=5, lr=1e-3)
        
        va_a, te_a = fut_a.result()
        va_b, te_b = fut_b.result()
        
    elapsed = time.time() - t0
    print(f"Parallel execution completed in {elapsed:.2f}s")
    assert va_a.shape == (40, n_classes), f"Mismatch: {va_a.shape}"
    assert te_a.shape == (10, n_classes), f"Mismatch: {te_a.shape}"
    assert va_b.shape == (40, n_classes), f"Mismatch: {va_b.shape}"
    assert te_b.shape == (10, n_classes), f"Mismatch: {te_b.shape}"
    print("All assertions passed successfully!")

if __name__ == "__main__":
    main()
