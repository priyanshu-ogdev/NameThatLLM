import time

def test_deadline_logic():
    print("Testing deadline guard logic...")
    t0 = time.perf_counter()
    # Set artificial deadline 0.1s in the future
    deadline = t0 + 0.1

    epochs_run = 0
    best_score = -1.0
    best_epoch = -1

    for epoch in range(10):
        time.sleep(0.05)
        val_score = 0.5 + epoch * 0.05
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch + 1

        epochs_run += 1
        if deadline is not None and time.perf_counter() > deadline:
            print(f"Time guard triggered after epoch {epoch+1}/10")
            break

    assert epochs_run < 10, f"Expected early stop, but ran {epochs_run} epochs"
    assert best_score > 0, "Expected valid best_score"
    print(f"Deadline test passed! Stopped at epoch {epochs_run}/10 with best score {best_score:.2f} (best epoch {best_epoch})")

if __name__ == "__main__":
    test_deadline_logic()

