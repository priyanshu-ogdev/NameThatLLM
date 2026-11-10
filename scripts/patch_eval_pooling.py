from pathlib import Path

def patch_eval_pooling():
    path = Path("notebooks/kaggle_tri_ensemble.py")
    code = path.read_text(encoding="utf-8")

    old_target = """            mask = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device)
            sum_hidden = (out.last_hidden_state * mask).sum(dim=1).float()
            pooled = sum_hidden / torch.clamp(mask.sum(dim=1).float(), min=1e-9)"""

    new_target = """            m = batch_kwargs["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.device).float()
            pooled = (out.last_hidden_state.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)"""

    assert old_target in code, "Target not found"
    code = code.replace(old_target, new_target, 1)
    path.write_text(code, encoding="utf-8")
    print(f"Patched {path}")

if __name__ == "__main__":
    patch_eval_pooling()

