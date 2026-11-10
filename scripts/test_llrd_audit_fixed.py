import torch
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification, BertConfig, BertForSequenceClassification

def get_optimizer_grouped_parameters_fixed(model, lr_head, lr_backbone, layer_decay=0.85, weight_decay=0.01):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    base_model = getattr(model, "deberta", getattr(model, "bert", getattr(model, "model", model)))
    encoder = getattr(base_model, "encoder", None)
    layers = getattr(encoder, "layer", None)
    embeddings = getattr(base_model, "embeddings", None)
    
    assigned_param_ids = set()
    params_group = []
    
    head_modules = [getattr(model, "classifier", None), getattr(model, "pooler", None), getattr(base_model, "pooler", None)]
    head_params = []
    for mod in head_modules:
        if mod is not None:
            head_params.extend(list(mod.named_parameters()))
            
    if head_params:
        params_group.append({
            "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            "lr": lr_head, "weight_decay": weight_decay,
        })
        params_group.append({
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "lr": lr_head, "weight_decay": 0.0,
        })
        for n, p in head_params:
            assigned_param_ids.add(id(p))
            
    if layers is not None:
        n_layers = len(layers)
        for i in reversed(range(n_layers)):
            layer = layers[i]
            decay_factor = layer_decay ** (n_layers - 1 - i)
            layer_lr = lr_backbone * decay_factor
            l_params = list(layer.named_parameters())
            params_group.append({
                "params": [p for n, p in l_params if not any(nd in n for nd in no_decay)],
                "lr": layer_lr, "weight_decay": weight_decay,
            })
            params_group.append({
                "params": [p for n, p in l_params if any(nd in n for nd in no_decay)],
                "lr": layer_lr, "weight_decay": 0.0,
            })
            for n, p in l_params:
                assigned_param_ids.add(id(p))
                
    if embeddings is not None:
        n_layers = len(layers) if layers is not None else 24
        emb_lr = lr_backbone * (layer_decay ** n_layers)
        e_params = list(embeddings.named_parameters())
        params_group.append({
            "params": [p for n, p in e_params if not any(nd in n for nd in no_decay)],
            "lr": emb_lr, "weight_decay": weight_decay,
        })
        params_group.append({
            "params": [p for n, p in e_params if any(nd in n for nd in no_decay)],
            "lr": emb_lr, "weight_decay": 0.0,
        })
        for n, p in e_params:
            assigned_param_ids.add(id(p))
            
    leftover = [p for n, p in model.named_parameters() if id(p) not in assigned_param_ids]
    if leftover:
        params_group.append({
            "params": leftover,
            "lr": lr_backbone,
            "weight_decay": weight_decay,
        })
        
    return params_group

deb_cfg = DebertaV2Config(num_labels=18, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=256)
deb_model = DebertaV2ForSequenceClassification(deb_cfg)
groups = get_optimizer_grouped_parameters_fixed(deb_model, lr_head=1e-4, lr_backbone=2e-5)
all_params = list(deb_model.parameters())
grouped = [p for g in groups for p in g["params"]]
assert len(all_params) == len(grouped), f"Mismatch: {len(all_params)} != {len(grouped)}"
print("DeBERTa: All", len(all_params), "parameters successfully grouped!")

bge_cfg = BertConfig(num_labels=18, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=256)
bge_model = BertForSequenceClassification(bge_cfg)
groups_b = get_optimizer_grouped_parameters_fixed(bge_model, lr_head=1e-4, lr_backbone=2e-5)
all_b = list(bge_model.parameters())
grouped_b = [p for g in groups_b for p in g["params"]]
assert len(all_b) == len(grouped_b), f"Mismatch BGE: {len(all_b)} != {len(grouped_b)}"
print("BGE: All", len(all_b), "parameters successfully grouped!")

