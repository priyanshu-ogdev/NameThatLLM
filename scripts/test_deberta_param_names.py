from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

deb_cfg = DebertaV2Config(num_labels=18, hidden_size=128, num_hidden_layers=2, num_attention_heads=2, intermediate_size=256)
deb_model = DebertaV2ForSequenceClassification(deb_cfg)

print("All DeBERTa param names:")
for n, p in deb_model.named_parameters():
    print(" ", n)

