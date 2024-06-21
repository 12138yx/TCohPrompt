import torch

from load_data import TacredParse, tacred
from torch.utils.data import DataLoader
from prompt_re import PromptRE
from train import collate_fn
import torchmetrics


def eval(m, data, metric: torchmetrics.Metric):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch, (input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels) in enumerate(data):
            # Compute prediction error
            mlm_loss, sbo_loss, l2, pred = m(
                input_ids, att_mask=attention_mask,
                token_types=token_type_ids, sp_token_mask=sp_token_mask,
                prompt=relations)
            metric(pred, labels)
        v = metric.compute().item()
        print(f"f1_score: {v:>5f}", flush=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

saved_models_dir = "saved_models"
# _, dev_data, test_data = tacred(
#     ("data/re-tacred/data/train.json",
#      "data/re-tacred/data/dev.json",
#      "data/re-tacred/data/test.json"),
#     rel_map="data/re-tacred/rel_map.csv")

_, dev_data, test_data = tacred(
    ("data/re-tacred/data/train.json",
     "data/tacred-revised/data/dev.json",
     "data/tacred-revised/data/test.json"))

print(f"Dev data size: {len(dev_data)}", flush=True)
print(f"Test data size: {len(test_data)}", flush=True)

dev_loader = DataLoader(dev_data, batch_size=32, collate_fn=collate_fn)
tes_loader = DataLoader(test_data, batch_size=32, collate_fn=collate_fn)

rel_prompt2id = test_data.rel_prompt2ids
rel_num = len(rel_prompt2id)
rel_prompt_id = torch.tensor(list(rel_prompt2id.values()))
mask_length = rel_prompt_id.shape[-1]
model = PromptRE(rel_num, mask_length, span_bert="spanbert-large-cased/")
model.load_state_dict(torch.load("saved_models/tacred_large_05_73440.pt"))
model.to(device)

f1 = torchmetrics.F1Score(
    "multiclass", num_classes=rel_num, average="micro", ignore_index=0).to(device)
print("dev:")
eval(model, dev_loader, f1)
print("test:")
eval(model, tes_loader, f1)
