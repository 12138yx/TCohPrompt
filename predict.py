import torch

from load_data import TacredParse
from torch.utils.data import DataLoader
from prompt_re import PromptRE
from train import collate_fn


def prediction(m, data):
    rels = []
    model.eval()
    with torch.no_grad():
        for batch, (input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels) in enumerate(data):
            # Compute prediction error
            mlm_loss, sbo_loss, pred = m(
                input_ids, att_mask=attention_mask,
                token_types=token_type_ids, sp_token_mask=sp_token_mask,
                prompt=relations)
            pred_rel = torch.argmax(pred, -1)
            rels.append(pred_rel)

        return torch.concat(rels)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Predicting on {device}.")

saved_models_dir = "saved_models"
test_data = TacredParse("data/tacred/data/json/test.json", max_length=336)

print(f"Test data size: {len(test_data)}", flush=True)

tes_loader = DataLoader(test_data, batch_size=8, collate_fn=collate_fn)

rel_prompt2id = test_data.rel_prompt2ids
rel_num = len(rel_prompt2id)
rel_prompt_id = torch.tensor(list(rel_prompt2id.values()))
mask_length = rel_prompt_id.shape[-1]
model = PromptRE(rel_num, mask_length)
model.load_state_dict(torch.load("saved_models/model_03.pt"))
model.to(device)

rel_ids = prediction(model, tes_loader)
id2rel = dict([(v, k) for k, v in test_data.rel_id.items()])

with open("result.txt", "w") as f:
    for i in rel_ids:
        print(id2rel[i.item()], file=f)
