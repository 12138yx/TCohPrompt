from torch.utils.data import DataLoader
from lightning_model import PromptREpl
import torchmetrics
from torch import nn
import torch
from load_data import tacred, semeval
import pytorch_lightning  as pl


def collate_fn(inputs):
    batched = []
    for i in range(len(inputs[0])):
        data_i = []
        for j in range(len(inputs)):
            data_i.append(inputs[j][i])
        batched.append(torch.tensor(data_i, device=device))
    return batched


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}.", flush=True)

    saved_models_dir = "saved_models"
    data = tacred(("data/tacred/data/json/train.json",
                   "data/tacred/data/json/dev.json",
                   "data/tacred/data/json/test.json"))
    if len(data) == 3:
        train_data, dev_data, test_data = data
    else:
        train_data, test_data = data
        dev_data = test_data
    print(f"Training data size: {len(train_data)}")
    print(f"Dev data size: {len(dev_data)}")
    print(f"Test data size: {len(test_data)}")

    pretrained_path = {
        "base": "spanbert-base-cased/",
        "large": "spanbert-large-cased/"
    }

    epoch = 2
    batch_size = 6
    lr = 3e-5
    model_level = "base"
    mlm_w, sbo_w, l2_w, rel_w = 0.5, 0.5, 0.5, 1

    tra_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn,
                            shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=collate_fn)
    tes_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    rel_prompt2id = train_data.rel_prompt2ids
    rel_num = len(rel_prompt2id)
    rel_prompt_id = torch.tensor(list(rel_prompt2id.values()))
    mask_length = rel_prompt_id.shape[-1]

    loss = nn.CrossEntropyLoss(train_data.rel_w.to(device))
    f1 = torchmetrics.F1Score("multiclass", num_classes=rel_num, average="micro", ignore_index=0).to(device)

    model = PromptREpl(
        rel_num, mask_length, rel_prompt_id, span_bert=pretrained_path[model_level],
        mlm_w=mlm_w, sbo_w=sbo_w, l2_reg=l2_w, rel_w=rel_w,
        loss_fn=loss, metric=f1, lr=lr
    )

    trainer = pl.Trainer(
        enable_model_summary=False,
        reload_dataloaders_every_n_epochs=1,
        max_epochs=epoch,
    )
    trainer.fit(model, tra_loader, dev_loader, )

