import torch.optim

from load_data import tacred, semeval
from torch.utils.data import DataLoader, ConcatDataset
from prompt_re import PromptRE
from torch import nn
import torchmetrics
from torchmetrics import Metric
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}.", flush=True)


def train_fn(data: DataLoader, model: PromptRE,
             loss_fn: nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, metric: Metric,
             mlm_w=1., sbo_w=1., trans_rel=1., rel_w=1.):
    loss_, mlm_, sbo_, l2_, rel_ = [0] * 5
    model.train()
    for batch, (input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels) in enumerate(data):
        # Compute prediction error
        mlm_loss, sbo_loss, l2, pred = model(
            input_ids, att_mask=attention_mask,
            token_types=token_type_ids, sp_token_mask=sp_token_mask, prompt=relations)
        rel_loss = loss_fn(pred, labels)
        loss = mlm_loss * mlm_w + sbo_loss * sbo_w + l2 * trans_rel + rel_loss * rel_w

        mlm_ += (mlm_loss * mlm_w).item()
        sbo_ += (sbo_loss * sbo_w).item()
        l2_ += (l2 * trans_rel).item()
        rel_ += (rel_loss * rel_w).item()
        loss_ += loss.item()

        metric(pred, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    f1 = metric.compute().item()
    print(time.asctime())
    print(f"mlm_loss: {mlm_ / (batch + 1):>5f}, sbo_loss: {sbo_ / (batch + 1):>5f}, "
          f"rel_loss: {rel_ / (batch + 1):>5f}, rel_l2:{l2_ / (batch + 1):>5f}, "
          f"loss: {loss_ / (batch + 1):>5f}, "
          f"f1_score: {f1:>5f}", flush=True)
    return loss_ / (batch + 1)


def test_fn(data: DataLoader, model: nn.Module,
            loss_fn: nn.CrossEntropyLoss, metric: Metric,
            mlm_w=1., sbo_w=1., trans_rel=1., rel_w=1.):
    loss_, mlm_, sbo_, l2_, rel_ = [0] * 5
    model.eval()
    with torch.no_grad():
        for batch, (input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels) in enumerate(data):
            # Compute prediction error
            mlm_loss, sbo_loss, l2, pred = model(
                input_ids, att_mask=attention_mask,
                token_types=token_type_ids, sp_token_mask=sp_token_mask,
                prompt=relations)
            rel_loss = loss_fn(pred, labels)
            loss = mlm_loss * mlm_w + sbo_loss * sbo_w + l2 * trans_rel + rel_loss * rel_w

            mlm_ += (mlm_loss * mlm_w).item()
            sbo_ += (sbo_loss * sbo_w).item()
            l2_ += (l2 * trans_rel).item()
            rel_ += (rel_loss * rel_w).item()
            loss_ += loss.item()

            metric(pred, labels)

    f1 = metric.compute().item()
    print(time.asctime())
    print(f"mlm_loss: {mlm_ / (batch + 1):>5f}, sbo_loss: {sbo_ / (batch + 1):>5f}, "
          f"rel_loss: {rel_ / (batch + 1):>5f}, rel_l2:{l2_ / (batch + 1):>5f}, "
          f"loss: {loss_ / (batch + 1):>5f}, "
          f"f1_score: {f1:>5f}", flush=True)
    return loss_ / (batch + 1)


def collate_fn(inputs):
    batched = []
    for i in range(len(inputs[0])):
        data_i = []
        for j in range(len(inputs)):
            data_i.append(inputs[j][i])
        batched.append(torch.tensor(data_i, device=device))
    return batched


def train_loop():
    saved_models_dir = "saved_models"
    data = tacred(("data/tacred/data/json/train.json",
                   "data/tacred/data/json/dev.json",
                   "data/tacred/data/json/test.json"))
    # data = tacred(("data/re-tacred/data/train.json",
    #                "data/re-tacred/data/dev.json",
    #                "data/re-tacred/data/test.json"),
    #               rel_map="data/re-tacred/rel_map.csv")
    # data = semeval(("data/SemEval2010/train.txt",
    #                "data/SemEval2010/test.txt"))
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

    epoch = 20
    batch_size = 56
    lr = 3e-5
    model_level = "large"
    mlm_w, sbo_w, l2_w, rel_w = 0.5, 0.5, 1.0, 1.0

    # dev_loader = DataLoader(dev_data, batch_size=batch_size, collate_fn=collate_fn)
    tes_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)

    rel_prompt2id = train_data.rel_prompt2ids
    rel_num = len(rel_prompt2id)
    rel_prompt_id = torch.tensor(list(rel_prompt2id.values()))
    mask_length = rel_prompt_id.shape[-1]
    model = PromptRE(rel_num, mask_length, rel_prompt_id, span_bert=pretrained_path[model_level])
    model.to(device)

    loss = nn.CrossEntropyLoss(train_data.rel_w.to(device))
    adam = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    f1 = torchmetrics.F1Score("multiclass", num_classes=rel_num, average="micro", ignore_index=0).to(device)

    t_ = time.time()
    print(f"Training started at {time.asctime()}", flush=True)
    os.makedirs(saved_models_dir, exist_ok=True)
    for i in range(epoch):
        # shuffle training data every epoch
        tra_loader = DataLoader(ConcatDataset((train_data, dev_data)),
                                batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)

        # tra_loader = DataLoader(train_data,
        #                         batch_size=batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
        f1.reset()
        print(f"Epoch {i + 1:>2d} train loss:")
        _ = train_fn(tra_loader, model, loss, adam, f1,
                     mlm_w, sbo_w, l2_w, rel_w)
        torch.save(model.state_dict(), os.path.join(saved_models_dir, f"tacred_{model_level}_{str(i + 1).zfill(2)}.pt"))
        # f1.reset()
        # print(f"dev loss:")
        # _ = test_fn(dev_loader, model, loss, f1)
        f1.reset()
        print(f"test loss:")
        _ = test_fn(tes_loader, model, loss, f1)

    print(f"Training ended at {i + 1:>2d}-th epoch, costed {(time.time() - t_) / 3600:>3f} hours.")


if __name__ == "__main__":
    train_loop()
