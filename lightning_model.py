import pytorch_lightning as pl
import torch

from prompt_re import PromptRE


class PromptREpl(pl.LightningModule):
    def __init__(
            self,
            rel_num,
            mask_length,
            rel_prompts=None,
            padding_id=0,
            span_bert="spanbert-base-cased/",
            mlm_w=1., sbo_w=1., l2_reg=1., rel_w=1.,
            loss_fn=None, metric=None, lr=None):
        super(PromptREpl, self).__init__()
        self.lr = lr
        self.mlm_w = mlm_w
        self.sbo_w = sbo_w
        self.l2_reg = l2_reg
        self.rel_w = rel_w
        self.save_hyperparameters(ignore=["loss_fn", "metric", "rel_prompts"])
        self.model = PromptRE(rel_num, mask_length, rel_prompts, padding_id, span_bert)
        self.metric = metric
        self.loss_fn = loss_fn

    def computing_step(self, batch, batch_idx, prefix=None):
        input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels = batch
        mlm_loss, sbo_loss, l2, pred = self.model(
            input_ids, att_mask=attention_mask,
            token_types=token_type_ids, sp_token_mask=sp_token_mask, prompt=relations)
        rel_loss = self.loss_fn(pred, labels)
        loss = mlm_loss * self.mlm_w + sbo_loss * self.sbo_w + \
               l2 * self.l2_reg + rel_loss * self.rel_w
        self.metric(pred, labels)
        self.log_dict({
            f"{prefix}_mlm_loss": mlm_loss,
            f"{prefix}_sbo_loss": sbo_loss,
            f"{prefix}_l2_reg": l2,
            f"{prefix}_rel_loss": rel_loss}, on_epoch=True)
        self.log_dict({
            f"{prefix}_loss": loss,
            f"{prefix}_f1": self.metric
        }, prog_bar=True)
        self.log("hp_metric", self.metric)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.computing_step(batch, batch_idx, "test")
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels = batch
        mlm_loss, sbo_loss, l2, pred = self.model(
            input_ids, att_mask=attention_mask,
            token_types=token_type_ids, sp_token_mask=sp_token_mask, prompt=relations)
        return torch.argmax(pred, dim=-1)

    def configure_optimizers(self):
        op = torch.optim.Adam(self.parameters(), lr=self.lr)
        return op
