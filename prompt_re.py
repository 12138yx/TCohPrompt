import torch
from torch import nn
from transformers import AutoModel


class PromptRE(nn.Module):
    def __init__(self, rel_num, mask_length, rel_prompts=None, padding_id=0, span_bert="spanbert-base-cased/"):
        super(PromptRE, self).__init__()
        self.rel_num = rel_num
        self.mask_length = mask_length
        self.rel_prompts = rel_prompts
        self.padding_id = padding_id

        self.span_bert = AutoModel.from_pretrained(span_bert)
        self.vocab_emb: torch.Tensor = self.span_bert.embeddings.word_embeddings.weight
        self.vocab_size, self.bert_size = self.vocab_emb.shape

        self.mlm_transform = nn.Sequential(
            nn.Linear(self.bert_size, self.bert_size),
            nn.GELU(),
            nn.LayerNorm(self.bert_size),
        )

        self.sbo_embedding = nn.Embedding(self.mask_length, self.bert_size)
        self.sbo_transform = nn.Sequential(
            nn.Linear(self.bert_size * 3, self.bert_size * 2),
            nn.GELU(),
            nn.LayerNorm(self.bert_size*2),
            nn.Linear(self.bert_size * 2, self.bert_size),
            nn.GELU(),
            nn.LayerNorm(self.bert_size)
        )

        self.rel_transform = nn.Sequential(
            nn.Linear(self.bert_size, self.bert_size),
            nn.GELU(),
            nn.LayerNorm(self.bert_size)
        )

        self.rel_infer = self.rel_pred()

    def forward(self, ids, att_mask=None, token_types=None, sp_token_mask=None, prompt=None):
        tokens_rep = self.span_bert(
            ids,
            attention_mask=att_mask,
            token_type_ids=token_types,
        ).last_hidden_state
        # head entity was masked as 1, mask as 2 and tail entity as 3
        sp_mask = sp_token_mask.unsqueeze(-1)
        h_rep = tokens_rep.mul(sp_mask == 1).sum(1).div(torch.sum(sp_mask == 1, 1))
        t_rep = tokens_rep.mul(sp_mask == 3).sum(1).div(torch.sum(sp_mask == 3, 1))
        pair_rep = torch.cat((h_rep, t_rep), -1)
        mask_rep = torch.masked_select(tokens_rep, sp_mask == 2).view(
            tokens_rep.shape[0], self.mask_length, self.bert_size)

        # mlm_loss and sbo_loss return loss and mask_rep while prompt is not None,
        # otherwise return mask prediction and mask_rep
        mlm_out, mlm_rep = self.mlm_loss(mask_rep, prompt)
        sbo_out, sbo_rep = self.sbo_loss(pair_rep, prompt)

        if self.training:
            prompt_mask = prompt > 0
            logit, rel_rep = self.prompt2rel(mlm_rep, sbo_rep, prompt_mask)
        else:
            # predict masked tokens
            avg_rep = torch.mean(torch.stack((mlm_rep, sbo_rep)), 0)
            # [b, v, mask_length]
            token_logit = torch.matmul(self.vocab_emb, torch.transpose(avg_rep, 2, 1))
            pseudo_mask = torch.argmax(token_logit, 1) > 0
            logit, rel_rep = self.prompt2rel(mlm_rep, sbo_rep, pseudo_mask)

        # relation transfer regularization h+r=t
        residual = t_rep - h_rep - rel_rep
        trans_norm = torch.norm(residual, p=2, dim=-1)
        return mlm_out.mean(), sbo_out.mean(), trans_norm.mean(), logit

    def mlm_loss(self, mask_rep, prompt):
        trans_rep = self.mlm_transform(mask_rep)
        loss = self.mask_loss(trans_rep, prompt)
        return loss, trans_rep

    def sbo_loss(self, mention_rep, prompt):
        pos = torch.arange(0, self.mask_length, dtype=torch.int).to(mention_rep.device)
        pos_emb = self.sbo_embedding(pos)
        reps_ali = mention_rep.unsqueeze(1).tile((self.mask_length, 1))
        pos_ali = pos_emb.unsqueeze(0).tile((mention_rep.shape[0], 1, 1))
        # [b, mask_length, 3*hidden_size]
        sbo_inputs = torch.concat((reps_ali, pos_ali), -1)
        mask_reps = self.sbo_transform(sbo_inputs)
        loss = self.mask_loss(mask_reps, prompt)
        return loss, mask_reps

    def prompt2rel(self, mlm_rep, sbo_rep, prompt_mask):
        # combine mask representations of mlm and sbo
        prompt_rep = torch.mean(torch.stack((mlm_rep, sbo_rep)), 0)
        prompt_tran = self.rel_transform(prompt_rep)
        # gather relation representations while ignoring the mask of padding
        masked_rep = prompt_tran.mul(prompt_mask.unsqueeze(-1))
        rel_rep = masked_rep.sum(1).div(prompt_mask.sum(-1, keepdim=True))

        logit = self.rel_infer(rel_rep)
        return logit, rel_rep

    def rel_pred(self):
        predictor = nn.Linear(self.bert_size, self.rel_num, bias=False)
        if self.rel_prompts is not None:
            # [rel_num, dim_prompt, bert_dim]
            rel_emb = self.span_bert.embeddings.word_embeddings(self.rel_prompts)
            mask = self.rel_prompts != self.padding_id
            mask = torch.unsqueeze(mask.type_as(rel_emb), -1)
            rel_emb_masked = torch.mul(rel_emb, mask)
            # [rel_num, bert_dim]
            init = torch.mean(rel_emb_masked, 1)
            predictor.weight.data = init.detach()
        return predictor

    def mask_loss(self, mask_rep, prompt):
        # [b, v, mask_length]
        logit = torch.matmul(self.vocab_emb, torch.transpose(mask_rep, 2, 1))
        p = torch.softmax(logit, 1)
        if prompt is not None:
            # [b, mask_length, 1]
            target_p = torch.gather(torch.transpose(p, 2, 1), 2, prompt.unsqueeze(-1))
            outputs = torch.sum(- torch.log(target_p+1e-7), 1)
        else:
            outputs = torch.argmax(p, 1)
        return outputs

