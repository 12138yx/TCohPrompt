import json
import re
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset
import pickle
import torch


class TacredParse(Dataset):
    def __init__(self, data_path, max_length=128,
                 tokenizer="spanbert-base-cased/", rel_map="data/tacred_rel_map.csv"):
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.rel_map = rel_map

        self.rel_id, self.rel_prompt = self.rel2id()
        self.rel_w, self.rel_cnt = None, None
        self.rel_prompt2tokens, self.rel_prompt2ids = self.prompt_token()
        self.list_inputs = self.get_inputs(self.max_length)
        self.len = len(self.list_inputs[0])

    def parsing(self, path):
        parsed_ = []
        rel_count = [0]*len(self.rel_id)
        with open(path, "r") as f:
            ori_data = json.load(f)
        for d in ori_data:
            r = d["relation"]
            rel_count[self.rel_id[r]] += 1
            rel_id = self.rel_id[r]
            token = d["token"]
            sub = token[d["subj_start"]:d["subj_end"]+1]
            obj = token[d["obj_start"]:d["obj_end"]+1]
            parsed_.append({
                "tokens": token,
                "rel": r,
                "rel_id": rel_id,
                "sub": sub,
                "obj": obj,
            })
            # debug
            # if len(parsed_) > 23:
            #     break
        self.rel_w, self.rel_cnt = self.rel_weights(rel_count)
        return parsed_

    @staticmethod
    def rel_weights(rel_counts: list):
        log_cnt = torch.log(torch.tensor(rel_counts))
        soft = torch.softmax(log_cnt, 0)
        mask = torch.ones_like(log_cnt)
        mask[0] = 0
        neg_log = - torch.log(soft)
        w = soft + mask * neg_log
        return w, rel_counts

    def rel2id(self):
        rel_id = []
        rel_prompt = []
        with open(self.rel_map, "r") as f:
            rels = f.readlines()
            for i in range(len(rels)):
                r, fixed_r = rels[i].strip().split("\t")
                rel_id.append((r, i))
                prompt_type = ["person" if r[:3] == "per" else "organization"] if r.find(":") != -1 else []
                rel_prompt.append((r, prompt_type + fixed_r[fixed_r.find(":")+1:].split("_")))
        return dict(rel_id), dict(rel_prompt)

    def prompt_token(self):
        prompts = list(self.rel_prompt.values())
        ids = self.tokenizer.batch_encode_plus(
            prompts, add_special_tokens=False, is_split_into_words=True, padding=True)["input_ids"]
        tokens = [self.tokenizer.convert_ids_to_tokens(id_) for id_ in ids]
        prompt_tokens = dict(zip(self.rel_prompt.keys(), tokens))
        prompt_ids = dict(zip(self.rel_prompt.keys(), ids))
        return prompt_tokens, prompt_ids

    def tokenize(self):
        rel_prompt = self.rel_prompt2tokens
        rel_padding_num = len(list(rel_prompt.values())[0])
        parsed_ = self.parsing(self.data_path)
        sentences = []
        pairs_ = []
        rels = []
        labels = []
        for s in parsed_:
            sentences.append(s["tokens"])
            r = s["rel"]
            pair_ = [] + s["sub"]
            pair_ += [self.tokenizer.mask_token] * rel_padding_num
            pair_ += s["obj"]
            pairs_.append(pair_)
            rels.append(rel_prompt[r])
            labels.append(self.rel_id[r])
        inputs = self.tokenizer(
            text=sentences, text_pair=pairs_, text_target=rels,
            add_special_tokens=True, is_split_into_words=True)
        inputs["labels_y"] = labels
        return inputs

    def get_inputs(self, max_length=128):
        inputs_ = self.tokenize()
        input_ids, token_type_ids, attention_mask, sp_token_mask = [], [], [], []
        relations, labels = [], []
        for i in range(len(inputs_["input_ids"])):
            len_ = len(inputs_["input_ids"][i])
            if len_ > max_length:
                continue
            padding = [0]*(max_length-len_)
            input_ids.append(inputs_["input_ids"][i]+padding)
            token_type_ids.append(inputs_["token_type_ids"][i]+padding)
            attention_mask.append(inputs_["attention_mask"][i]+padding)
            sp_token_mask.append(self.generate_special_token_mask(inputs_["input_ids"][i])+padding)
            relations.append(inputs_["labels"][i][1:-1])  # fix-sized relation prompt
            labels.append(inputs_["labels_y"][i])  # single relation ids
        return input_ids, token_type_ids, attention_mask, sp_token_mask, relations, labels

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = lambda data_id: self.list_inputs[data_id][idx]
        return data(0), data(1), data(2), data(3), data(4), data(5)

    def generate_special_token_mask(self, input_ids: list):
        n = len(input_ids)
        mask_id = self.tokenizer.mask_token_id
        sep_id = self.tokenizer.sep_token_id
        mask = []
        sep_pos = input_ids.index(sep_id)
        mask += [0]*(sep_pos+1)
        mask_pos = input_ids.index(mask_id)
        mask += [1]*(mask_pos-sep_pos-1)
        mask2_pos = n-input_ids[::-1].index(mask_id)-1
        mask += [2]*(mask2_pos-mask_pos+1) + [3]*(n-mask2_pos-2) + [0]
        return mask


class SemEvalParse(TacredParse):
    def __init__(self, data_path, max_length=128,
                 tokenizer="bert-base-cased", rel_map="data/SemEval2010/rel_map.csv"):
        super(SemEvalParse, self).__init__(data_path, max_length=max_length,
                                           tokenizer=tokenizer, rel_map=rel_map)

    def parsing(self, path):
        parsed_ = []
        rel_count = [0] * len(self.rel_id)

        def revise_tokens(s: str, sub_):
            e1 = s[s.index("<e1>")+4:s.index("</e1>")].split(" ")
            e2 = s[s.index("<e2>") + 4:s.index("</e2>")].split(" ")
            s = re.sub(r"<[/]?e[12]>", "", s)
            if sub_ == "1":
                h, t = e1, e2
            else:
                h, t = e2, e1
            return s.split(" "), h, t

        with open(path, "r") as f:
            lines = f.readlines()
            assert len(lines) % 4 == 0
            for i in range(0, len(lines), 4):
                raw = lines[i]
                striped_ = raw[raw.index("\t")+2:][:-2]
                raw_rel = lines[i+1]
                if raw_rel.find("(") != -1:
                    rel_idx = raw_rel.index("(")
                    r = raw_rel[:rel_idx]
                    h_ = raw_rel[rel_idx+2]
                else:
                    r, h_ = raw_rel[:-1], "1"
                tokens, sub, obj = revise_tokens(striped_, h_)
                rel_count[self.rel_id[r]] += 1
                parsed_.append({
                    "tokens": tokens,
                    "rel": r,
                    "rel_id": self.rel_id[r],
                    "sub": sub,
                    "obj": obj,
                })
                # debug
                # if len(parsed_) > 11:
                #     break
        self.rel_w, self.rel_cnt = self.rel_weights(rel_count)
        return parsed_

    def rel2id(self):
        rel_id = []
        rel_prompt = []
        with open(self.rel_map, "r") as f:
            rels = f.readlines()
            for i in range(len(rels)):
                r = rels[i].split("\t")[0]
                fixed_r = rels[i].strip().split("\t")[1]
                rel_id.append((r, i))
                rel_prompt.append((r, fixed_r.split("-")))
        return dict(rel_id), dict(rel_prompt)


pickle_path = "pickled"


def tacred(data, tokenizer="spanbert-base-cased/", rel_map="data/tacred_rel_map.csv"):
    tra, dev, tes = data
    p_path = os.path.join(pickle_path, tes.split("/")[1]+".plk")
    os.makedirs(pickle_path, exist_ok=True)
    if os.path.exists(p_path):
        with open(p_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = TacredParse(tra, tokenizer=tokenizer, rel_map=rel_map), \
               TacredParse(dev, tokenizer=tokenizer, rel_map=rel_map), \
               TacredParse(tes, tokenizer=tokenizer, rel_map=rel_map)
        with open(p_path, "wb") as f:
            pickle.dump(data, f)
    return data


def semeval(data, rel_map="data/SemEval2010/rel_map.csv"):
    tra, tes = data
    p_path = os.path.join(pickle_path, "semeval.plk")
    if os.path.exists(p_path):
        with open(p_path, 'rb') as f:
            data = pickle.load(f)
    else:
        data = SemEvalParse(tra, rel_map=rel_map), SemEvalParse(tes, rel_map=rel_map)
        with open(p_path, "wb") as f:
            pickle.dump(data, f)
    return data
