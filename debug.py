from load_data import TacredParse, SemEvalParse
from transformers import AutoModel
from torch.utils.data import DataLoader
from train import collate_fn


# x = TacredParse("data/tacred/data/json/dev.json")
x = SemEvalParse("data/SemEval2010/test.txt")

d = DataLoader(x, batch_size=3, collate_fn=collate_fn)

m = AutoModel.from_pretrained("spanbert-base-cased/")
a = m()

