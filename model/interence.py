import torch
from torch import nn
from transformers import BertTokenizer

from model import BERTModel

class BERTEmbedding:
    def __init__(self,model_path,tokenizer_path):
        self.model = torch.load(model_path,weights_only=False)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)

    def cos_similarity(self, sentence1: str, sentence2: str) -> list[float]:
        embedding1 = self.tokenizer.encode(sentence1)[1:-1]
        embedding1 = self.model(torch.tensor([embedding1]))
        embedding2 = self.tokenizer.encode(sentence2)[1:-1]
        embedding2 = self.model(torch.tensor([embedding2]))
        cos_sim = nn.functional.cosine_similarity(embedding1, embedding2)
        return cos_sim.item()

    def embed_query(self, sentence: str) -> float:
        embedding = self.tokenizer.encode(sentence)[1:-1]
        embedding = self.model(torch.tensor([embedding])).squeeze(0)
        return embedding.tolist()



if __name__ == "__main__":
    embedding = BERTEmbedding(model_path="./bert_checkpoints/model_bert_embedding_512_8_epoch_9.pt",
                              tokenizer_path='../google-bert/bert-base-chinese')
    print(embedding.embed_query("她洗净了一件衣服。"))
    print(embedding.embed_query("她把那件衣服洗干净了。"))
    print(embedding.cos_similarity("她洗净了一件衣服。", "她把那件衣服洗干净了。"))
    print(embedding.cos_similarity("他正在阅读一本有趣的小说。", "他正在看一本引人入胜的故事书。"))

    print(embedding.cos_similarity("他经常去健身房锻炼身体。", "他偶尔会去游泳池游泳。"))

    print(embedding.cos_similarity("看什么电视，还不赶快做功课去！", "看电视，不去做功课！"))
    print(embedding.cos_similarity("看到实验成功，大家好高兴。", "看到实验成功，大家不高兴。"))
    print(embedding.cos_similarity("她正在学习法语。", "他们在讨论如何修理汽车。"))
    print(embedding.cos_similarity("猫是一种常见的宠物。", "飞机是现代交通的重要工具。"))
