import os

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from datasets import DatasetCSTS
from model import BERTModel
import torch
from tqdm import tqdm

class BertTrainer:
    def __init__(self, model, dataset, tokenizer, batch_size=16, learning_rate=1e-4, epochs=5, device=None,checkpoint_dir='bert_checkpoints'):
        self.model = model.to(device)
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.current_epoch = 0
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate,weight_decay=0.01)
        self.loss_fn = nn.MSELoss()

    def train(self):
        print(f"Begin epoch {self.current_epoch}")
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(self.loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                embeddings1 = [self.tokenizer.encode(sentence1) for sentence1 in batch["sentence1"]]
                embeddings2 = [self.tokenizer.encode(sentence2) for sentence2 in batch["sentence2"]]
                scores = batch["score"].to(device)

                embeddings1 = [self.model(torch.tensor([embedding])).squeeze(0) for embedding in embeddings1]
                embeddings2 = [self.model(torch.tensor([embedding])).squeeze(0) for embedding in embeddings2]

                # 计算余弦相似度
                cos_sim = nn.functional.cosine_similarity(torch.stack(embeddings1), torch.stack(embeddings2))

                # 计算损失
                loss = self.loss_fn(cos_sim, scores)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss+=loss.item()
                progress_bar.set_postfix({"Loss": loss.item()})

            avg_loss = total_loss / len(self.loader)
            print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
            # 每个 epoch 结束后保存模型
            self.save_checkpoint(self.current_epoch+epoch)
            self.save_model(self.current_epoch+epoch)

    def save_model(self,epoch):
        # 创建目录
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # 定义模型文件名
        model_name = f"model_bert_embedding_{self.model.d_model}_{self.model.num_blocks}_epoch_{epoch}.pt"
        model_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(self.model,model_path)
        print(f"Model saved at {model_path}")

    def save_checkpoint(self, epoch):
        # 创建目录
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # 定义检查点文件名
        checkpoint_name = f"checkpoint_bert_embedding_{self.model.d_model}_{self.model.num_blocks}_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # 保存模型、优化器状态和当前 epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)

        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, model_file):
        print(f"Restoring model {model_file}")
        checkpoint = torch.load(model_file)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model is restored.")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('../google-bert/bert-base-chinese')
    # 模型参数
    batch_size = 16
    vocab_size = len(tokenizer.vocab)
    d_model = 1024
    n_heads = 16
    head_size = d_model // n_heads
    n_layers = 12
    context_length = 256
    dropout = 0.1
    epochs = 3
    learning_rate = 1e-4
    # transformer块数量
    num_blocks = 6
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    dataset = DatasetCSTS(file_path="../simtrain_to05sts.txt", tokenizer=tokenizer)
    model = BERTModel(vocab_size, d_model, n_heads, head_size, context_length, num_blocks, dropout, device)
    trainer = BertTrainer(model,dataset,tokenizer,batch_size,learning_rate=learning_rate,epochs=epochs,device=device)
    # trainer.load_checkpoint('./bert_checkpoints/checkpoint_bert_embedding_1024_6_epoch_10.pt')
    trainer.train()