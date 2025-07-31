import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!


    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        # 토큰화 및 패딩 제거
        tokenized = tokenizer(corpus, padding=False, truncation=True).input_ids
        tokenized = [seq for seq in tokenized if len(seq) >= self.window_size * 2 + 1]

        for epoch in range(num_epochs):
            total_loss = 0
            for sequence in tokenized:
                sequence = torch.tensor(sequence)
                loss = self._train_cbow(sequence, criterion)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Epoch {epoch+1}] loss: {total_loss / len(tokenized):.4f}")

    def _train_cbow(
        self,
        # 구현하세요!
        sequence: Tensor,
        criterion: nn.Module
    ) -> Tensor:
        # 구현하세요!
        loss = 0.0
        length = sequence.size(0)

        for center in range(self.window_size, length - self.window_size):
            context_ids = [i for i in range(center - self.window_size, center + self.window_size + 1) if i != center]
            context = sequence[context_ids]  # 주변 단어
            target = sequence[center]        # 예측 대상

            context_embed = self.embeddings(context)  # (2w, d)
            context_mean = context_embed.mean(dim=0)  # (d,)

            logits = self.weight(context_mean)        # (vocab_size,)
            loss += criterion(logits.unsqueeze(0), target.unsqueeze(0))

        return loss / (length - 2 * self.window_size) 


    def _train_skipgram(
        self,
        sequence: Tensor,
        criterion: nn.Module
    ) -> Tensor:
        # 구현하세요!
        loss = 0.0
        length = sequence.size(0)
    
        for center in range(self.window_size, length - self.window_size):
            center_id = sequence[center]
            context_ids = [i for i in range(center - self.window_size, center + self.window_size + 1) if i != center]
    
            center_embed = self.embeddings(center_id)  # (d_model,)
    
            for idx in context_ids:
                context_word_id = sequence[idx]
                logits = self.weight(center_embed)      # (vocab_size,)
                loss += criterion(logits.unsqueeze(0), context_word_id.unsqueeze(0))
    
        return loss / (length - 2 * self.window_size)