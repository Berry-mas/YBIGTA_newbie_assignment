# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    for split in ["train", "validation"]:
        for item in dataset[split]:
            corpus.append(item["verse_text"])

    return corpus