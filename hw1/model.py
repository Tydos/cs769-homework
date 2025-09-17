import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    emb = np.random.uniform(-0.05, 0.05, (len(vocab), emb_size)).astype(np.float32)
    word2idx = getattr(vocab, 'word2idx', vocab)
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split()
            if len(values) < emb_size + 1:
                continue
            word = values[0]
            if word in word2idx:
                idx = word2idx[word]
                vector = np.asarray(values[1:emb_size+1], dtype=np.float32)
                emb[idx] = vector
    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)
        self.fc = nn.Linear(self.args.emb_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        v = 0.05
        nn.init.uniform_(self.embedding.weight, -v, v)
        nn.init.uniform_(self.fc.weight, -v, v)
        nn.init.zeros_(self.fc.bias)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb_matrix = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        self.embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        emb = self.embedding(x)  # [batch_size, seq_length, emb_size]
        mask = (x != 0).float().unsqueeze(-1)  # assuming 0 is the padding index
        emb = emb * mask
        summed = emb.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        avg_emb = summed / lengths
        scores = self.fc(avg_emb)
        return scores
