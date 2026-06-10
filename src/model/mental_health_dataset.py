import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Maps string labels to integer class IDs for the model
LABEL2ID = {
    "Normal":0,
    "Anxiety":1,
    "Depression":2,
    "Bipolar":3,
    "Stress":4,
    "Personality disorder":5,
    "Suicidal":6,
}

def clean(text):
    """Lowercase and remove punctuation from text. will use while parsing training data"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def load_vocab_and_vectors(vocab_path, vector_path):
    """
    Loads vocab and pretrained glove embedding vectors
    
    Arguments:
    vocab_path: path to vocab.txt (word + frequency per line)
    vector_path: path to vectors_imbued.txt (word + embedding)

    Returns:
        vocab: dict mapping word -> integer ID
        vectors: FloatTensor of shape (vocab_size + 2, embeddings) (+2 because of adding punk_id and unk_id)
        UNK_ID: integer ID for unknown words
        PAD_ID: integer ID for padding tokens
    """
    # Line number becomes the word's integer ID
    vocab = {}
    with open(vocab_path, "r") as f:
        for idx, line in enumerate(f):
            word = line.split()[0]
            vocab[word] = idx

    # Load embedding vectors, skipping the word on each line
    vectors = []
    with open(vector_path, "r") as f:
        for line in f:
            values = line.split()
            vector = [float(v) for v in values[1:]]
            vectors.append(vector)

    if len(vectors) == 0:
        raise ValueError(f"No vectors found in {vector_path}")

    embedding_dim = len(vectors[0])

    # Add <unk> to vocab - vector already exists at end of vectors file
    # Add <PAD> to vocab - append a zero vector (padding carries no meaning)
    vocab["<unk>"] = len(vocab)
    vocab["<PAD>"] = len(vocab)
    pad_vector = np.zeros((1, embedding_dim))
    vectors = np.vstack([vectors, pad_vector])

    UNK_ID = vocab["<unk>"]
    PAD_ID = vocab["<PAD>"]

    # Convert to FloatTensor for use in PyTorch embedding layer
    vectors = torch.FloatTensor(vectors)
    
    return vocab, vectors, UNK_ID, PAD_ID


class MentalHealthDataset(Dataset):
    """
    PyTorch Dataset for the mental health classification task.
    Converts raw text and string labels into tensors for the model.
    """
    def __init__(self, df, vocab, max_len, pad_id, unk_id):
        """
        df: pandas DataFrame with 'statement' and 'status' columns
        vocab: word -> integer ID dictionary
        max_len: all sequences are truncated or padded to this length
        pad_id: integer ID used for padding tokens
        unk_id: integer ID used for unknown words
        """
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = pad_id
        self.unk_id = unk_id

    def __len__(self):
        """Returns total number of examples in the dataset."""
        return len(self.df)

    def __getitem__(self, i):
        """
        Returns the i-th example as a tuple of tensors:
            X: token ID sequence of length max_len
            attention_mask: 1 for real tokens, 0 for padding
            Y: integer class label
        """
        statement = self.df.iloc[i]["statement"]
        label = self.df.iloc[i]["status"]

        # Clean and tokenize the statement
        cleaned = clean(statement)
        tokens = cleaned.split()

        # Convert words to integer IDs, using unk_id for unknown words
        token_ids = [self.vocab.get(t, self.unk_id) for t in tokens]

        # Truncate if longer than max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]

        # Pad with pad_id if shorter than max_len
        while len(token_ids) < self.max_len:
            token_ids.append(self.pad_id)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1 if token != self.pad_id else 0 for token in token_ids]

        # Convert to tensors
        X = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        Y = torch.tensor(LABEL2ID[label], dtype=torch.long)

        return X, attention_mask, Y


def load_data(data_path, vocab, max_len, batch_size, pad_id, unk_id):
    """
    Loads the classification dataset and returns DataLoaders for
    train, validation, and test splits (70/15/15).

    data_path: path to the CSV file
    vocab: word -> integer ID dictionary
    max_len: sequence length to pad/truncate to
    batch_size: number of examples per batch
    pad_id: integer ID for padding tokens
    unk_id: integer ID for unknown words

    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_df = pd.read_csv(data_path)

    # Stratified split into 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(dataset_df, test_size=0.30, random_state=12, stratify=dataset_df["status"],)
    val_df, test_df   = train_test_split(temp_df,    test_size=0.50, random_state=12, stratify=temp_df["status"],)

    # Wrap each split in a Dataset object
    train_ds = MentalHealthDataset(train_df, vocab, max_len=max_len, pad_id=pad_id, unk_id=unk_id)
    val_ds   = MentalHealthDataset(val_df,   vocab, max_len=max_len, pad_id=pad_id, unk_id=unk_id)
    test_ds  = MentalHealthDataset(test_df,  vocab, max_len=max_len, pad_id=pad_id, unk_id=unk_id)

    # Shuffle only the training set
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader