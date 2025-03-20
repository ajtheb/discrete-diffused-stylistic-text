import spacy
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
# import matplotlib.pyplot as plt
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize spaCy tokenizer
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger"])

def tokenize(text):
    """Custom tokenizer using spaCy"""
    return [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]

def build_vocab(texts, max_size=20000, min_freq=2):
    """Build vocabulary from tokenized texts"""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for token, count in counter.most_common(max_size):
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab

class TextDataset(Dataset):
    def __init__(self, csv_file, tokenizer_fn, vocab):
        self.data = pd.read_csv(csv_file)
        self.texts = self.data['Counterspeech'].values
        self.tokenizer = tokenizer_fn
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text)
        numericalized = [self.vocab.get(token, 1) for token in tokens]  # 1 = <unk>
        return numericalized

def collate_batch(batch, pad_to_length =256):
    """Handle variable-length sequences with padding"""
    batch = [torch.tensor(item, dtype=torch.long) for item in batch]
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    # If the padded length is less than the desired length, add additional padding
    if padded.size(1) < pad_to_length:
        additional_padding = torch.zeros(
            (padded.size(0), pad_to_length - padded.size(1)), dtype=torch.long
        )
        padded = torch.cat([padded, additional_padding], dim=1)
    
    # If the padded length is greater than the desired length, truncate
    elif padded.size(1) > pad_to_length:
        padded = padded[:, :pad_to_length]
    return padded


class TextVQVAE(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
        super(TextVQVAE, self).__init__()
        self.emb_dim = emb_dim
        self.commitment_cost = commitment_cost

        # Text embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Encoder (1D convolutions)
        self.encoder = nn.Sequential(
            nn.Conv1d(emb_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, kernel_size=3, padding=1)
        )

        # Vector Quantization
        self.vq_layer = VQEmbedding(num_embeddings, latent_dim, commitment_cost)

        # Decoder (1D transposed convolutions)
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, emb_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv1d(emb_dim, vocab_size, kernel_size=1)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, emb_dim, seq_len)
        
        z_e = self.encoder(emb)
        z_q, vq_loss = self.vq_layer(z_e)
        recon_logits = self.decoder(z_q).permute(0, 2, 1)  # (batch, seq_len, vocab_size)
        
        return recon_logits, vq_loss

class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z shape: (batch_size, latent_dim, seq_len)
        batch, dim, seq = z.size()
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, dim)  # (batch*seq, latent_dim)

        # Calculate distances
        distances = (torch.sum(z_flat**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_flat, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(batch, seq, dim).permute(0, 2, 1)

        # Commitment loss
        loss = F.mse_loss(z_q, z.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss

# Training adjustments
def vqvae_loss(recon_logits, target, vq_loss):
    recon_loss = F.cross_entropy(
        recon_logits.reshape(-1, recon_logits.size(-1)),  # Changed to reshape
        target.view(-1)
    )
    return recon_loss + vq_loss



# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



# Usage example
if __name__ == "__main__":
    # Load data and build vocab
    csv_path = "input/Person_specific_counterspeech3.csv"
    data = pd.read_csv(csv_path)
    # data = data[:1000]
    all_texts = data['Counterspeech'].tolist()
    vocab = build_vocab(all_texts)
    
    # Create dataset and dataloader
    dataset = TextDataset(csv_path, tokenize, vocab)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_batch
    )

    # Test one batch
    sample_batch = next(iter(train_loader))
    print(f"Batch shape: {sample_batch.shape}")
    print(f"Sample sequence: {sample_batch[0]}")
    # print(tokenize("I will one day"))
    
    
    # Example usage
    model = TextVQVAE(
        vocab_size=len(vocab), 
        emb_dim=256,
        latent_dim=64,
        num_embeddings=512
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    num_epochs = 30  # You can adjust this number based on your needs

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = batch.to(device)
            
            optimizer.zero_grad()
            recon_logits, vq_loss = model(inputs)
            loss = vqvae_loss(recon_logits, inputs, vq_loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")
