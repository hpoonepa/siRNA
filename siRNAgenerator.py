import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import os

# -----------------------------
# Global Definitions
# -----------------------------
NT2IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '<PAD>': 4}
IDX2NT = {0: 'A', 1: 'U', 2: 'G', 3: 'C', 4: '<PAD>'}
PAD_IDX = NT2IDX['<PAD>']
VOCAB_SIZE = 5

# -----------------------------
# Utilities for Sequence Processing
# -----------------------------
def standardize_sequence(seq):
    """Strip whitespace, uppercase letters, and convert T to U."""
    return seq.strip().upper().replace("T", "U")

def force_complement(seq):
    """Return the complement of the RNA sequence."""
    COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    return "".join(COMPLEMENT.get(nt, 'N') for nt in seq)

def sequence_to_graph(seq):
    """
    Convert a sequence string into node features (one-hot for A, U, G, C)
    and create an edge index linking consecutive nucleotides.
    """
    node_feats = []
    for nt in seq:
        onehot = [0, 0, 0, 0]
        if nt in NT2IDX and nt != '<PAD>':
            onehot[NT2IDX[nt]] = 1
        node_feats.append(onehot)
    x = torch.tensor(node_feats, dtype=torch.float)
    num_nodes = x.size(0)
    if num_nodes < 2:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(
            [[i for i in range(num_nodes - 1)], [i + 1 for i in range(num_nodes - 1)]],
            dtype=torch.long
        )
    return x, edge_index

def calculate_gc_content(seq):
    """Calculate the GC content (fraction of G and C nucleotides)."""
    if len(seq) == 0:
        return 0.0
    return (seq.count('G') + seq.count('C')) / len(seq)

def create_data_object_from_row(mrna_seq, sirna_seq):
    """
    Given an mRNA and a target siRNA sequence, preprocess and create a
    PyTorch Geometric Data object. The function also checks whether the
    siRNA is contained in (or its complement is contained in) the mRNA.
    """
    mrna_seq = standardize_sequence(mrna_seq)
    sirna_seq = standardize_sequence(sirna_seq)
    # Check if the siRNA (or its complement) is in the mRNA.
    start_idx = mrna_seq.find(sirna_seq)
    if start_idx == -1:
        start_idx = mrna_seq.find(force_complement(sirna_seq))
    if start_idx == -1:
        return None
    x, edge_index = sequence_to_graph(mrna_seq)
    gc_content = calculate_gc_content(mrna_seq)
    data = Data(x=x, edge_index=edge_index)
    data.gc_content = torch.tensor([gc_content], dtype=torch.float)
    data.seq_length = torch.tensor([len(mrna_seq)], dtype=torch.float)
    # Store the ground truth siRNA sequence.
    data.sirna_seq = sirna_seq  
    return data

def build_dataset_from_csv(csv_path):
    """Build a dataset (list of Data objects) from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    data_list = []
    for _, row in df.iterrows():
        mrna_seq = row['sequence_mRNA']
        sirna_seq = row['sequence_siRNA']
        data = create_data_object_from_row(mrna_seq, sirna_seq)
        if data is not None:
            data_list.append(data)
    print(f"Samples after filtering: {len(data_list)}")
    return data_list

# -----------------------------
# Helper Functions for siRNA Sequences
# -----------------------------
def seq_to_indices(seq):
    """Convert a nucleotide sequence into a list of indices."""
    return [NT2IDX[nt] for nt in seq if nt in NT2IDX and nt != '<PAD>']

def pad_seq(seq_indices, max_length, pad_value=PAD_IDX):
    """Pad (or truncate) a list of indices to a fixed length."""
    seq_indices = seq_indices[:max_length]
    padded = seq_indices + [pad_value] * (max_length - len(seq_indices))
    return padded

def indices_to_seq(indices):
    """Convert a list of indices back into a nucleotide string, ignoring PAD tokens."""
    return "".join([IDX2NT[i] for i in indices if i != PAD_IDX])

def compute_rule_loss(siRNA_seq):
    """
    Compute a penalty loss based on a set of siRNA design rules.
    (Rules include constraints on GC content, specific positions, and motifs.)
    Returns a scalar penalty.
    """
    loss = 0.0
    L = len(siRNA_seq)
    if L == 0:
        return 10.0  # High penalty for empty sequence.
    # Rule 1: GC content between 35% and 73%
    gc = (siRNA_seq.count('G') + siRNA_seq.count('C')) / L
    if gc < 0.35:
        loss += (0.35 - gc)
    elif gc > 0.73:
        loss += (gc - 0.73)
    # Rule 2: Must start with U
    if siRNA_seq[0] != 'U':
        loss += 0.5
    # Rule 3: Position 10 (index 9) must be A
    if L >= 10 and siRNA_seq[9] != 'A':
        loss += 0.5
    # Rule 4: Positions 14 and 18 (indexes 13 and 17) must not be G
    if L >= 14 and siRNA_seq[13] == 'G':
        loss += 0.5
    if L >= 18 and siRNA_seq[17] == 'G':
        loss += 0.5
    # Rule 5: Position 19 (index 18) must not be A
    if L >= 19 and siRNA_seq[18] == 'A':
        loss += 0.5
    # Rule 6: Must contain motifs "UCU" and "UCCG"
    if "UCU" not in siRNA_seq:
        loss += 0.5
    if "UCCG" not in siRNA_seq:
        loss += 0.5
    # Rule 7: Must not contain motifs "ACGA", "GCC", "GUGG"
    for motif in ["ACGA", "GCC", "GUGG"]:
        if motif in siRNA_seq:
            loss += 0.5
    # Rules 8-10: (Placeholders for additional criteria)
    return loss

# -----------------------------
# siRNA Generator Model (Encoder-Decoder)
# -----------------------------
class SiRNAGenerator(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, encoder_hidden=128, num_layers=3, dropout=0.3, max_length=21):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size

        # Encoder: GNN for mRNA graph.
        self.encoder_convs = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        self.encoder_convs.append(TransformerConv(in_channels=4, out_channels=encoder_hidden, heads=2, concat=True))
        self.encoder_norms.append(nn.LayerNorm(encoder_hidden * 2))
        for _ in range(num_layers - 1):
            self.encoder_convs.append(TransformerConv(encoder_hidden * 2, encoder_hidden, heads=2, concat=True))
            self.encoder_norms.append(nn.LayerNorm(encoder_hidden * 2))
        self.encoder_dropout = nn.Dropout(dropout)
        # After pooling, concatenate global features (gc_content and seq_length).
        self.enc_fc = nn.Linear(encoder_hidden * 2 + 2, embed_dim)

        # Decoder: LSTM for sequence generation.
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_lstm = nn.LSTM(embed_dim, embed_dim, num_layers=1, batch_first=True)
        self.decoder_fc = nn.Linear(embed_dim, vocab_size)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for conv, norm in zip(self.encoder_convs, self.encoder_norms):
            x = F.relu(conv(x, edge_index))
            x = norm(x)
            x = self.encoder_dropout(x)
        x_pool = global_mean_pool(x, batch)
        global_feats = torch.cat([
            x_pool,
            data.gc_content.view(-1, 1),
            data.seq_length.view(-1, 1)
        ], dim=1)
        enc_out = F.relu(self.enc_fc(global_feats))
        return enc_out  # (batch_size, embed_dim)

    def decode(self, enc_hidden, targets=None, teacher_forcing_ratio=0.5):
        batch_size = enc_hidden.size(0)
        device = enc_hidden.device
        # Use U as the start token.
        sos_token = torch.tensor([NT2IDX['U']], device=device)
        inputs = sos_token.expand(batch_size)
        inputs = self.embedding(inputs).unsqueeze(1)  # (batch_size, 1, embed_dim)
        hidden = (enc_hidden.unsqueeze(0), torch.zeros(1, batch_size, enc_hidden.size(1), device=device))
        outputs = []
        for t in range(self.max_length):
            out, hidden = self.decoder_lstm(inputs, hidden)
            logits = self.decoder_fc(out.squeeze(1))  # (batch_size, vocab_size)
            outputs.append(logits.unsqueeze(1))
            # Decide whether to use teacher forcing.
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                token = targets[:, t]
            else:
                token = logits.argmax(dim=1)
            inputs = self.embedding(token).unsqueeze(1)
        outputs = torch.cat(outputs, dim=1)  # (batch_size, max_length, vocab_size)
        return outputs

    def forward(self, data, targets=None, teacher_forcing_ratio=0.5):
        enc_hidden = self.encode(data)
        outputs = self.decode(enc_hidden, targets, teacher_forcing_ratio)
        return outputs

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def train_generator_epoch(model, loader, optimizer, device, lambda_rule=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        target_seqs = []
        for data in batch.to_data_list():
            seq_indices = seq_to_indices(data.sirna_seq)
            padded = pad_seq(seq_indices, model.max_length, pad_value=PAD_IDX)
            target_seqs.append(torch.tensor(padded, dtype=torch.long))
        targets = torch.stack(target_seqs, dim=0).to(device)
        
        optimizer.zero_grad()
        outputs = model(batch, targets, teacher_forcing_ratio=0.7)
        ce_loss = F.cross_entropy(outputs.view(-1, model.vocab_size), targets.view(-1))
        
        preds = outputs.argmax(dim=2)  # (batch_size, max_length)
        rule_loss = 0.0
        for seq_indices in preds:
            siRNA_str = indices_to_seq(seq_indices.tolist())
            rule_loss += compute_rule_loss(siRNA_str)
        rule_loss = rule_loss / preds.size(0)
        
        loss = ce_loss + lambda_rule * rule_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_generator(model, loader, device, lambda_rule=1.0):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target_seqs = []
            for data in batch.to_data_list():
                seq_indices = seq_to_indices(data.sirna_seq)
                padded = pad_seq(seq_indices, model.max_length, pad_value=PAD_IDX)
                target_seqs.append(torch.tensor(padded, dtype=torch.long))
            targets = torch.stack(target_seqs, dim=0).to(device)
            outputs = model(batch, targets, teacher_forcing_ratio=0.0)
            ce_loss = F.cross_entropy(outputs.view(-1, model.vocab_size), targets.view(-1))
            preds = outputs.argmax(dim=2)
            rule_loss = 0.0
            for seq_indices in preds:
                siRNA_str = indices_to_seq(seq_indices.tolist())
                rule_loss += compute_rule_loss(siRNA_str)
            rule_loss = rule_loss / preds.size(0)
            loss = ce_loss + lambda_rule * rule_loss
            total_loss += loss.item()
    return total_loss / len(loader)

def main_generator():
    """
    Train the siRNA generator model. Expects a CSV file named 
    'combined_dataset.csv' with columns 'sequence_mRNA' and 'sequence_siRNA'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = "combined_dataset.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Please provide the dataset CSV file.")
        return

    data_list = build_dataset_from_csv(csv_path)
    if len(data_list) == 0:
        print("No valid samples found. Please check your CSV file and sequence formats.")
        return

    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = SiRNAGenerator(vocab_size=VOCAB_SIZE, embed_dim=64, encoder_hidden=128, 
                           num_layers=3, dropout=0.3, max_length=21)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    num_epochs = 50
    lambda_rule = 1.0

    for epoch in range(num_epochs):
        train_loss = train_generator_epoch(model, train_loader, optimizer, device, lambda_rule)
        test_loss = evaluate_generator(model, test_loader, device, lambda_rule)
        scheduler.step(test_loss)
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    torch.save(model.state_dict(), "sirna_generator_model.pth")
    print("Generator model saved as 'sirna_generator_model.pth'")

# -----------------------------
# Generation Function
# -----------------------------
def generate_sirna_from_mrna(mrna_seq, model, device):
    """
    Given an input mRNA sequence, preprocess it, pass it through the model,
    and return the generated siRNA sequence.
    """
    mrna_seq = standardize_sequence(mrna_seq)
    x, edge_index = sequence_to_graph(mrna_seq)
    gc_content = calculate_gc_content(mrna_seq)
    data = Data(x=x, edge_index=edge_index)
    data.gc_content = torch.tensor([gc_content], dtype=torch.float)
    data.seq_length = torch.tensor([len(mrna_seq)], dtype=torch.float)
    batch_data = Batch.from_data_list([data]).to(device)

    model.eval()
    with torch.no_grad():
        enc_hidden = model.encode(batch_data)
        outputs = model.decode(enc_hidden, targets=None, teacher_forcing_ratio=0.0)
        preds = outputs.argmax(dim=2)
        predicted_seq = indices_to_seq(preds[0].tolist())
    return predicted_seq

def generate_mode():
    """
    Load the trained model and prompt the user for an mRNA sequence,
    then generate and print the corresponding siRNA sequence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiRNAGenerator(vocab_size=VOCAB_SIZE, embed_dim=64, encoder_hidden=128, 
                           num_layers=3, dropout=0.3, max_length=21)
    model = model.to(device)
    model_path = "sirna_generator_model.pth"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Please train the model first using --train.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded from", model_path)
    
    input_mrna = input("Enter an mRNA sequence: ").strip()
    generated_sirna = generate_sirna_from_mrna(input_mrna, model, device)
    print("Input mRNA sequence: ", input_mrna)
    print("Generated siRNA sequence:", generated_sirna)