import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

# ========= Define Model =========
class SiRNAEfficiencyPredictorGNN(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()

        from torch_geometric.nn import TransformerConv, global_mean_pool

        self.conv1 = TransformerConv(in_channels, hidden_dim, heads=2, concat=True)
        self.norm1 = nn.LayerNorm(hidden_dim * 2)
        
        self.conv2 = TransformerConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        self.conv3 = TransformerConv(hidden_dim * 2, hidden_dim, heads=2, concat=True)
        self.norm3 = nn.LayerNorm(hidden_dim * 2)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = nn.functional.relu(self.conv1(x, edge_index))
        x = self.norm1(x)
        x = self.dropout(x)

        x = nn.functional.relu(self.conv2(x, edge_index))
        x = self.norm2(x)
        x = self.dropout(x)

        x = nn.functional.relu(self.conv3(x, edge_index))
        x = self.norm3(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))  # Ensure output is between 0 and 1
        return x.view(-1)

# ========= Define Data Processing Functions =========
def standardize_sequence(seq):
    return seq.strip().upper().replace("T", "U")

def force_complement(seq):
    COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
    return "".join(COMPLEMENT.get(nt, 'N') for nt in seq)

def sequence_to_graph(seq):
    NT2IDX = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
    node_feats = []
    for nt in seq:
        onehot = [0, 0, 0, 0]
        if nt in NT2IDX:
            onehot[NT2IDX[nt]] = 1
        node_feats.append(onehot)
    x = torch.tensor(node_feats, dtype=torch.float)
    num_nodes = x.size(0)
    if num_nodes < 2:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([[i for i in range(num_nodes - 1)], [i + 1 for i in range(num_nodes - 1)]], dtype=torch.long)
    return x, edge_index

def create_data_object(mrna_seq, sirna_seq):
    mrna_seq = standardize_sequence(mrna_seq)
    sirna_seq = standardize_sequence(sirna_seq)

    start_idx = mrna_seq.find(sirna_seq)
    if start_idx == -1:
        start_idx = mrna_seq.find(force_complement(sirna_seq))
    if start_idx == -1:
        start_idx = len(mrna_seq) // 2  # Default index

    x, edge_index = sequence_to_graph(mrna_seq)
    y = torch.tensor([float(start_idx) / len(mrna_seq)], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# ========= Load Dataset & Create DataLoader =========
dataset_path = "combined_dataset.csv"

df = pd.read_csv(dataset_path)
data_list = [create_data_object(row["mRNA"], row["siRNA"]) for _, row in df.iterrows()]
data_list = [d for d in data_list if d is not None]

train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_data)}, Testing samples: {len(test_data)}")

# ========= Train the Model =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiRNAEfficiencyPredictorGNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}")

# ========= Save the Model =========
torch.save(model.state_dict(), "sirna_efficiency_predictor_model.pth")
print("Model retrained and saved successfully!")
