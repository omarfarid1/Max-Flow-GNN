import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle

# Define the Graph Neural Network
class MaxFlowGIN(torch.nn.Module):
    def __init__(self):
        super(MaxFlowGIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(1, 64), torch.nn.LeakyReLU(), torch.nn.Linear(64, 64))
        self.conv1 = GINConv(nn1)
        nn2 = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.LeakyReLU(), torch.nn.Linear(128, 128))
        self.conv2 = GINConv(nn2)
        nn3 = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.LeakyReLU(), torch.nn.Linear(256, 256))
        self.conv3 = GINConv(nn3)
        self.fc = torch.nn.Linear(256, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Load the dataset
with open("graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

with open("max_flows.pkl", "rb") as f:
    max_flows = pickle.load(f)

# Convert the dataset to PyTorch Geometric format
data_list = []
for i, (graph, max_flow) in enumerate(zip(graphs, max_flows)):
    edges = [edge for edge in graph.edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor([[1.0] for _ in range(len(graph.nodes))], dtype=torch.float)
    y = torch.tensor([[max_flow]], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, y=y, batch=torch.tensor([i] * len(graph.nodes), dtype=torch.long))
    data_list.append(data)

# Split the dataset into train and test sets
split_idx = int(len(data_list) * 0.9)
train_data = data_list[:split_idx]
test_data = data_list[split_idx:]

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaxFlowGIN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-4)
criterion = torch.nn.MSELoss()

# Train the model
def train():
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

# Test the model
def test():
    model.eval()
    all_outputs = []
    all_true_values = []
    test_loss = 0
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
        all_outputs.extend(output.cpu().numpy())
        all_true_values.extend(data.y.cpu().numpy())
        test_loss += criterion(output, data.y).item()
    return test_loss / len(test_loader), all_outputs, all_true_values

# Run the training loop
for epoch in range(250):
    train()
    test_loss, all_outputs, all_true_values = test()
    print(f"Epoch: {epoch+1}, Test loss: {test_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "max_flow_gin.pth")

# Calculate evaluation metrics for the test set
mse = mean_squared_error(all_true_values, all_outputs)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_true_values, all_outputs)
r2 = r2_score(all_true_values, all_outputs)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")
