import tkinter as tk
from tkinter import ttk
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GINConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define the MaxFlowGIN class
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
def convert_to_pyg_format(graph):
    # Assign a dummy feature vector to each node
    for node in graph.nodes():
        graph.nodes[node]['x'] = [1.0]

    # Convert the 'capacity' edge attribute to 'edge_attr'
    for edge in graph.edges():
        graph.edges[edge]['edge_attr'] = [graph.edges[edge]['capacity']]

    data = from_networkx(graph)
    return data

data_list = [convert_to_pyg_format(graph) for graph in graphs]
# Load the GNN model
def load_gnn_model(model_path):
    model = MaxFlowGIN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model_path = 'max_flow_gin.pth'
model = load_gnn_model(model_path)

# Predict the maximum flow using the GNN model
def predict_max_flow(graph_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    graph_data = graph_data.to(device)
    with torch.no_grad():
        output = model(graph_data)
    return output.item()

# Visualize the graph
def visualize_graph(graph, frame):
    if graph is not None:
        G = graph

        fig = plt.Figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        pos = nx.spring_layout(G, k=2)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, ax=ax)

        edge_labels = {(u, v): G[u][v]['capacity'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

# Create the main application window
window = tk.Tk()
window.title("Maximum Flow")

# Create a frame to hold the input elements
input_frame = ttk.Frame(window)
input_frame.grid(row=0, column=0)

# Create a label and entry for the graph index
graph_index_label = ttk.Label(input_frame, text="Enter graph index:")
graph_index_label.grid(row=0, column=0)
graph_index_entry = ttk.Entry(input_frame)
graph_index_entry.grid(row=0, column=1)

# Create a function to handle the visualize button click
# Create a function to handle the visualize button click
def visualize_button_click():
    try:
        graph_index = int(graph_index_entry.get())

        if 0 <= graph_index < len(graphs):
            graph_data = data_list[graph_index]
            true_max_flow = max_flows[graph_index]
            predicted_max_flow = predict_max_flow(graph_data)
            result_text.set(f"True max flow: {true_max_flow:.2f}\nPredicted max flow: {predicted_max_flow:.2f}")

            # Clear the previous graph visualization
            for widget in graph_frame.winfo_children():
                widget.destroy()

            # Display the new graph visualization
            visualize_graph(graphs[graph_index], graph_frame)
        else:
            result_text.set(f"Invalid index. Please enter a number between 0 and {len(graphs) - 1}.")

    except ValueError:
        result_text.set("Invalid input. Please enter a valid number.")

        # Display the new graph visualization
        visualize_graph(graphs[graph_index], graph_frame)
    except ValueError:
        result_text.set("Invalid index. Please enter a valid graph index.")

# Create a visualize button
visualize_button = ttk.Button(input_frame, text="Visualize", command=visualize_button_click)
visualize_button.grid(row=0, column=2)

# Create a label to display the results
result_text = tk.StringVar()
result_label = ttk.Label(window, textvariable=result_text)
result_label.grid(row=1, column=0)

# Create a frame to hold the graph visualization
graph_frame = ttk.Frame(window)
graph_frame.grid(row=2, column=0)

# Start the main application loop
window.mainloop()
