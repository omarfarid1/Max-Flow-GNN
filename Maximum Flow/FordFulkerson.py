import pickle
import numpy as np
import networkx as nx
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)

    def BFS(self, s, t, parent):
        visited = [False] * (self.ROW)
        queue = []
        queue.append(s)
        visited[s] = True

        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        return False

    def FordFulkerson(self, source, sink):
        parent = [-1] * (self.ROW)
        max_flow = 0

        while self.BFS(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            max_flow += path_flow
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
        return max_flow

# Load the generated graphs, their maximum flows, and source and sink nodes from the files
with open("graphs.pkl", "rb") as f:
    graphs = pickle.load(f)

with open("max_flows.pkl", "rb") as f:
    max_flows = pickle.load(f)

with open("sources_sinks.pkl", "rb") as f:
    sources_sinks = pickle.load(f)

# Run the Ford Fulkerson algorithm on each graph in the dataset and store the predicted maximum flows
predicted_max_flows = []
for i, G in enumerate(graphs):
    source, sink = sources_sinks[i]
    adj_matrix = nx.to_numpy_array(G, dtype=int)
    g = Graph(adj_matrix)
    predicted_flow = g.FordFulkerson(source, sink)
    predicted_max_flows.append(predicted_flow)

    # Compare the predicted maximum flow and the true maximum flow
    true_flow = max_flows[i]
    if predicted_flow != true_flow:
        print(f"Graph {i}: Predicted maximum flow = {predicted_flow}, True maximum flow = {true_flow}")

# Calculate the evaluation metrics
mse = mean_squared_error(max_flows, predicted_max_flows)
rmse = np.sqrt(mse)
mae = mean_absolute_error(max_flows, predicted_max_flows)
r2 = r2_score(max_flows, predicted_max_flows)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")