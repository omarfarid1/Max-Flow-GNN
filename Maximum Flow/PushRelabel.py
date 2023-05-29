import pickle
import networkx as nx
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def push_relabel(graph, source, sink):
    flow_network = nx.DiGraph(graph)
    max_flow = nx.maximum_flow_value(flow_network, source, sink, flow_func=nx.algorithms.flow.preflow_push)
    return max_flow

# Load dataset
with open('graphs.pkl', 'rb') as f:
    graphs = pickle.load(f)
with open('max_flows.pkl', 'rb') as f:
    true_max_flows = pickle.load(f)
with open('sources_sinks.pkl', 'rb') as f:
    sources_sinks = pickle.load(f)

start_time = time.time()

# Calculate max flows using Push-Relabel algorithm
predicted_max_flows = [push_relabel(graph, source, sink) for (graph, (source, sink)) in zip(graphs, sources_sinks)]

end_time = time.time()
elapsed_time = end_time - start_time

# Calculate max flows using Push-Relabel algorithm
predicted_max_flows = [push_relabel(graph, source, sink) for (graph, (source, sink)) in zip(graphs, sources_sinks)]

# Evaluation
mse = mean_squared_error(true_max_flows, predicted_max_flows)
rmse = np.sqrt(mse)
mae = mean_absolute_error(true_max_flows, predicted_max_flows)
r2 = r2_score(true_max_flows, predicted_max_flows)

print("Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print("Execution time:", elapsed_time, "seconds")
