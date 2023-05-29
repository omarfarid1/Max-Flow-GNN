import pickle
import networkx as nx
import random

# Define the number of graphs to generate
num_graphs = 1000

# Define the range of nodes and edges for each graph
min_nodes = 10
max_nodes = 20
min_edges = 20
max_edges = 40

# Define the capacity range for each edge
min_capacity = 1
max_capacity = 1000

# Create an empty list to store the generated graphs, their maximum flows, and source and sink nodes
graphs = []
max_flows = []
sources_sinks = []

def connect_components(G):
    # Connect the components by adding edges between them
    components = list(nx.weakly_connected_components(G))
    while len(components) > 1:
        for i in range(len(components) - 1):
            source_node = random.choice(list(components[i]))
            sink_node = random.choice(list(components[i + 1]))
            capacity = random.randint(min_capacity, max_capacity)
            G.add_edge(source_node, sink_node, capacity=capacity)
        components = list(nx.weakly_connected_components(G))

# Generate the graphs
for i in range(num_graphs):
    # Randomly generate the number of nodes and edges for the graph
    num_nodes = random.randint(min_nodes, max_nodes)
    num_edges = random.randint(min_edges, max_edges)

    # Create an empty graph
    G = nx.DiGraph()

    # Add the nodes to the graph
    for j in range(num_nodes):
        G.add_node(j)

    # Initialize edge_counts dictionary
    edge_counts = {k: 0 for k in range(num_nodes)}

    # Add the edges to the graph
    for j in range(num_edges):
        # Randomly select the source and target nodes for the edge
        source = random.randint(0, num_nodes - 1)
        sink = random.randint(0, num_nodes - 1)

        # Make sure the source and target nodes are not the same
        while source == sink:
            sink = random.randint(0, num_nodes - 1)

        # Randomly generate the capacity for the edge
        capacity = random.randint(min_capacity, max_capacity)

        # Add the edge to the graph with the generated capacity
        G.add_edge(source, sink, capacity=capacity)

        # Update edge_counts dictionary
        edge_counts[source] += 1
        edge_counts[sink] += 1

    # Connect the components to make the graph closed
    connect_components(G)

    # Find the source and sink nodes from the edge_counts dictionary
    source = max(edge_counts, key=edge_counts.get)
    sink = min(edge_counts, key=edge_counts.get)

    # Compute the maximum flow for the graph
    max_flow_value = nx.maximum_flow_value(G, source, sink)

    # Add the graph, its maximum flow, and source and sink nodes to the lists
    graphs.append(G)
    max_flows.append(max_flow_value)
    sources_sinks.append((source, sink))

# Print the generated graphs, their maximum flows, and source and sink nodes
for i in range(num_graphs):
    print(f"Graph {i + 1}:")
    print(graphs[i].edges(data=True))
    print("Maximum flow:", max_flows[i])
    print("Source and sink nodes:", sources_sinks[i])
    print()

# Save the generated graphs, their maximum flows, and source and sink nodes to files
with open("graphs.pkl", "wb") as f:
    pickle.dump(graphs, f)

with open("max_flows.pkl", "wb") as f:
    pickle.dump(max_flows, f)

with open("sources_sinks.pkl", "wb") as f:
    pickle.dump(sources_sinks, f)

print("Done!")