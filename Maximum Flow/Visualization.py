import networkx as nx
import matplotlib.pyplot as plt
import pickle


# Load the graph dataset
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


graphs = load_pickle('graphs.pkl')


def visualize_graph(index):
    if 0 <= index < len(graphs):
        G = graphs[index]

        # Increase the 'k' parameter to create more space between nodes
        pos = nx.spring_layout(G, k=2)

        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500)

        # Add edge labels with capacities
        edge_labels = {(u, v): G[u][v]['capacity'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.show()
    else:
        print("Invalid index. Please enter an index between 0 and", len(graphs) - 1)


# Visualize the graph at index 0 with edge capacities and increased node spacing
visualize_graph(0)