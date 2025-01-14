import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame(columns=['Source', 'Target', 'Attr1', 'Attr2'])

####################################################################################################################
# CREATE A NETWORK

# Create edges (and then also nodes) from source and target nodes in 2 separate columns
G =nx.from_pandas_edgelist(df=data, source='Source', target='Target', 
                           edge_attr=['Attr1', 'Attr2'], create_using= nx.Graph())

# We can also create a empty graphs and fill them in
G = nx.Graph()
G = nx.DiGraph()
G = nx.MultiGraph()
G = nx.MultiDiGraph()

# Add nodes
G.add_node(1)
G.add_nodes_from(range(2,9))  # add multiple nodes at once

# Add edges (adding an edge automatically adds the nodes)
G.add_edge(1,2)
edges = [(2,3), (1,3), (4,1), (4,5), (5,6), (5,7), (6,7), (7,8), (6,8)]
G.add_edges_from(edges)

# We can also set attributes after creating the graph
nx.set_node_attributes(G, data['Key'].to_dict(), 'Key' )

# We can create a subgraph from an existing graph (a list of nodes you want to keep)

subG = G.subgraph([(1,2), (2, 3)])

#####################################################################################################################
# Accessing information

# Gives a quick overview (gives amount of edges and nodes, might give attributes)
print(G)

# Print the nodes and edges
print(nx.nodes)
print(nx.edges)

# Prints sparsity 
nx.density(G) # L = abs(E)/abs(E_max), E_max = n*(n-1)/2

# Check if connected
nx.is_connected(G)
nx.connected_components(G)
comp = list(nx.connected_components(G))
nr_comp = len(comp)

largest_comp = max(comp, key=len)
percentage_lcc = len(largest_comp)/G.number_of_nodes() * 100

# Shortest path
nx.shortest_path(G, source="Node1", target="Node2")

# Transitivity (global measure, 1-2 and 2-3, how likely that 1-3)
nx.transitivity(G)

# Clustering, similar measure but locally, look up definition
nx.clustering(G, ['Node1', 'Node2'])

# Degrees
degrees = dict(G.degree(G.nodes()))

# Calculates betweenness_centrality (the more shortest path that goes through a node the more important it is)
nx.set_node_attributes(G, nx.betweenness_centrality(G), 'betweenness')

# The same measure but for edges (the more shortest path that goes through an edge the more important it is)
# Girvan Newman
import itertools
from networkx.algorithms.community.centrality import girvan_newman
comp = girvan_newman(G)
it = 0
for communities in itertools.islice(comp, 4):
    it +=1
    print('Iteration', it)
    print(tuple(sorted(c) for c in communities)) 

# Louvain method
from community import community_louvain
partition = community_louvain.best_partition(G)
for n in G.nodes:
    G.nodes[n]["louvain"] = partition[n]

###################################################################################################################
# DRAW NETWORKS

# Create subgraph based on an attribute
rel_edges = []
edges = nx.get_edge_attributes(G, 'attr')
for e, a in edges.items():
    if a == "something":
        rel_edges.append(e)

subG = G.subgraph(rel_edges)

# https://networkx.github.io/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
nx.draw_spring(G, with_labels=True,  alpha = 0.8)

nx.draw_circular(G, with_labels=True)

# Might help when wanting to plot node attributes
pos = nx.spring_layout(G)
ec = nx.draw_networkx(G, pos, nodelist=G.nodes(),
                         node_color=[G.nodes[n]["attr"] for n in G.nodes()], 
                         node_shape = '.', node_size=1200, font_color="white", font_weight="bold")

##################################################################################################################
# Helper functions from exercise

# Helper function for plotting the degree distribution of a Graph
def plot_degree_distribution(G):
    degrees = {}
    for node in G.nodes():
        degree = G.degree(node)
        if degree not in degrees:
            degrees[degree] = 0
        degrees[degree] += 1
    sorted_degree = sorted(degrees.items())
    deg = [k for (k,v) in sorted_degree]
    cnt = [v for (k,v) in sorted_degree]
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')
    plt.title("Degree Distribution")
    plt.ylabel("Frequency")
    plt.xlabel("Degree")
    ax.set_xticks([d+0.05 for d in deg])
    ax.set_xticklabels(deg)


# Helper function for printing various graph properties
def describe_graph(G):
    print(G)
    if nx.is_connected(G):
        print("Avg. Shortest Path Length: %.4f" %nx.average_shortest_path_length(G))
        print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
    else:
        print("Graph is not connected")
        print("Diameter and Avg shortest path length are not defined!")
    print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph
    # #closed-triplets(3*#triangles)/#all-triplets
    print("Global clustering coefficient aka Transitivity: %.4f" %nx.transitivity(G))


# Helper function for visualizing the graph
def visualize_graph(G, with_labels=True, k=None, alpha=1.0, node_shape='o'):
    #nx.draw_spring(G, with_labels=with_labels, alpha = alpha)
    pos = nx.spring_layout(G, k=k)
    if with_labels:
        lab = nx.draw_networkx_labels(G, pos, labels=dict([(n, n) for n in G.nodes()]))
    ec = nx.draw_networkx_edges(G, pos, alpha=alpha)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color='g', node_shape=node_shape)
    plt.axis('off')







