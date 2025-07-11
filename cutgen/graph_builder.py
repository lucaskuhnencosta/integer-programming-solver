from itertools import combinations
import networkx as nx

def graph_builder(S):
    G = nx.Graph()
    """This function takes a set of cliques and returns G, the graph. It doesnt plot the graph"""
    for clique in S:
        for node in clique:
            G.add_node(node)
        for u,v in combinations(clique,2):
            G.add_edge(u,v)
    return G

