import matplotlib.pyplot as plt
import networkx as nx
import itertools
from aux import get_node_label
from matplotlib.patches import Patch

def relabel_graph_with_labels(G,num_orders,num_aisles):
    mapping={
        idx: get_node_label(idx,num_orders,num_aisles)
        for idx in G.nodes()
    }
    return nx.relabel_nodes(G,mapping)

def plot_expression_graph(G,
                          num_orders,
                          num_aisles,
                          cliques=None,
                          title="Conflict Graph",
                          graph_layout="kamada_kawai",
                          figsize=(20,20),
                          nodesize=300,
                          filename=None):
    """This function plots the conflict graph with all its nodes and connects only the connected"""

    G_labeled=relabel_graph_with_labels(G,num_orders,num_aisles)

    layout_options={
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "shell": nx.shell_layout,
        "circular" : nx.circular_layout,
    }

    layout_func = layout_options.get(graph_layout)
    pos = layout_func(G_labeled, seed=42) if graph_layout in ['spring', 'random'] else layout_func(G_labeled)

    plt.figure(figsize=(figsize, figsize),dpi=300)

    node_colors={}
    legend_patches=[]

    if cliques:
        clique_labels=[f'Clique{i+1}' for i in range(len(cliques))]
        color_cycle=itertools.cycle(plt.cm.get_cmap("tab20").colors)
        for i,clique in enumerate(cliques):
            color=next(color_cycle)
            labeled_clique = [get_node_label(idx, num_orders, num_aisles) for idx in clique]
            for node in labeled_clique:
                node_colors[node]=color
            legend_patches.append(Patch(color=color, label=clique_labels[i]))

    for node in G_labeled.nodes:
        color=node_colors.get(node,'gray')
        nx.draw_networkx_nodes(G_labeled,
                               pos=pos,
                               nodelist=[node],
                               node_color=[color],
                               node_size=nodesize,
                               alpha=1)

    nx.draw_networkx_edges(G_labeled, pos, alpha=0.8,width=0.8)
    nx.draw_networkx_labels(G_labeled, pos, font_size=8,font_color="black")

    plt.title(title)
    if legend_patches:
        plt.legend(handles=legend_patches,loc='upper right',fontsize=5)
    plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename,bbox_inches='tight')
        print(f'saved plot to: {filename}')
        plt.close()
    plt.show()

