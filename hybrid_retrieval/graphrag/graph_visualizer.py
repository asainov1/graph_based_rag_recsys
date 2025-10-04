from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import os

class GraphVisualizer:
    def __init__(self):
        """Initialize graph visualizer"""
        pass
    
    def visualize_interactive(self, G, output_file="graph.html", height="750px", width="100%", subset=None):
        """Create an interactive visualization of the graph"""
        # If subset is specified, take a subgraph
        if subset is not None:
            if isinstance(subset, int):
                # Take an ego graph around a random node
                center_node = list(G.nodes())[0] if len(G.nodes()) > 0 else None
                if center_node:
                    G_vis = nx.ego_graph(G, center_node, radius=subset)
                else:
                    G_vis = G
            else:
                # Use the provided subgraph
                G_vis = subset
        else:
            # If the graph is too large, take a smaller subgraph
            if len(G.nodes()) > 100:
                print(f"Graph is large ({len(G.nodes())} nodes). Taking a subset for visualization.")
                # Find the most connected node
                degrees = dict(G.degree())
                center_node = max(degrees, key=degrees.get)
                G_vis = nx.ego_graph(G, center_node, radius=2)
            else:
                G_vis = G
        
        # Create network
        net = Network(height=height, width=width, notebook=True)
        
        # Add nodes with different colors based on type
        for node_id in G_vis.nodes():
            node_data = G_vis.nodes[node_id]
            if node_data.get('type') == 'article':
                net.add_node(node_id, 
                             label=node_data.get('title', node_id), 
                             color='blue',
                             title=f"Article: {node_data.get('title', '')}\nLanguage: {node_data.get('language', '')}")
            else:  # entity node
                color_map = {
                    'PERSON': 'red',
                    'ORG': 'green',
                    'GPE': 'orange',  # Countries, cities, states
                    'LOC': 'purple',  # Non-GPE locations
                    'DATE': 'brown',
                    'EVENT': 'pink'
                }
                entity_type = node_data.get('label', 'MISC')
                color = color_map.get(entity_type, 'gray')
                net.add_node(node_id, 
                             label=node_data.get('text', node_id), 
                             title=f"Type: {entity_type}", 
                             color=color)
        
        # Add edges
        for source, target, data in G_vis.edges(data=True):
            width = data.get('weight', 1)
            relationship = data.get('relationship', '')
            net.add_edge(source, target, 
                         width=width, 
                         title=f"{relationship} (weight: {width})")
        
        # Save and show the graph
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        net.save_graph(output_file)
        
        return output_file
    
    def visualize_static(self, G, output_file="graph.png", figsize=(12, 8), subset=None):
        """Create a static visualization of the graph"""
        # Handle large graphs
        if subset is not None:
            if isinstance(subset, int):
                # Take an ego graph around a random node
                center_node = list(G.nodes())[0] if len(G.nodes()) > 0 else None
                if center_node:
                    G_vis = nx.ego_graph(G, center_node, radius=subset)
                else:
                    G_vis = G
            else:
                # Use the provided subgraph
                G_vis = subset
        else:
            # If the graph is too large, take a smaller subgraph
            if len(G.nodes()) > 50:
                print(f"Graph is large ({len(G.nodes())} nodes). Taking a subset for visualization.")
                # Find the most connected node
                degrees = dict(G.degree())
                center_node = max(degrees, key=degrees.get)
                G_vis = nx.ego_graph(G, center_node, radius=1)
            else:
                G_vis = G
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Node colors
        node_colors = []
        for node in G_vis.nodes():
            node_type = G_vis.nodes[node].get('type')
            if node_type == 'article':
                node_colors.append('skyblue')
            else:
                entity_type = G_vis.nodes[node].get('label', 'MISC')
                color_map = {
                    'PERSON': 'red',
                    'ORG': 'green',
                    'GPE': 'orange',
                    'LOC': 'purple',
                    'DATE': 'brown',
                    'EVENT': 'pink'
                }
                node_colors.append(color_map.get(entity_type, 'gray'))
        
        # Node sizes
        node_sizes = []
        for node in G_vis.nodes():
            if G_vis.nodes[node].get('type') == 'article':
                node_sizes.append(300)
            else:
                node_sizes.append(100)
        
        # Edge weights
        edge_weights = [G_vis[u][v].get('weight', 1) for u, v in G_vis.edges()]
        
        # Draw the graph
        pos = nx.spring_layout(G_vis, seed=42)
        nx.draw_networkx_nodes(G_vis, pos, node_color=node_colors, node_size=node_sizes)
        nx.draw_networkx_edges(G_vis, pos, width=edge_weights, alpha=0.7)
        
        # Add labels to nodes
        labels = {}
        for node in G_vis.nodes():
            if G_vis.nodes[node].get('type') == 'article':
                labels[node] = G_vis.nodes[node].get('title', node)[:15]  # Truncate long titles
            else:
                labels[node] = G_vis.nodes[node].get('text', node)[:15]  # Truncate long entity names
        
        nx.draw_networkx_labels(G_vis, pos, labels=labels, font_size=8)
        
        # Save figure
        plt.axis('off')
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    