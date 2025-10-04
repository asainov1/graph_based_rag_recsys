import pickle
from graph_visualizer import GraphVisualizer

def test_visualizer():
    # Load the knowledge graph
    print("Loading knowledge graph...")
    try:
        with open("graph.pickle", "rb") as f:
            graph = pickle.load(f)
        print(f"Loaded graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return
    
    # Create visualizer
    visualizer = GraphVisualizer()
    
    # Generate visualization
    output_file = visualizer.visualize_interactive(graph, output_file="knowledge_graph.html")
    
    print(f"Interactive visualization saved to {output_file}")
    print(f"Open this file in a web browser to explore the graph")

if __name__ == "__main__":
    test_visualizer() 