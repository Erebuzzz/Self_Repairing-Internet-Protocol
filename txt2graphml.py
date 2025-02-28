import networkx as nx

def convert_txt_to_graphml(txt_filename, graphml_filename):
    """Convert a TXT edge list to GraphML format."""
    G = nx.read_edgelist(txt_filename, create_using=nx.Graph(), nodetype=int)  
    nx.write_graphml(G, graphml_filename)
    print(f"✅ Successfully converted '{txt_filename}' to '{graphml_filename}'")
    return graphml_filename

# Example usage
graphml_file = convert_txt_to_graphml("as19981231.txt", "as-733_19981231.graphml")