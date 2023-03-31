import json
from networkx.readwrite import json_graph

def export_graph(G):

    # Remove the arrow style attribute from the edges
    # (Causes problems in serialization)
    for u, v, d in G.edges(data=True):
        if "arrowstyle" in d:
            del d["arrowstyle"]

    # Convert the NetworkX graph to a JSON object
    data = json_graph.node_link_data(G)


    # Write the JSON object to a file
    with open("graph.json", "w") as outfile:
        json.dump(data, outfile)
