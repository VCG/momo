import importlib.metadata
import pathlib
import anywidget
import traitlets
import arkouda as ak
import arachne as ar
import numpy as np

try:
    __version__ = importlib.metadata.version("anywidget_test")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

def get_mapping(g, subgraph):
    src_sub, dst_sub = subgraph.edges()
    src_sub = src_sub.to_ndarray()
    dst_sub = dst_sub.to_ndarray()
    isos = ar.subgraph_isomorphism(g, subgraph, semantic_check= True)
    isos_ndarray = isos.to_ndarray()  

    # Check if the length of isomorphisms is a multiple of the number of subgraph nodes
    if len(isos) % len(subgraph) != 0:
        raise ValueError("The length of isomorphisms is not a multiple of the number of subgraph nodes.")

    subgraph_nodes = sorted(list(np.unique(np.concatenate((src_sub, dst_sub)))))
    number_isos_found = len(isos) // len(subgraph_nodes)

    # Prepare the hostgraph_nodes as a 2D array
    hostgraph_nodes = isos_ndarray.reshape(-1, len(subgraph_nodes))

    # Create all mappings at once using a list comprehension
    all_mappings = [
        dict(zip(subgraph_nodes, hostgraph_nodes[i]))
        for i in range(number_isos_found)
    ]

    # print(f"Number of Mappings found: {number_isos_found}")
    return all_mappings

def test_graph_creation(edges):
    first_elements = edges[:, 0]
    second_elements = edges[:, 1]
    neighbor_matrix = (first_elements[:, np.newaxis] == second_elements[np.newaxis, :])

    # Step 3: Extract neighbor pairs
    neighbor_indices = np.argwhere(neighbor_matrix)
    neighbor_indices = neighbor_indices[neighbor_indices[:, 0] != neighbor_indices[:, 1]]
    neighbor_pairs = edges[neighbor_indices]

    # Step 4: Combine node indices to create unique identifiers for edges
    src_combined = neighbor_pairs[:, 0, 0] * 10000 + neighbor_pairs[:, 0, 1]
    dst_combined = neighbor_pairs[:, 1, 0] * 10000 + neighbor_pairs[:, 1, 1]

    # Convert to lists for Arkouda
    src = src_combined.astype(np.int64).tolist()
    dst = dst_combined.astype(np.int64).tolist()
    return src, dst

def drawing_transformation(branches, synapses, color):
    nodes=[]
    branch_type= []

    for b in branches:
        nodes.append(b[0] * 10000 + b[1])

    color_mapping = dict(zip(nodes, color))

    src, dst = test_graph_creation(branches)
    for s in src:
        branch_type.append("n")

    for s in synapses:
        src.append(s[0][0] * 10000 + s[0][1])
        dst.append(s[1][0] * 10000 + s[1][1])
        branch_type.append("s")
    
    return src, dst, branch_type, color_mapping

def motif_to_vis(d, motif):
    ak.connect()

    g = ar.PropGraph()
    g.load_edge_attributes(d, source_column="src", destination_column="dst", 
                               relationship_columns=["s_bef", "s_bef_x", "s_bef_y", "s_bef_z", "s_af", "s_af_x", 
                                                     "s_af_y", "s_af_z", 's_x', "s_y", "s_z", "s_distance", "d_bef",
                                                     "d_bef_x", "d_bef_y", "d_bef_z", "d_af", "d_af_x", "d_af_y", "d_af_z",
                                                       "d_x", "d_y", "d_z", "d_distance", "n_id", "connection_type"])
    
    branches, synapses, color = [], [], []

    for m in motif:
        if m["properties"][0] == "neuron connection":
            branches.append(m["label"])
            color.append(m["properties"][1])
            
        if m["properties"][0] == "synaptic connection":
            synapses.append([m["properties"][1], m["properties"][2]])
    
    branches = np.array(branches)
    synapses = np.array(synapses)

    src, dst, branch_type, color_mapping = drawing_transformation(branches, synapses, color)

    subgraph_dict = {
        "src": src,
        "dst": dst,
        "connection_type": branch_type
        }
    
    subgraph = ar.PropGraph()
    df = ak.DataFrame(subgraph_dict)
    subgraph.load_edge_attributes(df, source_column="src", destination_column="dst", 
                                relationship_columns=["connection_type"])
    
    node_mapping = get_mapping(g, subgraph)

    if len(node_mapping) > 6 :
        node_mapping = node_mapping[:5]
    
    nodeid_color_mapping=[]
    for element in node_mapping:
        neuron_ids=[]
        node_colors=[]
        for index in element:
            temp_df = d[ (d["src"] == element[index]) & (d["connection_type"] == "n")]
            
            if len(temp_df) == 0:
                neuron_id = d[ (d["dst"] == element[index]) & (d["connection_type"] == "n")][0]["n_id"]
                if neuron_id not in neuron_ids:
                    neuron_ids.append(neuron_id)
                    node_colors.append(color_mapping[index])
    
            else:
                neuron_id = temp_df[0]["n_id"]
                if neuron_id not in neuron_ids:
                    neuron_ids.append(neuron_id)
                    node_colors.append(color_mapping[index])
        nodeid_color_mapping.append(dict(zip(neuron_ids,node_colors)))
    
    return node_mapping, nodeid_color_mapping

class Widget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    value = traitlets.Int(0).tag(sync=True)

    # Create a traitlet to hold the motifJson data
    motif_json = traitlets.List([]).tag(sync=True)
    node_mapping = traitlets.List([]).tag(sync=True)
    current_mapping = traitlets.Dict({}).tag(sync=True)

    nodeid_color_mapping = traitlets.List([]).tag(sync=True)
    selectedIndex = traitlets.Int(-1).tag(sync=True)


    def __init__(self, arkouda_df=None, **kwargs):
        super().__init__(**kwargs)
        self.arkouda_df = arkouda_df
        self.observe(self.on_motif_json_change, names="motif_json")
        self.observe(self.on_current_motif_change, names="current_motif")
        self.observe(self.on_selectedIndex_change, names="selectedIndex")


    def on_motif_json_change(self, change):
        motif = change['new'] 
        self.node_mapping, self.nodeid_color_mapping =  motif_to_vis(self.arkouda_df, motif)

    def on_current_motif_change(self, change):
        self.current_mapping = change['new']

    def on_selectedIndex_change(self, change):
        self.selectedIndex = change['new']

