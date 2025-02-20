import importlib.metadata
import pathlib
import anywidget
import traitlets
import sys
import os
sys.path.insert(0, os.path.abspath("/home/michaelshewarega/Desktop/test/arkouda"))
import arkouda as ak
import arachne as ar
import numpy as np
import time

try:
    __version__ = importlib.metadata.version("anywidget_test")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

def get_post_processed_mappings(g, subgraph_src, subgraph_dst, color, mapping):
    start = time.time()
    dict_to_check_against = {}
    for c in color:
        if c not in dict_to_check_against:
            dict_to_check_against[c] = 1
        else:
            dict_to_check_against[c] += 1

    isos_by_vertices = mapping[0]
    isos_by_edges_src = mapping[1][0]
    isos_by_edges_dst = mapping[1][1]
    
    num_edges_subgraph = len(subgraph_src)
    number_isos_found = len(isos_by_edges_src) // len(subgraph_src)

    indices = ak.find([isos_by_edges_src,isos_by_edges_dst],[g.edge_attributes["src"], g.edge_attributes["dst"]])
    vals = g.edge_attributes["n_id"][indices]

    # TODO: Can the below be done with Arkouda?
    isos_by_edges_src_ndarray = np.split(isos_by_edges_src.to_ndarray(), number_isos_found)
    isos_by_edges_dst_ndarray = np.split(isos_by_edges_dst.to_ndarray(), number_isos_found)
    vals_ndarray = np.split(vals.to_ndarray(), number_isos_found)

    matches = 0
    curr_mapping_id = 0
    final_mappings = []
    for src,dst,vals in zip(isos_by_edges_src_ndarray,isos_by_edges_dst_ndarray,vals_ndarray):
        inner_matches = {}
        for val in vals:
            if val not in inner_matches:
                inner_matches[val] = 1
            else:
                inner_matches[val] += 1
        if sorted(inner_matches.values()) == sorted(dict_to_check_against.values()):
            final_mappings.append(isos_by_vertices[curr_mapping_id])
            matches += 1
        curr_mapping_id += 1

    end = time.time()
    print(f"Before post processing number of motifs found was {number_isos_found} and after was {matches}")
    print(f"Post processing took: {end-start} seconds.")

    return final_mappings

def get_mapping(g, subgraph, iso_cap):
    # Process subgraph information for mapping after subgraph isomorphism is invoked.
    src_sub, dst_sub = subgraph.edges()
    src_sub = src_sub.to_ndarray()
    dst_sub = dst_sub.to_ndarray()
    subgraph_nodes = sorted(list(np.unique(np.concatenate((src_sub, dst_sub)))))

    start = time.time()
    if iso_cap > 0 :
        isos = ar.subgraph_isomorphism(g, subgraph, algorithm_type="si", return_isos_as="complete", semantic_check="or", size_limit=iso_cap)
    else:
        isos = ar.subgraph_isomorphism(g, subgraph, algorithm_type="si", return_isos_as="complete", semantic_check="or")
    end = time.time()
    
    # Extract the returned array information from subgraph_isomorphism.
    isos_by_vertices = isos[0]
    isos_by_vertices_map = isos[1]
    isos_by_edges_src = isos[2]
    isos_by_edges_dst = isos[3]

    if len(isos_by_vertices) % len(subgraph) != 0:
        raise ValueError("The length of isomorphisms is not a multiple of the number of subgraph nodes.")

    # Get the number of motifs found.
    number_isos_found = len(isos_by_vertices) // len(subgraph_nodes)
    print(f"Finding {number_isos_found:_} motifs took: {end-start} seconds.")

    # Prepare the returned isomorphisms as a 2D array.
    start = time.time()
    isos_ndarray = isos_by_vertices.to_ndarray()
    hostgraph_nodes = isos_ndarray.reshape(-1, len(subgraph_nodes))
    end = time.time()
    print(f"Reshaping isomorphisms took: {end-start} seconds.")

    # Create all mappings at once using a list comprehension.
    start = time.time()
    all_mappings = [
        {int(k): int(v) for k, v in zip(subgraph_nodes, hostgraph_nodes[i])}
        for i in range(number_isos_found)
    ]
    end = time.time()
    print(f"Generating mappings took: {end-start} seconds.")

    return (all_mappings,(isos_by_edges_src,isos_by_edges_dst))

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
    id_to_col= {}
    con_color=[]

    for b in range(len(branches)):
        nodes.append(branches[b][0] * 10000 + branches[b][1])
        id_to_col[str(branches[b][0] * 10000 + branches[b][1])] = color[b]
            
    color_mapping = dict(zip(nodes, color))

    src, dst = test_graph_creation(branches)
    for s in src:
        branch_type.append("n")
        con_color.append(id_to_col[str(s)]) 

    for s in synapses:
        src.append(s[0][0] * 10000 + s[0][1])
        dst.append(s[1][0] * 10000 + s[1][1])
        branch_type.append("s")
        con_color.append(id_to_col[str(s[0][0] * 10000 + s[0][1])] + "_" + id_to_col[str(s[1][0] * 10000 + s[1][1])])
    
    return src, dst, branch_type, color_mapping, con_color

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

    src, dst, branch_type, color_mapping, con_color = drawing_transformation(branches, synapses, color)
    # return src, dst, branch_type, con_color

    subgraph_dict = {
        "src": dst,
        "dst": src,
        "connection_type": branch_type
        }
    
    subgraph = ar.PropGraph()
    df = ak.DataFrame(subgraph_dict)
    subgraph.load_edge_attributes(df, source_column="src", destination_column="dst", 
                                relationship_columns=["connection_type"])
    cap=1000000
    node_mapping = get_mapping(g, subgraph, cap)
    final_mapping = get_post_processed_mappings(g, dst, src, con_color, node_mapping)
    nodeid_color_mapping=[]
    for element in final_mapping[:30]:
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
    
    return final_mapping[:30], nodeid_color_mapping

class Widget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    value = traitlets.Int(0).tag(sync=True)

    # Create a traitlet to hold the motifJson data
    motif_json = traitlets.List([]).tag(sync=True)
    node_mapping = traitlets.List([]).tag(sync=True)
    current_mapping = traitlets.Dict({}).tag(sync=True)

    nodeid_color_mapping = traitlets.List([]).tag(sync=True)
    current_nodeid_color_mapping = traitlets.Dict({}).tag(sync=True)
    selectedIndex = traitlets.Int(-1).tag(sync=True)


    def __init__(self, arkouda_df=None, **kwargs):
        super().__init__(**kwargs)
        self.arkouda_df = arkouda_df
        self.observe(self.on_motif_json_change, names="motif_json")
        self.observe(self.on_current_motif_change, names="current_motif")
        self.observe(self.on_selectedIndex_change, names="selectedIndex")
        self.observe(self.on_current_nodeid_color_mapping_change, names="current_nodeid_color_mapping")


    def on_motif_json_change(self, change):
        self.motif_json = change['new'] 
        motif = change['new'] 
        self.node_mapping, self.nodeid_color_mapping =  motif_to_vis(self.arkouda_df, motif)

    def on_current_motif_change(self, change):
        self.current_mapping = change['new']

    def on_selectedIndex_change(self, change):
        self.selectedIndex = change['new']

    def on_current_nodeid_color_mapping_change(self, change):
        self.current_nodeid_color_mapping = change['new']

