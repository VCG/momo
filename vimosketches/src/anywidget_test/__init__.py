import importlib.metadata
import pathlib

import anywidget
import traitlets
import arkouda as ak
import arachne as ar
import numpy as np
import pandas as pd
import networkx as nx
from compcon.create_graph import get_neuron_local

try:
    __version__ = importlib.metadata.version("anywidget_test")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

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

def findd(g, subgraph):
    # ak.connect()
    # find_isomorphic_subgraphs(g, subgraph)
    src_sub, dst_sub = subgraph.edges()
    src_sub = src_sub.to_ndarray()
    dst_sub = dst_sub.to_ndarray()
    isos = ar.subgraph_isomorphism(g, subgraph, semantic_check= True)
    # print(f"Isomorphisms found: {isos}")
    isos_ndarray = isos.to_ndarray()  # Convert pdarray to ndarray

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

def motif_to_vis(d, motif):
    ak.connect()
    branches, synapses, color = [], [], []
    for m in motif:
        if m["properties"][0] == "neuron connection":
            branches.append(m["label"])
            color.append(m["properties"][1])
            
        if m["properties"][0] == "synaptic connection":
            synapses.append([m["properties"][1], m["properties"][2]])
        # print(type((m["properties"][0])))
    
    branches = np.array(branches)
    synapses = np.array(synapses)

    src, dst, branch_type, color_mapping = drawing_transformation(branches, synapses, color)

    dicts = {
        "src": src,
        "dst": dst,
        "connection_type": branch_type
        }
    
    g = ar.PropGraph()
    # spatial_connectome_edge_df = ak.DataFrame(d.to_dict(orient='list'))
    # spatial_connectome_edge_df = ak.DataFrame(d)
    g.load_edge_attributes(d, source_column="src", destination_column="dst", 
                               relationship_columns=["s_bef", "s_bef_x", "s_bef_y", "s_bef_z", "s_af", "s_af_x", 
                                                     "s_af_y", "s_af_z", 's_x', "s_y", "s_z", "s_distance", "d_bef",
                                                     "d_bef_x", "d_bef_y", "d_bef_z", "d_af", "d_af_x", "d_af_y", "d_af_z",
                                                       "d_x", "d_y", "d_z", "d_distance", "n_id", "connection_type"])
    
    
    subgraph = ar.PropGraph()
    df = ak.DataFrame(dicts)
    subgraph.load_edge_attributes(df, source_column="src", destination_column="dst", 
                                relationship_columns=["connection_type"])
    
    m = findd(g, subgraph)
    if len(m) > 6 :
        m = m[:5]
    

    new_list=[]
    for i in m:
        l1=[]
        l2=[]
        for val in i:
            temp = d[ (d["src"] == i[val]) & (d["connection_type"] == "n")]
            
            if len(temp) == 0:
                temp = d[ (d["dst"] == i[val]) & (d["connection_type"] == "n")][0]["n_id"]
                if temp not in l1:
                    l1.append(temp)
                    l2.append(color_mapping[val])
    
            else:
                # print(temp)
                temp = temp[0]["n_id"]
                if temp not in l1:
                    l1.append(temp)
                    l2.append(color_mapping[val])
        new_list.append(dict(zip(l1,l2)))
    
    # new_list
    return m, color_mapping, new_list

def mapping_edges(subgraph, mapping):
    edges = subgraph.edges()
    src = edges[0].to_ndarray()
    dst = edges[1].to_ndarray()

    src = src.astype(str)
    dst = dst.astype(str)

    src_mapped = np.vectorize(mapping.get)(src)
    dst_mapped = np.vectorize(mapping.get)(dst)

    return src_mapped, dst_mapped

def back_transformation_edges(d, src_mapped, dst_mapped):
    new_edges = []
    n_ids = []
    unique_offset = 0
    for i in range(len(src_mapped)):
        t = d[(d['src'] == src_mapped[i]) & (d['dst'] == dst_mapped[i])][0]
        if t["connection_type"] == 'n':
            new_element = [t["s_bef"], t["s_af"]]
            # wenn edges in verschiedenen neurons gleiche ids haben geht das nicht!!(check if n_ids übereinstimmen!)
            if new_element not in new_edges:
                new_edges.append(new_element)
                n_ids.append(int(t["n_id"]))

            # new_edges_temp = new_edges
            # if (len(np.unique(new_edges_temp, axis=0).tolist()) + unique_offset) != len(new_edges_temp):
            #     unique_offset +=1
            # else:
            #     n_ids.append(int(t["n_id"]))

            new_element = [t["d_bef"], t["d_af"]]
            if new_element not in new_edges:
                new_edges.append(new_element)
                n_ids.append(int(t["n_id"]))

            # new_edges_temp = new_edges
            # if (len(np.unique(new_edges_temp, axis=0).tolist()) + unique_offset) != len(new_edges_temp):
            #     unique_offset +=1
            # else:
            #     n_ids.append(int(t["n_id"]))


    # seen = set()
    # seen_add = seen.add
    # new_edges = [x for x in new_edges if not (x in seen or seen_add(x))]
    # new_edges = np.unique(new_edges, axis=0).tolist()
    return new_edges, n_ids

def get_slabs(new_edges, n_ids):
    paths = []
    unique_n_ids = set(n_ids)
    short_ids = []
    nds_neurons = []

    for index, id in enumerate(unique_n_ids):
        n, c = get_neuron_local(id, 3000)
        nx_display = nx.Graph()
        nx_display.add_edges_from(n.edges)
        shortest_path=[]
        temp_paths=[]
        for edge_index, edge in enumerate(new_edges):
            if n_ids[edge_index]==id:
                start_node_id = new_edges[edge_index][0]
                end_node_id = new_edges[edge_index][1]
                try:
                    shortest_path = nx.shortest_path(nx_display, source=start_node_id, target=end_node_id, weight='weight')
                    
                except nx.NetworkXNoPath:
                    print(f"No path found between and")

                # shortest_path = np.stack([shortest_path[:-1], shortest_path[1:]], axis=1).tolist()
                temp_paths.append(shortest_path)
                short_ids.extend([id])
        nds_neurons.append(n)
        paths.append(temp_paths)
    return paths, short_ids, nds_neurons

def get_segment_idx_nds(final_path, paths): 
    # offset = 0
    # neuron3d = []
    # testlist = []

    # for neuron in nds_neurons:
    #     for segment_idx, segment in enumerate(neuron.edges):
    #         for index, path in enumerate(paths):
    #             if int(neuron.id) == short_ids[index]:
    #                 for element in path:
    #                     if (segment[0] == element[0] and segment[1] == element[1]):
    #                         testlist.append(offset + segment_idx)
    #     offset += len(neuron.edges)

    # return testlist
    offset = 0
    testlist = []
    for path in final_path:
        for segment_idx, segment in enumerate(path):
            for hh in paths:
                for h in hh:
                    if segment == h:
                        testlist.append(offset + segment_idx)
                        # print(h)
        offset += len(path)
    return testlist

def normalize_column(column):
    return (column) / (1000)

def normalize_neuron_list(neuron3d):
    neuron_nodes = []
    neuron_edges = []
    neuron_segments = []
    for n in neuron3d:
        n.nodes['x'] = normalize_column(n.nodes['x'])
        n.nodes['y'] = normalize_column(n.nodes['y'])
        n.nodes['z'] = normalize_column(n.nodes['z'])

        n.connectors['x'] = normalize_column(n.connectors['x'])
        n.connectors['y'] = normalize_column(n.connectors['y'])
        n.connectors['z'] = normalize_column(n.connectors['z'])

        n.nodes.radius = 0.1
        neuron_nodes.append(n.nodes)
        neuron_edges.append(n.edges)
        neuron_segments.append(n.segments)
    
    return neuron_nodes, neuron_edges, neuron_segments

def get_path(neuron_segments, paths):
    final_path=[]
    for index, path in enumerate(paths):
        temp_path = []
        temp_path = idk2(neuron_segments[index], path)
        final_path.append(temp_path)

    return final_path

def remove_subsequence_if_exists(list1, list2):
    # Find the starting index of list2 in list1
    n, m = len(list1), len(list2)
    
    # Iterate over list1 to find a matching subsequence
    for i in range(n - m + 1):
        if list1[i:i + m] == list2:  # Check if list2 matches list1 starting at index i
            # print(list1)
            # print(list2)
            if i==0:
                # print(list1[i + m-1:])
                return list1[i + m-1:]
            
            
            if (i+m) == len(list1):
                # print(list1[:i+1])
                return list1[:i+1]
            # print(list1[:i+1], list1[i + m-1:])
            return list1[:i+1], list1[i + m-1:]  # Remove the subsequence
    
    # If no matching subsequence is found, return the original list
    return list1

def idk2(list1, list2):
    aha = []
    for l2 in list2:
        for l1 in list1:
            result = remove_subsequence_if_exists(l1, l2)
            if type(result) == tuple:
                #FIX THE PROBLEM
                list1[list1.index(l1)] = result[1]
                aha.append(result[0])
            # print(list1.index(l1), result)
            else:
                list1[list1.index(l1)] = result
            aha
            
            # print(list1)
            
    return list1+aha+list2 

def overfinal(d, motif, mapping):
    ak.connect()
    branches, synapses = [], []
    for m in motif:
        if m["properties"][0] == "neuron connection":
            branches.append(m["label"])
            
        if m["properties"][0] == "synaptic connection":
            synapses.append([m["properties"][1], m["properties"][2]])
        # print(type((m["properties"][0])))
    
    branches = np.array(branches)
    synapses = np.array(synapses)

    src, dst, branch_type = drawing_transformation(branches, synapses)
    dicts = {
        "src": src,
        "dst": dst,
        "connection_type": branch_type
        }
    subgraph = ar.PropGraph()
    df = ak.DataFrame(dicts)
    subgraph.load_edge_attributes(df, source_column="src", destination_column="dst", 
                                relationship_columns=["connection_type"])
    src_mapped, dst_mapped = mapping_edges(subgraph, mapping)
    
    new_edges, n_ids = back_transformation_edges(d, src_mapped, dst_mapped)
    
    paths, short_ids, nds_neurons = get_slabs(new_edges, n_ids)
    
    neuron_nodes, neuron_edges, neuron_segments = normalize_neuron_list(nds_neurons)
    
    final_path = get_path(neuron_segments, paths)
    
    final_path = [[j for j in i if len(j) >= 2] for i in final_path]
    
    testlist = get_segment_idx_nds(final_path, paths)
    ak.disconnect()

    return neuron_nodes, final_path, testlist

class Widget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    value = traitlets.Int(0).tag(sync=True)

    # Create a traitlet to hold the motifJson data
    motif_json = traitlets.List([]).tag(sync=True)
    m = traitlets.List([]).tag(sync=True)
    currentM = traitlets.Dict({}).tag(sync=True)
    color_mapping = traitlets.Dict({}).tag(sync=True)

    nnodes = traitlets.List([]).tag(sync=True)
    finalpath = traitlets.List([]).tag(sync=True)
    testlist = traitlets.List([]).tag(sync=True)
    new_list = traitlets.List([]).tag(sync=True)


    def __init__(self, arkouda_df=None, **kwargs):
        super().__init__(**kwargs)
        self.arkouda_df = arkouda_df
        self.observe(self.on_motif_json_change, names="motif_json")
        self.observe(self.on_currentM_change, names="currentM")


    def on_motif_json_change(self, change):
        motif = change['new'] 
        self.m, self.color_mapping, self.new_list =  motif_to_vis(self.arkouda_df, motif)
        # self.m = result  

    def on_currentM_change(self, change):
        current_mapping = change['new']

