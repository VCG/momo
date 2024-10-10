import arkouda as ak
import arachne as ar
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import numpy as np
import navis
from fafbseg import flywire
from compcon.navis_api import get_flywire_neuron, get_hemibrain_neuron
from compcon.create_graph import get_neuron, get_graph, draw_graph, get_neuron_local, overall, draw_connection, draw3d_graph
from compcon.isomporphic_subgraphs import find_isomorphic_subgraphs
import itertools
from skelescope.skelescope import Skelescope

def normalize_column(column):
    return (column) / (1000)
    
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


def drawing_transformation(branches, synapses):
    nodes=[]
    branch_type= []
    for b in branches:
        nodes.append(b[0] * 10000 + b[1])

    src, dst = test_graph_creation(branches)
    for s in src:
        branch_type.append("n")

    for s in synapses:
        src.append(s[0][0] * 10000 + s[0][1])
        dst.append(s[1][0] * 10000 + s[1][1])
        branch_type.append("s")
    
    return src, dst, branch_type

def draw(src, dst, branch_type):
    # ak.connect()
    nx_display = nx.Graph()
    for (u, v, etype) in zip(src, dst, branch_type):
        nx_display.add_edge(u, v, edge_type=etype)

    plt.figure(figsize=(10, 10))

    pos = nx.kamada_kawai_layout(nx_display)

    edge_colors = ["blue" if nx_display[u][v]['edge_type'] == "n" else "red" if nx_display[u][v]['edge_type'] == "s" else "black" for u, v in nx_display.edges()]

    nx.draw_networkx(nx_display, pos, with_labels=True, node_size=750, edge_color=edge_colors)
    plt.show()
    # ak.disconnect()


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


def mapping_edges(subgraph, mapping):
    edges = subgraph.edges()
    src = edges[0].to_ndarray()
    dst = edges[1].to_ndarray()

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
            
def get_segment_idx(new_edges, n_ids):
    unique_n_ids = set(n_ids)
    offset = 0
    neuron3d = []
    testlist = []
    for i in unique_n_ids:
        n, c = get_neuron_local(i, 3000, 1000)
        for segment_idx, segment in enumerate(n.edges):
            for edge_id, edge in enumerate(new_edges):
                if n_ids[edge_id] == i:
                    if (segment[0] == edge[0] and segment[1] == edge[1]):
                        testlist.append(offset + segment_idx)
        neuron3d.append(n)  
        offset += len(n.edges)
    return testlist, neuron3d

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

# Example usage:
list1 = [[1, 2, 3, 4, 5], [3,6,7]]
list2 = [[2, 3], [3,4]]
# list2 = [[2, 3], [3,4]]
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

# x1 = idk2(list1, list2)
# x2 = idk2(list1, list2)
# paths = [x1] + [x2]
# paths

def get_path(neuron_segments, paths):
    final_path=[]
    for index, path in enumerate(paths):
        temp_path = []
        temp_path = idk2(neuron_segments[index], path)
        final_path.append(temp_path)

    return final_path



def motif_to_vis(g, d, n, motif):
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
    
    # ak.connect()
    subgraph = ar.PropGraph()
    df = ak.DataFrame(dicts)
    subgraph.load_edge_attributes(df, source_column="src", destination_column="dst", 
                                relationship_columns=["connection_type"])
    
    m = findd(g, subgraph)
    mapping = m[0]
    
    src_mapped, dst_mapped = mapping_edges(subgraph, mapping)
    # print(src_mapped)
    # print(dst_mapped)
    
    new_edges, n_ids = back_transformation_edges(d, src_mapped, dst_mapped)
    # print(new_edges, n_ids)
    # n_ids = [720575940622927541, 720575940622927541, 720575940622927541, 720575940634524441, 720575940634524441, 720575940634524441]
    paths, short_ids, nds_neurons = get_slabs(new_edges, n_ids)
    
    neuron_nodes, neuron_edges, neuron_segments = normalize_neuron_list(nds_neurons)
    
    final_path = get_path(neuron_segments, paths)
    
    final_path = [[j for j in i if len(j) >= 2] for i in final_path]
    
    testlist = get_segment_idx_nds(final_path, paths)
    
    unique_n_ids = set(n_ids)
    # filtered_rows = neuron3d[0].connectors[neuron3d[0].connectors['partner_id'].isin(unique_n_ids)]
    filtered_rows = nds_neurons[0].connectors[nds_neurons[0].connectors['partner_id'].isin(unique_n_ids)]

    ###SYNAPSES
    synapse_src = df[df["connection_type"] == "s"].src.values.to_ndarray()
    synapse_src = np.vectorize(mapping.get)(synapse_src)
    synapse_dst = df[df["connection_type"] == "s"].dst.values.to_ndarray()
    synapse_dst = np.vectorize(mapping.get)(synapse_dst)
    # synapse_src , synapse_dst

    d_pandas = d.to_pandas()
    synapses = pd.DataFrame({'src': synapse_src, 'dst': synapse_dst})
    
    result = pd.merge(d_pandas, synapses, on=['src', 'dst'], how='inner')
    
    # result
    filtered_rows1 = n[720575940634524441]['c'][n[720575940634524441]['c']['partner_id'] == 720575940622927541]

    filt1 = filtered_rows1[((filtered_rows1["x"] == (754392.0)) & (filtered_rows1["y"] == (222796.0))) | ((filtered_rows1["x"] == (753388.0)) & (filtered_rows1["y"] == (222332.0))) | ((filtered_rows1["x"] == (752600.0)) & (filtered_rows1["y"] == (228184.0)))]
    
    x_temp, y_temp, z_temp = [], [], []
    x_temp.extend(filt1.loc[:, "x"])
    y_temp.extend(filt1.loc[:, "y"])
    z_temp.extend(filt1.loc[:, "z"])
    
    c_coordinates = np.vstack((x_temp, y_temp, z_temp)).T
    
    nt2, ct2 = get_neuron_local(720575940622927541, 3000)
    
    n2_synapse_nodes, dists = nt2.snap(c_coordinates)
    
    filt1['x'] = normalize_column(filt1['x'])
    filt1['y'] = normalize_column(filt1['y'])
    filt1['z'] = normalize_column(filt1['z'])
    # filt1
    ak.disconnect()

    return neuron_nodes, filt1, final_path, testlist