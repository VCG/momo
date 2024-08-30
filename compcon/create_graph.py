import arkouda as ak
import arachne as ar
from pathlib import Path
import pandas as pd
import numpy as np
import navis
from fafbseg import flywire
from compcon.navis_api import get_flywire_neuron
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import os
pd.options.mode.chained_assignment = None  # default='warn'


# from compcon.mapping import *
def get_neuron_local(id, prune_factor=None, ds_factor=None):
    try:
        # Read the neuron data from the SWC file
        n = navis.read_swc(f'test_folder/sk_lod1_783_healed/{id}.swc')  
        flywire.get_synapses(n, attach=True, materialization=783)
        c = n.connectors

        # Apply pruning if prune_factor is provided
        if prune_factor is not None:
            n_prune = navis.prune_twigs(n, prune_factor)
        else:
            n_prune = n

        # Apply downsampling if ds_factor is provided and not zero
        if ds_factor is not None:
            n_ds = navis.downsample_neuron(n_prune, downsampling_factor=ds_factor, inplace=False)
        else:
            n_ds = n_prune
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        n_ds = None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        n_ds = None

    return n_ds, c



def get_neuron(id,prune_factor, ds_factor):
    repo_folder = Path.cwd()
    data_folder = repo_folder/"data"
    annotations_from_paper_folder = data_folder/"local"/"EM_data"/"Raw-data"
    neurons = pd.read_csv(repo_folder/"flywire_annotations"/"supplemental_files"/"Supplemental_file1_neuron_annotations.tsv", sep="\t", low_memory=False)
    root_ids = set(neurons.root_id.values)
    intermediate_folder = repo_folder/"data"/"intermediate"
    tmp_folder = repo_folder/"data"/"tmp"
    cache_folder = intermediate_folder/"cache"

    neurons_ext = neurons.copy()
    ol_columns = pd.read_csv(data_folder/"local"/"ol_columns.csv")

    for _, row in ol_columns.iterrows():
            root_id = row["cell id"]
            column_descriptor = row["column id"]
            if(column_descriptor != "not assigned"):
                    neurons_ext.loc[neurons_ext.root_id == root_id, "optic_lobe_column"] = int(column_descriptor)
            
    neurons_ext.to_csv(data_folder/"intermediate"/"neurons_ext.tsv", sep="\t")

    n= get_flywire_neuron(id, cache_folder=cache_folder, flywire_materialization=783, crop_to_hemibrain=False)

    n_prune = navis.prune_twigs(n, prune_factor)

    n_ds = navis.downsample_neuron(n_prune, downsampling_factor=ds_factor, inplace=False)

    return n_ds


def get_graph(edges):
    ak.connect()
    graph = ar.PropGraph()
    src = edges[:, 0].astype(np.int64) #since i'm importing the downloaded neurons
    dst = edges[:, 1].astype(np.int64) #since i'm importing the downloaded neurons
    graph.add_edges_from(ak.array(src), ak.array(dst))

    return graph

def draw_graph(graph, fig_size_x, fig_size_y):
    ak.connect()
    src, dst = graph.edges()
    src = src.to_ndarray()
    dst = dst.to_ndarray()

    edge_list = np.column_stack((src, dst)).tolist()

    nx_display = nx.Graph()
    nx_display.add_edges_from(edge_list)
    plt.figure(figsize=(fig_size_x, fig_size_y))

    pos = nx.kamada_kawai_layout(nx_display)
    nx.draw_networkx(nx_display, pos, with_labels=True, node_size=450)
    plt.show()
    ak.disconnect()


def separate_numbers(s):
    # Split the string by the underscore character
    num1, num2 = s.split('_')
    return num1, num2

def draw_connection(df, fig_size_x, fig_size_y, connection_id):
    ak.connect()

    id1, id2 = separate_numbers(connection_id)
    temp_1 = df[df["n_id"] == id1]
    temp_2 = df[df["n_id"] == id2]
    temp_12 = df[df["n_id"] == connection_id]

    src_concat = ak.concatenate([temp_1['src'], temp_2['src'], temp_12['src']])
    dst_concat = ak.concatenate([temp_1['dst'], temp_2['dst'], temp_12['dst']])
    type_concat = ak.concatenate([temp_1['connection_type'], temp_2['connection_type'], temp_12['connection_type']])
    id_concat = ak.concatenate([temp_1['n_id'], temp_2['n_id'], temp_12['n_id']])

    df_concat = ak.DataFrame({'src': src_concat, 'dst': dst_concat, 'type': type_concat, "id": id_concat})

    edge_src = df_concat["src"].to_ndarray().tolist()
    edge_dst = df_concat["dst"].to_ndarray().tolist()
    edge_type = df_concat["type"].to_ndarray().tolist()
    id_type = df_concat["id"].to_ndarray().tolist()

    edge_list = []
    nx_display = nx.Graph()
    for (u, v, etype) in zip(edge_src, edge_dst, id_type):
        # edge_list.append((u, v, etype))
        nx_display.add_edge(u, v, edge_type=etype)

    plt.figure(figsize=(fig_size_x, fig_size_y))

    pos = nx.kamada_kawai_layout(nx_display)

    # Get the edge colors based on the edge_type attribute
    # edge_colors = ["red" if nx_display[u][v]['edge_type'] == 's' else "blue" if nx_display[u][v]['edge_type'] == 'n' else "green" for u, v in nx_display.edges()]
    edge_colors = ["blue" if nx_display[u][v]['edge_type'] == id1 else "green" if nx_display[u][v]['edge_type'] == id2 else "red" for u, v in nx_display.edges()]

    nx.draw_networkx(nx_display, pos, with_labels=True, node_size=750, edge_color=edge_colors)
    plt.show()
    ak.disconnect()



def spatial_connectome_creation(edges, n_ds):
    ### create new Graph ### 
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

    ### get coordinates and calculate distances of nodes ###
    s_bef = neighbor_pairs[:, 0, 0].astype(np.int64).tolist()
    s_af = neighbor_pairs[:, 0, 1].astype(np.int64).tolist()

    d_bef = neighbor_pairs[:, 1, 0].astype(np.int64).tolist()
    d_af = neighbor_pairs[:, 1, 1].astype(np.int64).tolist()

    # Create DataFrame for s_bef and s_af with their corresponding coordinates
    s_bef_coords = n_ds.nodes.set_index('node_id').loc[s_bef][['x', 'y', 'z']].reset_index()
    s_af_coords = n_ds.nodes.set_index('node_id').loc[s_af][['x', 'y', 'z']].reset_index()

    d_bef_coords = n_ds.nodes.set_index('node_id').loc[d_bef][['x', 'y', 'z']].reset_index()
    d_af_coords = n_ds.nodes.set_index('node_id').loc[d_af][['x', 'y', 'z']].reset_index()

    # Ensure lengths match by repeating coordinates for each occurrence
    s_bef_coords_repeated = s_bef_coords.loc[s_bef_coords.index.repeat(s_bef_coords.index.value_counts().reindex(s_bef_coords.index).fillna(1))]
    s_af_coords_repeated = s_af_coords.loc[s_af_coords.index.repeat(s_af_coords.index.value_counts().reindex(s_af_coords.index).fillna(1))]

    d_bef_coords_repeated = d_bef_coords.loc[d_bef_coords.index.repeat(d_bef_coords.index.value_counts().reindex(d_bef_coords.index).fillna(1))]
    d_af_coords_repeated = d_af_coords.loc[d_af_coords.index.repeat(d_af_coords.index.value_counts().reindex(d_af_coords.index).fillna(1))]

    # Extract coordinates as numpy arrays
    x_bef, y_bef, z_bef = s_bef_coords_repeated['x'].values, s_bef_coords_repeated['y'].values, s_bef_coords_repeated['z'].values
    x_af, y_af, z_af = s_af_coords_repeated['x'].values, s_af_coords_repeated['y'].values, s_af_coords_repeated['z'].values

    s_bef_x = x_bef.astype(np.int64).tolist()
    s_bef_y = y_bef.astype(np.int64).tolist()
    s_bef_z = z_bef.astype(np.int64).tolist()
    
    s_af_x = x_af.astype(np.int64).tolist()
    s_af_y = y_af.astype(np.int64).tolist()
    s_af_z = z_af.astype(np.int64).tolist()
    
    s_distances = np.sqrt((x_bef - x_af)**2 + (y_bef - y_af)**2 + (z_bef - z_af)**2).tolist()
    s_x = ((x_bef + x_af)/2).tolist()
    s_y = ((y_bef + y_af)/2).tolist()
    s_z = ((z_bef + z_af)/2).tolist()

    x_bef, y_bef, z_bef = d_bef_coords_repeated['x'].values, d_bef_coords_repeated['y'].values, d_bef_coords_repeated['z'].values
    x_af, y_af, z_af = d_af_coords_repeated['x'].values, d_af_coords_repeated['y'].values, d_af_coords_repeated['z'].values

    d_bef_x = x_bef.astype(np.int64).tolist()
    d_bef_y = y_bef.astype(np.int64).tolist()
    d_bef_z = z_bef.astype(np.int64).tolist()

    d_af_x = x_af.astype(np.int64).tolist()
    d_af_y = y_af.astype(np.int64).tolist()
    d_af_z = z_af.astype(np.int64).tolist()

    # Calculate distances using vectorized operations
    d_distances = np.sqrt((x_bef - x_af)**2 + (y_bef - y_af)**2 + (z_bef - z_af)**2).tolist()
    d_x = ((x_bef + x_af)/2).tolist()
    d_y = ((y_bef + y_af)/2).tolist()
    d_z = ((z_bef + z_af)/2).tolist()

    connection_type_n = ["n"] * len(src)
    
    return src, dst, s_bef, s_bef_x, s_bef_y, s_bef_z, s_af, s_af_x, s_af_y, s_af_z, s_x, s_y, s_z, s_distances, d_bef, d_bef_x, d_bef_y, d_bef_z, d_af, d_af_x, d_af_y, d_af_z, d_x, d_y, d_z, d_distances, connection_type_n


def snap_coordinates(src, dst, s_x, s_y, s_z, d_x, d_y, d_z, c_coordinates):
    data = {
        'x': s_x + d_x,
        'y': s_y + d_y,
        'z': s_z + d_z,
        'node_id': src + dst,
        'parent_id': [0] * len(src + dst)
    }
    
    df = pd.DataFrame(data).copy()

    new_n = navis.TreeNeuron(
        df[['node_id', 'parent_id', 'x', 'y', 'z']],
        None
    )

    n_ids, dists = new_n.snap(c_coordinates)

    return list(n_ids), dists


def synaptic_data(neurons, id_list):
    synaptic_src, synaptic_dst, x, y, z, n_combo, dists = [], [], [], [], [], [], []
    combinations = list(itertools.combinations(id_list, 2))

    for combo in combinations:
            x_temp, y_temp, z_temp = [], [], []
            
            df1 = neurons[combo[0]]["c"][neurons[combo[0]]["c"]["partner_id"] == combo[1]]
            # df2 = neurons[combo[1]]["n"].connectors[neurons[combo[1]]["n"].connectors["partner_id"] == combo[0]]

            # if df1.empty and df2.empty:
            if df1.empty:
                continue
    
            # connectors_df = pd.concat([df1, df2])

            x_temp.extend(df1.loc[:, "x"])
            y_temp.extend(df1.loc[:, "y"])
            z_temp.extend(df1.loc[:, "z"])

            c_coordinates = np.vstack((x_temp, y_temp, z_temp)).T

            idloc1, dist1 = snap_coordinates(neurons[combo[0]]["src"],neurons[combo[0]]["dst"], neurons[combo[0]]["s_x"],neurons[combo[0]]["s_y"],
                             neurons[combo[0]]["s_z"],neurons[combo[0]]["d_x"],neurons[combo[0]]["d_y"],neurons[combo[0]]["d_z"],
                             c_coordinates)
            idloc2, dist2 = snap_coordinates(neurons[combo[1]]["src"],neurons[combo[1]]["dst"], neurons[combo[1]]["s_x"],neurons[combo[1]]["s_y"],
                             neurons[combo[1]]["s_z"],neurons[combo[1]]["d_x"],neurons[combo[1]]["d_y"],neurons[combo[1]]["d_z"],
                             c_coordinates)
            
            synaptic_src.extend(idloc1)
            synaptic_dst.extend(idloc2)
            n_combo.extend([(f"{combo[0]}_{combo[1]}")] * len(idloc1))
            x.extend(x_temp)
            y.extend(y_temp)
            z.extend(z_temp)
            dists.extend(dist1 + dist2)
            
    connection_type_s = ["s"] * len(synaptic_src)

    return synaptic_src, synaptic_dst, connection_type_s, x, y, z, n_combo, dists


def overall(id_list):
    #ensure no duplicate ids
    id_list= list(set(id_list))

    #to remove neurons that failed loading
    id_list_copy= id_list.copy()

    #all of the columns in the new df
    neurons={}
    src, dst, s_bef, s_bef_x, s_bef_y, s_bef_z, s_af, s_af_x, s_af_y, s_af_z, s_x, s_y, s_z, s_distances, d_bef, d_bef_x, d_bef_y, d_bef_z, d_af, d_af_x, d_af_y, d_af_z, d_x, d_y, d_z, d_distances, connection_type_n, n_id = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for id in id_list:
        #load neuron and its connectors
        n, c = get_neuron_local(id, 3000, 1000)

        #if loading failed remove from copied list
        if n == None:
             id_list_copy.remove(id)
             continue
        
        #store neuron and connectors information in neuron dict
        neurons[id] = {"n": n, "c": c}
        edges = n.edges

        ##angst das namen doppelt vergeben werden, fueg noch neuronen id zu namen hinzu
        src_temp, dst_temp, s_bef_temp, s_bef_x_temp, s_bef_y_temp, s_bef_z_temp, s_af_temp, s_af_x_temp, s_af_y_temp, s_af_z_temp, s_x_temp, s_y_temp, s_z_temp, s_distances_temp, d_bef_temp, d_bef_x_temp, d_bef_y_temp, d_bef_z_temp, d_af_temp, d_af_x_temp, d_af_y_temp, d_af_z_temp, d_x_temp, d_y_temp, d_z_temp, d_distances_temp, connection_type_n_temp = spatial_connectome_creation(edges, n)
        
        neurons[id].update({"src": src_temp, "dst": dst_temp, "s_x": s_x_temp, "s_y": s_y_temp, "s_z": s_z_temp, "d_x": d_x_temp, "d_y": d_y_temp, "d_z": d_z_temp})

        src.extend(src_temp)
        dst.extend(dst_temp)
        s_bef.extend(s_bef_temp)
        s_bef_x.extend(s_bef_x_temp)
        s_bef_y.extend(s_bef_y_temp)
        s_bef_z.extend(s_bef_z_temp)
        s_af.extend(s_af_temp)
        s_af_x.extend(s_af_x_temp)
        s_af_y.extend(s_af_y_temp)
        s_af_z.extend(s_af_z_temp)
        s_x.extend(s_x_temp)
        s_y.extend(s_y_temp)
        s_z.extend(s_z_temp)
        s_distances.extend(s_distances_temp)
        d_bef.extend(d_bef_temp)
        d_bef_x.extend(d_bef_x_temp)
        d_bef_y.extend(d_bef_y_temp)
        d_bef_z.extend(d_bef_z_temp)
        d_af.extend(d_af_temp)
        d_af_x.extend(d_af_x_temp)
        d_af_y.extend(d_af_y_temp)
        d_af_z.extend(d_af_z_temp)
        d_x.extend(d_x_temp)
        d_y.extend(d_y_temp)
        d_z.extend(d_z_temp)
        d_distances.extend(d_distances_temp)
        connection_type_n.extend(connection_type_n_temp)
        n_id.extend([id]* len(src_temp))
    
    if id_list_copy == []:
        return None
    
    synaptic_src, synaptic_dst, connection_type_s, x, y, z, n_combo, dists= synaptic_data(neurons, id_list_copy)

    spatial_connectome_edge_dict = {
    "src": src + synaptic_src,
    "dst": dst + synaptic_dst,
    "s_bef": s_bef + [0]* len(synaptic_src),
    "s_bef_x": s_bef_x + [0]* len(synaptic_src),
    "s_bef_y": s_bef_y + [0]* len(synaptic_src),
    "s_bef_z": s_bef_z + [0]* len(synaptic_src),
    "s_af": s_af+ [0]* len(synaptic_src),
    "s_af_x": s_af_x+ [0]* len(synaptic_src),
    "s_af_y": s_af_y+ [0]* len(synaptic_src),
    "s_af_z": s_af_z+ [0]* len(synaptic_src),
    "s_x": s_x+ x,
    "s_y": s_y+ y,
    "s_z": s_z+ z,
    "s_distance": s_distances+ dists,
    "d_bef": d_bef+ [0]* len(synaptic_src),
    "d_bef_x": d_bef_x+ [0]* len(synaptic_src),
    "d_bef_y": d_bef_y+ [0]* len(synaptic_src),
    "d_bef_z": d_bef_z+ [0]* len(synaptic_src),
    "d_af": d_af+ [0]* len(synaptic_src),
    "d_af_x": d_af_x+ [0]* len(synaptic_src),
    "d_af_y": d_af_y+ [0]* len(synaptic_src),
    "d_af_z": d_af_z+ [0]* len(synaptic_src),
    "d_x": d_x+ x,
    "d_y": d_y+ y,
    "d_z": d_z+ z,
    "d_distance": d_distances+ dists,
    "n_id": n_id + n_combo,
    "connection_type": connection_type_n + connection_type_s
    }
    
    # Step 5: Connect to Arkouda and initialize the graph
    ak.connect()
    graph = ar.PropGraph()
    spatial_connectome_edge_df = ak.DataFrame(spatial_connectome_edge_dict)
    graph.load_edge_attributes(spatial_connectome_edge_df, source_column="src", destination_column="dst", 
                               relationship_columns=["s_bef", "s_bef_x", "s_bef_y", "s_bef_z", "s_af", "s_af_x", 
                                                     "s_af_y", "s_af_z", 's_x', "s_y", "s_z", "s_distance", "d_bef",
                                                     "d_bef_x", "d_bef_y", "d_bef_z", "d_af", "d_af_x", "d_af_y", "d_af_z",
                                                       "d_x", "d_y", "d_z", "d_distance", "n_id", "connection_type"])
    ak.disconnect()

    return graph ,spatial_connectome_edge_df, neurons
    # return 0


def draw3d_graph(neuron_list, colors):
    combinations = list(itertools.combinations(neuron_list, 2))
    new_n=[]
    for combo in combinations:

        conn = (combo[0].connectors[combo[0].connectors["partner_id"] == int(combo[1].name)])
        x_c = conn["x"].values.tolist()
        y_c = conn["y"].values.tolist()
        z_c = conn["z"].values.tolist()

        c_coordinates = np.vstack((x_c, y_c, z_c)).T
        n0_ids, dists = combo[0].snap(c_coordinates)
        n1_ids, dists = combo[1].snap(c_coordinates)

        for i in range(len(n0_ids)):
            x=[]
            y=[]
            z=[]
            node_id=[]
            parent_id=[]
            # dk = (dt[(dt["d_skel_id"] == 1064) & (dt["connection_type"] == 'n')][0]["s_x"])
            x.append(combo[0].nodes[combo[0].nodes['node_id'] == n0_ids[i]]['x'].values[0])
            y.append(combo[0].nodes[combo[0].nodes['node_id'] == n0_ids[i]]['y'].values[0])
            z.append(combo[0].nodes[combo[0].nodes['node_id'] == n0_ids[i]]['z'].values[0])
            
            node_id.append(n0_ids[i])
            parent_id.append(-1)

            x.append(combo[1].nodes[combo[1].nodes['node_id'] == n1_ids[i]]['x'].values[0])
            y.append(combo[1].nodes[combo[1].nodes['node_id'] == n1_ids[i]]['y'].values[0])
            z.append(combo[1].nodes[combo[1].nodes['node_id'] == n1_ids[i]]['z'].values[0])

            node_id.append(n1_ids[i])
            parent_id.append(n0_ids[i])

            data = {
            'x': x,
            'y': y,
            'z': z,
            'node_id': node_id,
            'parent_id': parent_id,
            }

            df = pd.DataFrame(data)
            new_n.append(navis.TreeNeuron(df, name=f'{combo[0].name}_{n0_ids[i]}_{combo[1].name}_{n1_ids[i]}'))

    new_n.extend(neuron_list)
    
    fig = navis.plot3d(new_n, color=['red'] * (len(new_n) - len(neuron_list)) + colors)
