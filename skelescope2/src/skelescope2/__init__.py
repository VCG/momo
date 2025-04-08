import importlib.metadata
import pathlib

import anywidget
import traitlets

import anywidget
import traitlets
import arkouda as ak
import arachne as ar
import numpy as np
import pandas as pd
import networkx as nx
from compcon.create_graph import get_neuron_local

try:
    __version__ = importlib.metadata.version("neuronal_motif_anywidget_test")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

def back_transformation_edges(d, src_mapped, dst_mapped):
    skeleton_edges = []
    neuron_ids = []

    s_skeleton_edges = []
    s_neuron_ids = []
    synapse_coordinates=[]
    x_syn, y_syn, z_syn = [], [], []
    for i in range(len(src_mapped)):
        df_temp = d[(d['dst'] == src_mapped[i]) & (d['src'] == dst_mapped[i])][0]
        if df_temp["connection_type"] == 'n':
            new_element = [df_temp["s_bef"], df_temp["s_af"]]
            # if new_element not in skeleton_edges:
            skeleton_edges.append(new_element)
            neuron_ids.append(int(df_temp["n_id"]))

            new_element = [df_temp["d_bef"], df_temp["d_af"]]
            # if new_element not in skeleton_edges:
            skeleton_edges.append(new_element)
            neuron_ids.append(int(df_temp["n_id"]))

        else:
            string = df_temp["n_id"]
            num1, num2 = map(int, string.split("_"))
            
            skeleton_edges.append([df_temp["s_bef"], df_temp["s_af"]])
            neuron_ids.append(num1)
            
            skeleton_edges.append([df_temp["d_bef"], df_temp["d_af"]])
            neuron_ids.append(num2)
            df_temp = d[(d['dst'] == src_mapped[i]) & (d['src'] == dst_mapped[i])]
            for i in range(len(df_temp)):
                synapse_coordinates.append([df_temp[i]["d_x"], df_temp[i]["d_y"], df_temp[i]["d_z"]])


    return skeleton_edges, neuron_ids, s_skeleton_edges, s_neuron_ids, synapse_coordinates

def test_graph_creation(edges):
    first_elements = edges[:, 0]
    second_elements = edges[:, 1]
    neighbor_matrix = (first_elements[:, np.newaxis] == second_elements[np.newaxis, :])

    neighbor_indices = np.argwhere(neighbor_matrix)
    neighbor_indices = neighbor_indices[neighbor_indices[:, 0] != neighbor_indices[:, 1]]
    neighbor_pairs = edges[neighbor_indices]

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
        dst.append(s[0][0] * 10000 + s[0][1])
        src.append(s[1][0] * 10000 + s[1][1])
        branch_type.append("s")
    
    return src, dst, branch_type
    
def mapping_edges(subgraph, mapping):
    edges = subgraph.edges()
    src = edges[0].to_ndarray()
    dst = edges[1].to_ndarray()

    src = src.astype(str)
    dst = dst.astype(str)

    src_mapped = np.vectorize(mapping.get)(src)
    dst_mapped = np.vectorize(mapping.get)(dst)
    
    return src_mapped, dst_mapped

def vis_transfromation(d, motif, mapping, nodeid_color_mapping, dataset="cave"):
    ak.connect()
    transformed_dataset = ak.DataFrame(d.to_dict(orient='list'))
    transformed_dataset = transformed_dataset[~((transformed_dataset["s_bef"] == 0) & (transformed_dataset["s_af"] == 0))]
    d = transformed_dataset[~((transformed_dataset["d_bef"] == 0) & (transformed_dataset["d_af"] == 0))]
    branches, synapses = [], []
    for m in motif:
        if m["properties"][0] == "neuron connection":
            branches.append(m["label"])
            
        if m["properties"][0] == "synaptic connection":
            synapses.append([m["properties"][1], m["properties"][2]])
    
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

    skeleton_edges, neuron_ids, s_skeleton_edges, s_neuron_ids, synapse_coordinates = back_transformation_edges(d, src_mapped, dst_mapped)

    def sagments1(neuronids, neuronedges, dataset):
        kp = {}
        for idx, x in enumerate(neuronids):
            n = get_neuron_local(x, dataset=dataset)
            nx_display = nx.Graph()
            nx_display.add_edges_from(n.edges)
            
            shortest_path = nx.shortest_path(nx_display, source=neuronedges[idx][0], target=neuronedges[idx][1], weight='weight')
    
            if neuronids[idx] not in kp:
                kp[neuronids[idx]] = shortest_path
            else:
                kp[neuronids[idx]].extend(shortest_path)
        
        return kp
    dicc = sagments1(neuron_ids, skeleton_edges, dataset)

    def neuron_data_creation(pf, nodeid_color_mapping, dataset):
        neuron_format=[]
        for i in pf:
            dic={}
            temptext=None
            if dataset == 'flywire':
                with open(f'/home/michaelshewarega/Desktop/test/test_folder/sk_lod1_783_healed/{i}.swc', 'r') as file:
                    temptext = file.read() 
            
            elif dataset=='cave':
                with open(f'/home/michaelshewarega/Desktop/test/cave_data_converted/{i}.swc', 'r') as file:
                    temptext = file.read() 
            
            dic["swcText"]= temptext
            dic["neuronName"] = "test"
            dic["color"]=nodeid_color_mapping[str(i)]
            dic["kp"]=pf[i]
            dic["kpcolor"]={}
            neuron_format.append(dic)
        return neuron_format

    fin = neuron_data_creation(dicc, nodeid_color_mapping, dataset)
    
    return fin, synapse_coordinates


class Skelescope2(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    value = traitlets.Int(0).tag(sync=True)
    
    neuronss = traitlets.List([]).tag(sync=True)
    synapsess = traitlets.List([]).tag(sync=True)
    dataset = traitlets.Unicode("cave").tag(sync=True)
    

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observe(self.on_dataset_change, names="dataset")
        
    def visualize_motif(self, d, motif, mapping, nodeid_color_mapping, dataset):
        self.dataset=dataset
        self.neuronss, self.synapsess = vis_transfromation(d, motif, mapping, nodeid_color_mapping, dataset)
    
    def on_dataset_change(self, change):
        self.dataset = change['new']
        
