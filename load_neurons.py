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
from compcon.navis_api import get_flywire_neuron
from compcon.mapping import *
from compcon.delta import FeatureVector, GroupFeatures, DeltaFeatures

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

unique_col = neurons_ext["optic_lobe_column"].dropna().unique()[0:1]
# print("UNIQUE COLUMN", unique_col)

col_id={}
for col in unique_col:
    col_id[col] = neurons_ext[neurons_ext["optic_lobe_column"]==col]["root_id"].dropna().tolist()

def loads():
    neurons = {}
    for col in col_id:
        neurons[col]=[]
        for idd in col_id[col]:
            try:
                neurons[col].append(get_flywire_neuron(idd, cache_folder=cache_folder, flywire_materialization=783, crop_to_hemibrain=False))

            except Exception as e:
                print(f"Error retrieving data for root_id {idd}: {e}")

    return neurons

def neurons_extf():
     return neurons_ext





