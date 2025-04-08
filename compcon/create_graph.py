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

def get_neuron_local(id, prune_factor=None, ds_factor=None, preserve_nodes=[], dataset=''):
    try:
        n=None
        # Read the neuron data from the SWC file
        if dataset== 'cave':
            n = navis.read_swc(f'cave_data_converted/{id}.swc')  

        elif dataset== 'flywire':
            n = navis.read_swc(f'test_folder/sk_lod1_783_healed/{id}.swc')
        #prune neuron if factor is provided            
        if prune_factor is not None:
            n_prune = navis.prune_twigs(n, prune_factor, recursive= 1)

        else:
            n_prune = n
        #downsample neuron if factor is provided
        if ds_factor is not None:
            n_ds = navis.downsample_neuron(n_prune, downsampling_factor=ds_factor, preserve_nodes=preserve_nodes, inplace=False)

        else:
            n_ds = n_prune
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        n_ds = None

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        n_ds = None
    
    return n_ds