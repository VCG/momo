import navis
import pandas as pd
from pathlib import Path
import os

import flybrains
import navis.interfaces.neuprint as neu
from fafbseg import flywire
from navis.io.mesh_io import write_mesh, read_mesh

RESOURCES = None

def init_resources():
    global RESOURCES

    if(RESOURCES is not None):
        return
    
    RESOURCES = {}
    
    def get_neuprint_token():
        token_file = Path(__file__).parent.parent/"_private"/"token_neuprint"
        with open(token_file) as f:
            return f.readline()

    RESOURCES["neuprint_client"] = neu.Client(
        "https://neuprint.janelia.org/",
        token=get_neuprint_token(),
        dataset='hemibrain:v1.2.1'
    )

    def get_cave_token():
        token_file = Path(__file__).parent.parent/"_private"/"token_cave"
        with open(token_file) as f:
            return f.readline()
    
    try:
        flywire.get_chunkedgraph_secret()
    except:
        flywire.set_chunkedgraph_secret(get_cave_token())

    bbox = flybrains.JRCFIB2018Fraw.bbox
    RESOURCES["hemibrain_BB"] = navis.xform_brain(bbox, source="JRCFIB2018Fraw", target="JRCFIB2018F")#changed
    
    
def get_hemibrain_neuron(body_id, cache_folder=None, verbose=True):
    global RESOURCES

    if(cache_folder is None):
        cache_hit = False
    else:
        cache_folder_hemibrain = cache_folder/"hemibrain"
        os.makedirs(cache_folder_hemibrain, exist_ok=True)
        
        swc_file = cache_folder_hemibrain/f"{body_id}.swc"
        connectors_file = cache_folder_hemibrain/f"{body_id}_connectors.csv"
        cache_hit = swc_file.exists()

    if(cache_hit):
        neuron = navis.read_swc(swc_file)
        neuron.connectors = pd.read_csv(connectors_file)
        if(verbose):
            print(f"read from cache {body_id}")
    else:
        init_resources()

        nl = navis.interfaces.neuprint.fetch_skeletons(body_id, with_synapses=True)
        assert len(nl) == 1
        neuron = navis.xform_brain(nl[0], source="JRCFIB2018Fraw", target="JRCFIB2018F")#changed from JRC2018F

        if(cache_folder is not None):
            neuron.to_swc(swc_file, export_connectors=True)
            neuron.connectors.to_csv(connectors_file, index=False)

    return neuron


def get_flywire_neuron(flywire_id, mirror_hemisphere=False, crop_to_hemibrain=False, cache_folder=None, 
                        flywire_materialization=783, verbose=True, transform_target = "JRCFIB2018F", #changed
                        load_mesh=False):
    global RESOURCES

    if(cache_folder is None):
        cache_hit = False
    else:
        cache_folder_flywire = cache_folder/f"flywire_{flywire_materialization}"
        if(mirror_hemisphere):
            cache_folder_flywire = cache_folder_flywire/"mirrored"
        else:
            cache_folder_flywire = cache_folder_flywire/"unmirrored"
        if(crop_to_hemibrain):
            cache_folder_flywire = cache_folder_flywire/"cropped"
        else:
            cache_folder_flywire = cache_folder_flywire/"uncropped"
        os.makedirs(cache_folder_flywire, exist_ok=True)

        if(load_mesh):
            morphology_file = cache_folder_flywire/f"{flywire_id}.ply"
            connectors_file = cache_folder_flywire/f"{flywire_id}_connectors_mesh.csv"
        else:
            morphology_file = cache_folder_flywire/f"{flywire_id}.swc"
            connectors_file = cache_folder_flywire/f"{flywire_id}_connectors.csv"

        cache_hit = morphology_file.exists() and connectors_file.exists()
        
    if(cache_hit):

        if(load_mesh):
            neuron = read_mesh(morphology_file)  
        else:
            neuron = navis.read_swc(morphology_file)

        connectors = pd.read_csv(connectors_file)
        neuron.connectors = connectors
   
        if(verbose):
            print(f"read from cache {flywire_id}")
    else:
        init_resources()

        if(load_mesh):
            #flywire.utils.get_cave_datastacks()
            morphology = flywire.get_mesh_neuron(flywire_id, dataset='flywire_fafb_public')
        else:
            morphology = flywire.get_skeletons(flywire_id, dataset=flywire_materialization)
        flywire.get_synapses(morphology, attach=True, materialization=flywire_materialization)
        
        neuron = navis.xform_brain(morphology, source="FLYWIRE", target=transform_target)
        if(mirror_hemisphere):
            neuron = navis.mirror_brain(neuron, template="JRCFIB2018F")#changed
        if(crop_to_hemibrain):
            neuron = navis.in_volume(neuron, RESOURCES["hemibrain_BB"])

        if(cache_folder is not None):

            if(load_mesh):
                write_mesh(neuron, morphology_file)
            else:
                neuron.to_swc(morphology_file)
            
            neuron.connectors.to_csv(connectors_file, index=False)
    
    return neuron

