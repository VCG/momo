import numpy as np
import json

def load_celltype_meta(celltype_meta_folder, celltpye_descriptor):
    filename = celltype_meta_folder/f"type_{celltpye_descriptor}.json"
    with open(filename) as f:
        return json.load(f)


def get_best_matched_pair(celltype_meta, mirrored=False):
    if(mirrored):
        scores = np.array(celltype_meta["scores_hemibrain_flywire_mirrored"])
    else:
        scores = np.array(celltype_meta["scores_hemibrain_flywire"])
     
    indices = np.unravel_index(np.argmax(scores), scores.shape)        
    hemibrain_id = celltype_meta["ids_hemibrain"][indices[0]]
    flywire_id = celltype_meta["ids_flywire"][indices[1]]

    return hemibrain_id, flywire_id, np.max(scores)


def get_best_representative(celltype_meta):
    hemibrain_id, flywire_id, score = get_best_matched_pair(celltype_meta, mirrored=False)
    hemibrain_id_mirrored, flywire_id_mirrored, score_mirrored = get_best_matched_pair(celltype_meta, mirrored=True)

    if(score_mirrored > score):
        return hemibrain_id_mirrored, flywire_id_mirrored, True
    else:
        return hemibrain_id, flywire_id, False


def split_by_annotation(celltype_meta, key_neuron_id = "ids_flywire", key_annotation = "sides_flywire"):
    ids_by_annotation = {}
    for idx, neuron_id in enumerate(celltype_meta[key_neuron_id]):
        annotation = celltype_meta[key_annotation][idx]
        if(annotation not in ids_by_annotation):
            ids_by_annotation[annotation] = [neuron_id]
        else:
            ids_by_annotation[annotation].append(neuron_id)
    return ids_by_annotation