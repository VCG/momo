## Explanation of columns for `Supplemental_file1_neuron_annotations.tsv` and `Supplemental_file2_non_neuron_annotations.tsv`
- `pos_x`, `pos_y`, `pos_z` are anchor coordinates in 4x4x40nm voxel space (typically on the backbone of the neuron) for given neuron
- `supervoxel_id` is the ID of the supervoxel the anchor coordinates map to
- `root_id` is the ID of the neuron in the FlyWire `630` release
- `soma_x`, `soma_y`, `soma_z` are soma coordinates in 4x4x40nm voxel space
- `nucleus_id` maps to the nucleus detection table (available through CAVE); neurons with `soma_x/y/z` but without `nucleus_id` have had their soma manually marked
- `flow`, `super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `hemibrain_type` are the hierarchical annotations
- `ito_lee_hemilineage` and `hartenstein_hemilineage` provide the hemilineage in ItoLee and Hartenstein nomenclature; note that not all labels exist in the Hartenstein nomenclature
- `morphology_group` provides a coarse morphological grouping based on hemilineage clustering
- `top_nt` is the top predicted neurotransmitter for the given neuron, calculated by averaging confidences over the transmitter predictions for all presynapses for the given neuron and choosing the most confident transmitter
- `top_nt_conf` is the average confidence for the top neurotransmitter
- `side` refers to the soma side for brain-intrinsic neurons and the nerve-entry side for sensory/ascending neurons
- `fbbt_id` and `vfb_id` provide mappings to the database of the [VirtualFlyBrain](https://virtualflybrain.org/); the former is used for ontology terms (like cell types), the latter is an identifier for individual neurons
- `status` records a number of outliers: `outlier_seg` are neurons with segmentation issues (often due to dark cytosol); `outlier_bio` are neurons with small to medium sized differences (e.g. extra branches) compared with their contralateral or hemibrain homologues


## Explanation of columns for `Supplemental_file3_summary_with_ngl_links.csv`
- `ito_lee_hemilineage` and `hartenstein_hemilineage` provide the hemilineage in ItoLee and Hartenstein nomenclature; note that not all labels exist in the Hartenstein nomenclature.
- `notes` record some of our observations of select published clones.
- `*_ngl_short` (`all`,`center`,`left` and `right` sides) columns contain neuroglancer links with the neurons in that hemilineage selected, on that side, and coloured based on the colouring provided in `Supplemental_file3_hemilineages_clustering.csv`. Each morphological group is put in a separate layer. There is also a de-selected layer, named based on the hemilineage, that contains the same neuorns.
- `*_fw` (`all`,`center`,`left` and `right` sides) columns contain flywire links with the neurons in that hemilineage selected, on that side, and coloured based on the colouring provided in `Supplemental_file3_hemilineages_clustering.csv`.
- `number_of_clusters_*` (`center`,`left` and `right`) columns contain the number of morphological groups in that hemilineage on different sides.
- `number_of_neurons_*` (`center`,`left` and `right`) columns contain the number of neurons per hemilineage on different sides.
- `roots_*` (`left`,`right`) contains the root_630 ids of each hemilineage on each side.
- `ito_lee_lineage` and `hartenstein_lineage` provide the corresponding lineage names to the hemilineages.
- `hemibrain_map` contains the rough* mapping of that hemilineage to a hemibrain cell body fiber.
- `is_hemilineage` contains our best guesses on whether the entry is a hemilineage or not. For instance, `H(NT)` means that we guess that this is a hemilineage (instead of a combination of two hemilineages of one lineage) based on the neurotransmitter information from Eckstein et al. 2023.

## Explanation of columns for `Supplemental_file3_hemilineages_clustering.csv`
- `supervoxel_id` corresponds to the `supervoxel_id`s in the `Supplemental_file1_annotations.tsv` file.
- `root_630` is the ID of the neuron in the FlyWire `630` release.
- `ito_lee_hemilineage` and `hartenstein_hemilineage` provide the hemilineage in ItoLee and Hartenstein nomenclature; note that not all labels exist in the Hartenstein nomenclature.
- `dendrogram_order` records the order of `root_ids` on the dendrograms (exemplified in `Fig 2I`) / morphological proximity of neurons of one hemilineage.
- `group` records the groups formed after cutting the dendrograms using the `elbow` method. The integers start from 1 on the left of the dendrogram.
- `morphological_group` is `ito_lee_hemilineage` and `group` concatenated together, such that each morphological group has a unique value.
- `neuropil` contains the top three innervated neuropils by neurons of the current morphological group, as counted by the amount of skeleton in the respective neuropils. The neuropils are alphabetically ordered within each entry.
- `col_elbow` is the colour (in hex) used in `Fig S2.1` and `Supplemental_file_2_summary_with_links.csv`. Each morphological group within a hemilineage is assigned one colour.
- `height_10` contains the groups within each hemilineage after cutting the dendrograms (exemplified in `Fig 2I`) at `distanct = 10` - this distance seemed the next best place to separate the hemilineages uniformly, following the `elbow` method.

## Explanation of columms for `Supplemental_file5_hemibrain_meta.csv`
The following columns correspond 1:1 to columns shown/available for download through neuPrint: `bodyId`, `type`, `notes`, `status`, `cellBodyFiber`, `somaLocation`, `pre`, `post` and `cropped`.

We additionally added the following columns:
- in `morphology_type` we collapsed connectivity types back into morphology types (i.e. removed the `_a`, `_b`, etc. suffixes)
- `cell_class` contains labels analogous to those the cell class labels we provide for FlyWire neurons
- `side` refers to the soma side and is principally based on the `instance` column where an `_L` and `_R` typically indicates the side; we did however make a sizeable number of manual adjustments
- `pre_con2` contains the number of outgoing connections for a given neuron whereas `pre` contains the number of presynapses (remember that insect synapses are polyadic); these numbers were used to compare hemibrain vs FlyWire presynapse counts
- `fbbt_id` contains a FBbt ID that maps to the entry for a given hemibrain type in the VirtualFlyBrain database
- `ito_lee_hemilineage` provide identified hemilineages in ItoLee nomenclature
- `morphology_group` provides a coarse morphological grouping based on hemilineage clustering


