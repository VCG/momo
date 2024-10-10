import arachne as ar
import numpy as np

def find_isomorphic_subgraphs(hostgraph, subgraph):
    src_sub, dst_sub = subgraph.edges()
    src_sub = src_sub.to_ndarray()
    dst_sub = dst_sub.to_ndarray()

    # Find isomorphic subgraphs
    isos = ar.subgraph_isomorphism(hostgraph, subgraph)
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

    print(f"Number of Mappings found: {number_isos_found}")
    return all_mappings