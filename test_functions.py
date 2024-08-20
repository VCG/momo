def tast_snap_coordinates(src, dst, s_x, s_y, s_z, d_x, d_y, d_z, c_coordinates):
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


def tast_synaptic_data(neurons, id_list):
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

            idloc1, dist1 = tast_snap_coordinates(neurons[combo[0]]["src"],neurons[combo[0]]["dst"], neurons[combo[0]]["s_x"],neurons[combo[0]]["s_y"],
                             neurons[combo[0]]["s_z"],neurons[combo[0]]["d_x"],neurons[combo[0]]["d_y"],neurons[combo[0]]["d_z"],
                             c_coordinates)
            idloc2, dist2 = tast_snap_coordinates(neurons[combo[1]]["src"],neurons[combo[1]]["dst"], neurons[combo[1]]["s_x"],neurons[combo[1]]["s_y"],
                             neurons[combo[1]]["s_z"],neurons[combo[1]]["d_x"],neurons[combo[1]]["d_y"],neurons[combo[1]]["d_z"],
                             c_coordinates)
            
            synaptic_src.extend(idloc1)
            synaptic_dst.extend(idloc2)
            n_combo.extend([(f"{combo[0]}_{combo[1]}")] * len(idloc1))
            x.extend(x_temp)
            y.extend(y_temp)
            z.extend(z_temp)
            dists.extend(dist1 + dist2)
            # Method 1: Using np.vectorize
            vectorized_mapping = np.vectorize(neurons[combo[0]].get)
            srccc = vectorized_mapping(idloc1)

            vectorized_mapping = np.vectorize(neurons[combo[1]].get)
            dsttt = vectorized_mapping(idloc2)

            
    connection_type_s = ["s"] * len(synaptic_src)

    return synaptic_src, synaptic_dst, connection_type_s, x, y, z, n_combo, dists, list(srccc), list(dsttt)



def tast_overall(id_list):
    id_list= list(set(id_list))
    id_list_copy= id_list.copy()
    neurons={}
    src, dst, s_x, s_y, s_z, d_x, d_y, d_z, connection_type_n, n_id, s_skel_id, d_skel_id = [], [], [], [], [], [], [], [], [], [], [], []
    i=0
    for id in id_list:
        n, c = get_neuron_local(id, 3000, 1000)

        if n == None:
             i+=1
             print(i)
             id_list_copy.remove(id)
             continue
        
        neurons[id] = {"n": n, "c": c}
        
        # Filter the relevant nodes (excluding "root")
        filtered_nodes = n.nodes[n.nodes["type"] != "root"]

        # Extract src and dst IDs
        src_temp = filtered_nodes["parent_id"].values.tolist()
        dst_temp = filtered_nodes["node_id"].values.tolist()

        filtered_nodes = n.nodes

        # Create a DataFrame that maps node_id to its coordinates
        id_to_coordinates = filtered_nodes.set_index('node_id')[['x', 'y', 'z']]

        # Use .loc to get the coordinates for src and dst IDs
        s_coordinates = id_to_coordinates.loc[src_temp]
        d_coordinates = id_to_coordinates.loc[dst_temp]

        src_id = np.array(src_temp) + 10 * np.array(id) 
        dst_id = np.array(dst_temp) + 10 * np.array(id) 
        
        src_id = list(src_id)
        dst_id = list(dst_id)
        # print(src_temp)

        # Extract x, y, z coordinates for src and dst
        s_x_temp, s_y_temp, s_z_temp = s_coordinates['x'].values.tolist(), s_coordinates['y'].values.tolist(), s_coordinates['z'].values.tolist()
        d_x_temp, d_y_temp, d_z_temp = d_coordinates['x'].values.tolist(), d_coordinates['y'].values.tolist(), d_coordinates['z'].values.tolist()

        connection_type_n_temp = ["n"] * len(src_id)

        # neurons[id] = {"n": n, "c": c}
        # edges = n.edges

        ##angst das namen doppelt vergeben werden, fueg noch neuronen id zu namen hinzu
        # src_temp, dst_temp, s_x_temp, s_y_temp, s_z_temp,  d_x_temp, d_y_temp, d_z_temp, connection_type_n_temp = spatial_connectome_creation(edges, n)
        
        neurons[id].update({"src": src_id, 
                            "dst": dst_id, 
                            "s_x": s_x_temp, 
                            "s_y": s_y_temp, 
                            "s_z": s_z_temp, 
                            "d_x": d_x_temp, 
                            "d_y": d_y_temp, 
                            "d_z": d_z_temp, 
                            **dict(zip(src_id, src_temp)), 
                            **dict(zip(dst_id, dst_temp))})

        src.extend(src_id)
        dst.extend(dst_id)
        s_x.extend(s_x_temp)
        s_y.extend(s_y_temp)
        s_z.extend(s_z_temp)
        d_x.extend(d_x_temp)
        d_y.extend(d_y_temp)
        d_z.extend(d_z_temp)
        connection_type_n.extend(connection_type_n_temp)
        n_id.extend([id]* len(src_temp))
        s_skel_id.extend(src_temp)
        d_skel_id.extend(dst_temp)
    
    if id_list_copy == []:
        return None
    
    synaptic_src, synaptic_dst, connection_type_s, x, y, z, n_combo, dists , srccc, dsttt= tast_synaptic_data(neurons, id_list_copy)

    spatial_connectome_edge_dict = {
    "src": src + synaptic_src,
    "dst": dst + synaptic_dst,
    "s_skel_id": s_skel_id+ srccc,
    "s_x": s_x+ x,
    "s_y": s_y+ y,
    "s_z": s_z+ z,
    "d_skel_id": d_skel_id+ dsttt,
    "d_x": d_x+ x,
    "d_y": d_y+ y,
    "d_z": d_z+ z,
    "n_id": n_id + n_combo,
    "connection_type": connection_type_n + connection_type_s
    }
    # Step 5: Connect to Arkouda and initialize the graph
    ak.connect()
    graph = ar.PropGraph()
    spatial_connectome_edge_df = ak.DataFrame(spatial_connectome_edge_dict)
    graph.load_edge_attributes(spatial_connectome_edge_df, source_column="src", destination_column="dst", 
                               relationship_columns=['s_x', "s_y", "s_z",  
                                                     "d_x", "d_y", "d_z", "n_id", "connection_type"])
    # ak.disconnect()

    return graph ,spatial_connectome_edge_df, neurons