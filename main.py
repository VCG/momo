from enum import Enum
from fastapi import FastAPI, Query, Depends, HTTPException
from pydantic import BaseModel
import arkouda as ak
import arachne as ar
from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt
from contextlib import asynccontextmanager
from compcon.create_graph import get_graph, get_edge_dict, get_node_dict
import json

# Global variables to store nodes and edges
first_graph = None
nodes = None
edges = None
node_dict = None
edge_dict = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    ak.connect()
    initialize_graph()
    yield
    ak.disconnect()

app = FastAPI(lifespan=lifespan)

# Function to initialize graph
def initialize_graph():
    global first_graph, nodes, edges, node_dict, edge_dict, subgraph        
    subgraph = ar.PropGraph()
    first_graph = get_graph()
    nodes = first_graph.nodes()
    edges = first_graph.edges()
    # node_dict = get_node_dict()
    # edge_dict = get_edge_dict()
    
# Pydantic models to serialize data
class NodesResponse(BaseModel):
    nodes: List[int]

class EdgesResponse(BaseModel):
    src: List[int]
    dst: List[int]

class FilterResponse(BaseModel):
    nodes: int
    edges: int
    message: str

# Default endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the graph API"}

#include to which dataset: {flywire} | {hemibrain}
# Endpoint to return nodes
@app.get("/nodes", response_model=NodesResponse)
def get_nodes() -> NodesResponse:
    nodes_list = nodes.to_list()  # Convert pdarray to list for JSON serialization
    return NodesResponse(nodes=nodes_list)

#include to which dataset: {flywire} | {hemibrain}
# Endpoint to return edges
@app.get("/edges", response_model=EdgesResponse)
def get_edges() -> EdgesResponse:
    src_list, dst_list = edges
    src_list = src_list.to_list()  # Convert pdarray to list for JSON serialization
    dst_list = dst_list.to_list()  # Convert pdarray to list for JSON serialization
    return EdgesResponse(src=src_list, dst=dst_list)

# Endpoint to return nodes and plot the graph
@app.get("/nodes_and_plot")
def get_nodes_and_plot():
    # Fetch nodes and edges
    nodes = first_graph.nodes()
    edge_src, edge_dst = first_graph.edges()
    edge_src = edge_src.to_list()
    edge_dst = edge_dst.to_list()
    
    # Convert edge arrays to a list of tuples
    edge_list = [(u, v) for u, v in zip(edge_src, edge_dst)]
    
    # Create a networkx graph
    nx_display = nx.Graph()
    nx_display.add_edges_from(edge_list)
    
    seed= 31
    # Plot the graph
    plt.figure(figsize=(30, 30))  # Adjust the figure size as needed
    pos = nx.spring_layout(nx_display, seed=seed)
    nx.draw(nx_display, pos, with_labels=True, node_size=200, node_color='skyblue', edge_color='gray')
    
    # Save the plot to a file
    plt.savefig("graph_plot.png")
    
    # Return the nodes and the path to the generated plot
    return {
        "nodes": nodes.to_list(),
        "plot_path": "graph_plot.png"
    }


def s_parse_int_list(s: str) -> List[int]:
    try:
        return [int(i) for i in s.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid integer in query parameters")
    
def d_parse_int_list(d: str) -> List[int]:
    try:
        return [int(i) for i in d.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid integer in query parameters")


@app.get("/subgraph/")
def read_items(s: List[int] = Depends(s_parse_int_list), d: List[int] = Depends(d_parse_int_list)):
    
    subgraph.add_edges_from(ak.array(s), ak.array(d))
    nodes = subgraph.nodes()
    source, destination= subgraph.edges()
    isos = ar.subgraph_isomorphism(first_graph, subgraph)
    # Ensure the returned data is JSON-serializable
    #
    isos_list = isos.to_list()
    num_sub = len(isos_list) / len(nodes)
    unique_isos_list = list(set(isos_list))
    return {"vertices in subgraph": len(nodes),"number of isomorphic subgraphs": num_sub,
             "number of isos": len(isos_list), "unique elements": unique_isos_list,
             "total number of subgraphs": len(unique_isos_list)/len(nodes)}

    
