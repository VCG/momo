import * as React from "react";
import {useState, useEffect} from "react";
import { createRender, useModelState } from "@anywidget/react";
import {Sketch} from "@vimo-public/vimo-sketches";
import "./widget.css";
import TableContainer from "@mui/material/TableContainer";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableRow from '@mui/material/TableRow';
import TableCell from '@mui/material/TableCell';
import Paper from '@mui/material/Paper';
import { IconButton, Collapse, List, ListItem, ListItemText } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const render = createRender(() => {
	const [value, setValue] = useModelState("value");
	const [motif_json, setMotif_json] = useModelState("motif_json");
	const [limm, setLimm] = useModelState("limm");
	const [node_mapping, setNode_mapping] = useModelState("node_mapping");
	const [current_mapping, setCurrent_mapping] = useModelState("current_mapping");
	const [nodeid_color_mapping, setNodeid_color_mapping] = useModelState("nodeid_color_mapping");
	const [current_nodeid_color_mapping, setCurrent_nodeid_color_mapping] = useModelState("current_nodeid_color_mapping");
  const [selectedIndex, setSelectedIndex] = useModelState("selectedIndex");
  const [loading, setLoading] = useModelState("loading");
  
  const [isLoading, setIsLoading] = useState(false);
  const [expandedIndex, setExpandedIndex] = useState(null);
	const [attributes, setAttributes] = useState({displayMotifCount: false});

  const sketch_colors = ["#000000", "#00880A", "#003090", "#028785", "#c2c800"];
  const transformed_colors = ["#000000", "#00880A", "#003090", "#028785", "#c2c800"];
  const hexToRGBA = (hex, opacity) => {
    // Remove '#' if it's present
    const cleanHex = hex.replace('#', '');
  
    // Parse r, g, b from the hex color
    const r = parseInt(cleanHex.substring(0, 2), 16);
    const g = parseInt(cleanHex.substring(2, 4), 16);
    const b = parseInt(cleanHex.substring(4, 6), 16);
  
    // Return RGBA color with specified opacity
    return `rgba(${r}, ${g}, ${b}, ${opacity})`;
  };
  const color_transformation = sketch_colors.reduce((acc, color, index) => {
    acc[color] = transformed_colors[index];
    return acc;
  }, {});

  const toggleExpand = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index); // Toggle the expanded state
  };


  useEffect(() => {
    
  }, [loading]);

  const processRequest = async (motifJson, lim) => {
      console.log("This function is called upon clicking the search button.");
         if ((motifJson.edges !== motif_json) | (lim !== limm)) {
          console.log(motifJson.edges);
          setLimm(lim);
          setMotif_json(motifJson.edges);
          setLoading(true);  // Set loading to true when the request starts
             
        }
 
  };
  
	return (
          <div>
            <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center'}}>
              <div style={{width: "1000px"}}>
                <Sketch processRequest={processRequest} attributes={attributes} />
              </div>
              <div style={{width: "300px", marginRight:"20px"}}>
              {loading ? (
              <div>Loading...</div>
            ) : node_mapping.length === 0 ? (
              <div>Nothing found</div>
             ) :Array.isArray(node_mapping) && node_mapping.length === 1 && node_mapping[0] === "error" ? (<div>Runtime Error</div>): (<div>
                  <h3 style={{ textAlign: 'center', border: '0.5px solid gray', borderRadius: '12px', padding: '8px', maxWidth: '300px' }}>Results</h3>

                  <TableContainer component={Paper} style={{ maxHeight: '200px', maxWidth: '300px', overflowY: 'auto', border: '0.5px solid gray', borderRadius: '12px', alignContent: 'center' }}>
                    <Table aria-label="collapsible table" style={{ tableLayout: "fixed" }}>
                      <TableBody>
                        {node_mapping.map((row, index) => (
                          <React.Fragment key={index}>
                            <TableRow>
                              <TableCell 
                                style={{ 
                                  textAlign: 'center', 
                                  cursor: 'pointer', 
                                  backgroundColor: selectedIndex === index ? 'lightgray' : 'transparent' 
                                }} 
                                onClick={() => {
                                  setSelectedIndex(index);
                                  setCurrent_mapping(row); 
                                  setCurrent_nodeid_color_mapping(nodeid_color_mapping[index]);
                                }}
                              >
                                {`Motif Instance ${index}`}
                                <IconButton size="small" onClick={() => toggleExpand(index)} style={{ marginLeft: '8px' }}>
                                  {expandedIndex === index ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                                </IconButton>
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell style={{ padding: 0 }}>
                              <Collapse in={expandedIndex === index} timeout="auto" unmountOnExit>
                                <List dense style={{ paddingTop: 0, paddingBottom: 0 }}>
                                  {nodeid_color_mapping[index] && Object.entries(nodeid_color_mapping[index]).map(([neuronId, color]) => (
                                    <ListItem 
                                      key={index} 
                                      style={{ 
                                        backgroundColor: hexToRGBA(color, 0.8) || 'transparent', 
                                        paddingTop: 10, 
                                        paddingBottom: 10,
                                        justifyContent: 'center'
                                      }} 
                                    >
                                      <ListItemText 
                                        primary={`Neuron ID: ${neuronId}`} 
                                        style={{ textAlign: 'center', padding: 0, margin: 0, color: 'white' }}
                                      />
                                    </ListItem>
                                  ))}
                                </List>
                            </Collapse>
                              </TableCell>
                            </TableRow>
                          </React.Fragment>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </div>)}
              </div>
            </div>
           </div>
    );
});

export default { render };