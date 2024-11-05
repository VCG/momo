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
	const [node_mapping, setNode_mapping] = useModelState("node_mapping");
	const [current_mapping, setCurrent_mapping] = useModelState("current_mapping");
	const [nodeid_color_mapping, setNodeid_color_mapping] = useModelState("nodeid_color_mapping");
  const [selectedIndex, setSelectedIndex] = useModelState("selectedIndex");
  
  const [expandedIndex, setExpandedIndex] = useState(null);
	const [attributes, setAttributes] = useState({displayMotifCount: false});

  const sketch_colors = ["#000000", "#00880A", "#003090", "#028785", "#c2c800"];
  const transformed_colors = ["#c9c9c9", "#8cfa94", "#87afff", "#99f7f6", "#fcff9e"];

  const color_transformation = sketch_colors.reduce((acc, color, index) => {
    acc[color] = transformed_colors[index];
    return acc;
  }, {});

  const toggleExpand = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index); // Toggle the expanded state
  };

  useEffect(() => {
  }, [node_mapping]);

  const processRequest = async (motifJson, lim) => {
      console.log("This function is called upon clicking the search button.");
      console.log(motifJson.edges);
      setMotif_json(motifJson.edges); 
  };

	return (
          <div>
            <div style={{ display: 'flex', flexDirection: 'row', justifyContent: 'center'}}>
              <div style={{width: "1000px"}}>
                <Sketch processRequest={processRequest} attributes={attributes} />
              </div>
              <div style={{width: "300px", marginRight:"20px"}}>
                {node_mapping.length === 0 ? (<div/>):(<div>
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
                                        backgroundColor: color_transformation[color] || 'transparent', 
                                        paddingTop: 10, 
                                        paddingBottom: 10,
                                        justifyContent: 'center'
                                      }} 
                                    >
                                      <ListItemText 
                                        primary={`Neuron ID: ${neuronId}`} 
                                        style={{ textAlign: 'center', padding: 0, margin: 0 }}
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