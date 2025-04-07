import * as React from "react";
import { useState, useEffect } from 'react';
import { createRender, useModelState } from "@anywidget/react";
import "./widget.css";
import {SharkViewerComponent} from "skelescope2";

const render = createRender(() => {
const [neuronss, setNeuronss] = useModelState("neuronss");
const [synapsess, setSynapsess] = useModelState("synapsess");
const [dataset, setDataset] = useModelState("dataset");
    
const [open, setOpen] = React.useState(false);

  const rad = dataset === 'flywire' ? 500 : dataset === 'cave' ? 2 : null; 
  const synapse_color='#e41a1c'
    
    return (
		<div>
            <SharkViewerComponent neurons={neuronss} synapseLocations={synapsess} synapseRadius={rad} synapseColor ={synapse_color}/>
		</div>
	);
});

export default { render };
