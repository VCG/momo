import * as React from "react";
import {useState} from "react";
import { createRender, useModelState } from "@anywidget/react";
import {Sketch} from "@vimo-public/vimo-sketches";
import "./widget.css";

const render = createRender(() => {
	const [value, setValue] = useModelState("value");
	const [motif_json, setMotif_json] = useModelState("motif_json");
	// const [attributes, setAttributes] = useModelState({displayMotifCount: false});
	const [attributes, setAttributes] = useState({displayMotifCount: false});
    const processRequest = async (motifJson, lim) => {
      console.log("This function is called upon clicking the search button.");
      console.log(motifJson.edges);
      // try {
      //       const serializedMotifJson = JSON.stringify(motifJson);
            setMotif_json(motifJson.edges); // Save the serialized data in the widget state
            // setMotif_json({"a":"v"}); // Save the serialized data in the widget state
      //   } catch (error) {
      //       console.error("Error serializing motifJson:", error);
      //   }
      // Ensure comm is available and send motifJson in a structured message
};
	return (
		<div className="anywidget_test">
            <Sketch processRequest={processRequest} attributes={attributes} />
			{/* <button onClick={() => setValue(value + 1)}>
				count is {value}
			</button> */}
		</div>
	);
});

export default { render };