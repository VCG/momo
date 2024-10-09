import {Sketch} from "@vimo-public/vimo-sketches";
import {useState} from "react";

function App() {
    const [attributes, setAttributes] = useState({
      displayMotifCount: false,
    });

    const processRequest = async (motifJson, lim) => {
        console.log("This function is called upon clicking the search button.");
    };
    return (
        <div>
            <Sketch processRequest={processRequest} attributes={attributes} />
        </div>
    );
}
export default App;