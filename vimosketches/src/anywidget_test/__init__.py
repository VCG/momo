import importlib.metadata
import pathlib

import anywidget
import traitlets

try:
    __version__ = importlib.metadata.version("anywidget_test")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class Widget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    value = traitlets.Int(0).tag(sync=True)

    # Create a traitlet to hold the motifJson data
    motif_json = traitlets.List([]).tag(sync=True)
    
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     # Register a message handler for comm messages
    #     self.comm.on_msg(self.handle_message)
    
    # def handle_message(self, msg):
    # # Extract the motifJson from the correct path in the received message
    #     if "data" in msg["content"] and "motifJson" in msg["content"]["data"]:
    #         self.motif_json = msg["content"]["data"]["motifJson"]

