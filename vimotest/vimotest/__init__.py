import importlib.metadata
import pathlib
import pandas as pd
import numpy as np
import secrets
import hashlib

import anywidget
import traitlets

try:
    __version__ = importlib.metadata.version("vimotest")   
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class Vimotest(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "vimo.js"
    _css = pathlib.Path(__file__).parent / "static" / "vimo.css"
    
