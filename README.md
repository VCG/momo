[![Example](https://img.shields.io/badge/example-notebook-blue.svg?colorB=4AC8F4)](example_notebook.ipynb)
[![Data](https://img.shields.io/badge/data-gdrive-red.svg?colorB=f25100)](https://drive.google.com/drive/folders/15LN6hxLkhVxG4Oht5Ijj3IziPFeIrEqz?usp=sharing)

# 🧠 MoMo: Morphology-Aware Motif Analysis in Connectomes

**MoMo** is a toolkit for transforming raw neuron and synapse data into morphology-aware graphs.  
It supports scalable processing with [Arkouda](https://github.com/Bears-R-Us/arkouda) and interactive motif analysis through sketching and 3D visualization.

---

## 📦 Installation

To get started, clone the repository and install the required Python packages:

```bash
pip install pandas numpy networkx navis
```

Download the data from GDrive [here](https://drive.google.com/drive/folders/15LN6hxLkhVxG4Oht5Ijj3IziPFeIrEqz?usp=sharing).

---


## 📓 Example Notebook

You can find an example Jupyter notebook [`example_notebook.ipynb`](example_notebook.ipynb)

---

## ⚙️ Data Transformation Workflow

MoMo provides a pipeline to:
- Preprocess neuron and synapse data
- Map synapses using a segment-aware strategy
- Generate morphology-aware graphs

All these steps are demonstrated in the [`data_transformation_example.ipynb`](data_transformation_example.ipynb) notebook.

---

## 🧰 Arkouda Server Setup

To use MoMo with Arkouda for large-scale data processing, follow the official Arkouda server setup instructions here:  
👉 [Arkouda Setup Guide](https://github.com/Bears-R-Us/arkouda-njit/tree/main)

---

