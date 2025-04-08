# 🧠 MoMo: Morphology-Aware Motif Analysis in Connectomes

**MoMo** is a Python toolkit for transforming raw neuron and synapse data into morphology-aware graphs.  
It supports scalable processing with [Arkouda](https://github.com/Bears-R-Us/arkouda), integrates with [Navis](https://github.com/navis-org/navis) for neuron morphologies, and supports synapse-aware connectomics workflows.

---

## 📦 Installation

To get started, clone the repository and install the required Python packages:

```bash
pip install pandas numpy networkx navis
```

---

## 📓 Example Notebook

You can find an example Jupyter notebook in the `example_notebook/` directory:

📁 [`data_transformation_example.ipynb`](data_transformation_example.ipynb)

This notebook demonstrates how to:
- Load and process neuron and synapse data
- Map synapses to neuron segments
- Build a neuron morphology-aware graph with MoMo

---

## ⚙️ Data Transformation Workflow

MoMo provides a pipeline to:
- Preprocess neuron and synapse data
- Map synapses using a segment-aware strategy
- Generate morphology-aware graphs, with or without synaptic connections

All these steps are demonstrated in the [`data_transformation_example.ipynb`] notebook.

---

## 🧰 Arkouda Server Setup

To use MoMo with Arkouda for large-scale data processing, follow the official Arkouda server setup instructions here:  
👉 [Arkouda Setup Guide](https://github.com/Bears-R-Us/arkouda-njit/tree/main)

---

## 📹 Demo Video

Watch a full walkthrough of MoMo in action here:  
🎥 [Demo Video](https://placeholder.link/to-demo-video)

---

## 📋 Requirements

- `pandas`
- `numpy`
- `networkx`
- `navis`
- `arkouda-client`
- `arachne-tools`

Install all dependencies with:

```bash
pip install pandas numpy networkx navis arkouda-client arachne-tools

