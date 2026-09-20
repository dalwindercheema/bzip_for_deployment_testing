# bPPI: bZIP Protein-Protein Interaction Predictor

**bPPI** is a deep-learning model to predict bZIP protein-protein interaction. 

## Command Line Interface (CLI)

### Step 1: Download Model Architecture
Download model weights from the Hugging Face registry:

```bash
# Options: "General", "Temperature4C", "Temperature21C", "Temperature37C"
python predict.py --download "Temperature37C"
```

### Step 2: Evaluate Sequence Interaction Pairs
```bash
python predict.py --seqA path/to/protein1.fasta --seqB path/to/protein2.fasta
```

### Available Model Variations

Every prediction query targets the structural layers defined within `motifDectector.pt` alongside one environment-specific system matrix:

| Model Type | Checkpoint name | 
| :--- | :--- |
| **`General`** | `GeneralInteraction.pt` | 
| **`Temperature 4C`** | `Temperature_4.pt` | 
| **`Temperature 21C`** | `Temperature_21.pt` | 
| **`Temperature 37C`** | `Temperature_37.pt` | 

---

## Google Colab

bPPI is also available online through Google Colaboratory.  
No installation required!

👉 **Launch it here**: [bPPI@Google-Colab](https://colab.research.google.com/drive/1WGjaFLSPk-Y5mipuuJ6ki5FY5xPMrJEH#scrollTo=sVnTq1QQXF_C)

----


## Citation

If you use bPPI in your research, please cite:

----
