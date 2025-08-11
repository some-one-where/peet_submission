
# Paper Submission – Data and Scorer

We provide the dataset, scripts, models, and evaluation code used in our paper:  
**Anonymized Paper Name - PEET Scorer**.

This repository contains:
- **Datasets/** - GEC Tool output for CONLL14 and BEA19 GEC Datasets along with the final Editor Corrections.
- **Models/** - (Statistical and Neural)  Regression for PEET Scorer.
- **Scripts/**  - Script to calculate regression coefficient and PEET Scorer tool to evaluate GEC Tools.
- **``README.md``** - File explaining all the Supplementary Material.

---

## 📂 Dataset

The `Dataset` folder contains text files with GEC Tool outputs and Editor Targetted Corrections for two test sets: 
- [CONLL14](https://www.comp.nus.edu.sg/~nlp/conll14st.html) : The CoNLL-2014 Shared Task on Grammatical Error Correction -  https://aclanthology.org/W14-1701/
- [BEA19](https://www.cl.cam.ac.uk/research/nl/bea2019st/) : The BEA-2019 Shared Task on Grammatical Error Correction - https://aclanthology.org/W19-4406/

The GEC Tools used for first-pass correction on the datasets were:
 - [GECToR](https://github.com/grammarly/gector) : GECToR – Grammatical Error Correction: Tag, Not Rewrite - https://aclanthology.org/2020.bea-1.16/
 - [GEC-PD](https://github.com/butsugiri/gec-pseudodata) : An Empirical Study of Incorporating Pseudo Data into Grammatical Error Correction - https://www.aclweb.org/anthology/D19-1119

For each dataset, we provide the following files:

- `source_EC.txt` – Editor Correction for the dataset source sentence.
- `gector_OP.txt` – GECToR Tool output (first-pass correction).
- `gector_EC.txt` – Targeted Editor Corrections for GECToR Tool output.
- `gecpd_OP.txt` – GEC-PD Tool output (first-pass correction).
- `gecpd_EC.txt` – Targeted Editor Corrections for GEC-PD Tool output.

**Additional Training Pandas DataFrame:**
- `combined_dataframe.pkl` – Preprocessed DataFrame combining all datasets and Tool corrections, Editor Corrections, Edit Type and Counts, Sentence Features and Time to Correct value for model training/evaluation.
---
## 📂 Models (`small/medium/large`)
We list the files to train different PEET Scorer models.

### **1. Statistical Model**
- **File:** [`stat_regression.py`](./Models/stat_regression.py)  
- **Description:** Implements a Linear-Ridge and SVR based regression models using different extracted features (edit-types + sentence structure features). (small = 4, medium = 25, or, large = 55 Edit Types)
- **Requirements:** [`req_python3.12.10.txt`](./Models/req_python3.12.10.txt)
- **Example Run:**
```
>>> python stat_regression.py --feature_set small/medium/large
```

### **2. Neural Model** (`Models/Neural Model/`)
-   **Files:**
    -   `bert.py` – Training file for BERT/RoBERTa models using Sentence Edit and POS-Syntax features.
    -   `bert_syntax_parse.py` – Training file for BERT/RoBERTa models using Parse Tree Syntactic Variation features.
    -   `eval.py` – Evaluate and print the results saved from the training run of `bert.py`
    -   `req_python3.11.3.txt` – Python Installation dependencies
 - **Example Run:**
 ```
>>> CUDA_VISIBLE_DEVICES=0 python bert.py --data_type mo_trg --model_type bert-large-cased
--data_path ../Dataset/combined_dataframe.pkl --model_path <location of bert/roberta models>
--num_epochs 50 --batch_size 32 --grad_acc_steps 1

>>> CUDA_VISIBLE_DEVICES=1 python bert_syntax_parse.py --data_type syntax_word --model_type roberta-large
--data_path ../Dataset/combined_dataframe.pkl --model_path <location of bert/roberta models>
--num_epochs 30 --batch_size 4 --grad_acc_steps 3
 ```

---

## 📂 Scripts

The **`Scripts/`** folder contains utilities for scoring and evaluating GEC Tools and studying the impact of edit features on time-to-correct.
### **1 Regression Coefficient Calculation**
- **`calc_regression_coefficients.py`** – Prints the sorted regression coefficients for the passed Linear model.
- **Example Run:**
```>>> python calc_regression_coefficients.py --model Scripts/PEET_Scorer/modelLR.sav```

### **2 PEET Scorer** (`Scripts/PEET_Scorer/`)
- **`M2/`, `MO/`, `REF/`** – Folder to copy all target reference and hypothesis files for M2 scoring.
- **`genM2.py`** – Generates ERRANT M2 format files for all hypothesis-reference pairs and saves them in `M2/`.
- **`peet_scorer.py`** – Estimates the average PEET Score for each hypothesis GEC Tool output file.
- **`modelLR.sav`** – Pre-trained regression model weights for PEET Scorer.
- **`wer.py`** - Computes the average WER for each hypothesis file used in the paper Human Judgment Ranking comparison.
- **Example Run:**
```
Instructions : 
	1)Copy all GEC Tool Corrections to be evaluated in MO/ folder
	2)Copy all available dataset Target Refereces in the REF/ folder. Ideally these should be Post-Edited Corrections.
>>> python genM2.py
>>> python peet_scorer.py
```
---
## 📊 Human Judgment Ranking Datasets

Please find and download the GEC Evaluation Datasets used in this paper and for GEC Tool evaluation at the following location.

- [Grundkiewicz-C14(EW)](https://github.com/grammatical/evaluation) : Human Evaluation of Grammatical Error Correction Systems - https://aclanthology.org/D15-1052/
- [SEEDA-C14-All(TS)](https://github.com/tmu-nlp/SEEDA) : Revisiting Meta-evaluation for Grammatical Error Correction - https://aclanthology.org/2024.tacl-1.47/
- [Napoles-FCE / Napoles-Wiki](https://github.com/grammarly/GMEG) : Enabling Robust Grammatical Error Correction in New Domains: Data Sets, Metrics, and Analyses - https://github.com/grammarly/GMEG

---


        
