Negation and NLI
===================================================================

## Data Requirements
Download RTE, SNLI, and MNLI using "download_glue_data.py" script of https://github.com/nyu-mll/GLUE-baselines.
```bash
python ./data/download_glue_data.py --data_dir ./data/GLUE --tasks RTE,SNLI,MNLI
```bash

## Python Requirements
Python 3.6+ (Recommended Python 3.7)  
Python packages: list of packages are provided in ./env-setup/requirements.txt file.  
(We used an older version of the huggingface transformers (version 2.1.1)).

```bash
# Create virtual env (Assuming you have Python 3.7 installed in your machine) -> optional step
python3 -m venv your_location/negation-and-nli
source your_location/negation-and-nli/bin/activate

# Install required packages -> required step
pip install -r ./env-setup/requirements.txt
```

## Fine-tune Transformers and evaluate on original dev splits
At first, we need to create below directories. The predicted labels are saved in "outputs/predictions" and the fine-tuned models are saved in "utputs/models" locations.
```bash
mkdir outputs
mkdir outputs/predictions
mkdir outputs/models
```

Fine-tune transformers using RTE training split and evaluate on RTE dev split:  
```bash
sh rte-train.sh
```
Fine-tune transformers using SNLI:  
```bash
sh snli-train.sh
```
Fine-tune transformers using MNLI:  
```bash
sh mnli-train.sh
```

## Evaluate the fine-tuned systems on new benchmarks containing negations
# Evaluate on new RTE benchmark 
```bash
sh rte-evaluate.sh
```
# Evaluate on new SNLI benchmark 
```bash
sh snli-evaluate.sh
```
# Evaluate on new MNLI benchmark 
```bash
sh mnli-evaluate.sh
```



## Train/evaluate RoBERTa on SNLI
Get the results (Table 7 in the paper)
```bash
  python evaluate.py --corpus corpus_name
```
  + Arguments:
	  --corpus: Name of the corpus to show results (e.g., rte, snli, and mnli), (required)
  

## Citation

Please cite our paper if the paper (and code) is useful to you. Paper title: "An Analysis of Natural Language Inference Benchmarks through the Lens of Negation".  
```bibtex
@inproceedings{hossain-etal-2020-analysis,
    title = "An Analysis of Natural Language Inference Benchmarks through the Lens of Negation",
    author = "Hossain, Md Mosharaf  and
      Kovatchev, Venelin  and
      Dutta, Pranoy  and
      Kao, Tiffany  and
      Wei, Elizabeth  and
      Blanco, Eduardo",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.732",
    pages = "9106--9118",
}
```
