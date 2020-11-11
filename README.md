Negation and Natural Language Inference
===================================================================
This repository contains code of the EMNLP2020 paper "An Analysis of Natural Language Inference Benchmarks through the Lens of Negation". Paper link: https://www.aclweb.org/anthology/2020.emnlp-main.732.pdf  
Authors: Md Mosharaf Hossain, Venelin Kovatchev, Pranoy Dutta, Tiffany Kao, Elizabeth Wei and Eduardo Blanco  

## Data Requirements
Download RTE, SNLI, and MNLI using "download_glue_data.py" script of https://github.com/nyu-mll/GLUE-baselines.
```bash
python ./data/download_glue_data.py --data_dir ./data/GLUE --tasks RTE,SNLI,MNLI
```

## Python Requirements
Python 3.6+ (Recommended: Python 3.7)  
Python packages: list of packages are provided in ./env-setup/requirements.txt file.  
(We used an older version of the huggingface transformers (version 2.1.1) package).

```bash
# Create virtual env (Assuming you have Python 3.7 installed in your machine) -> optional step
python3 -m venv your_location/negation-and-nli
source your_location/negation-and-nli/bin/activate

# Install required packages -> required step
pip install -r ./env-setup/requirements.txt
```

## Fine-tune Transformer systams and evaluate on the original dev splits
At the very begining, below directories need to be created. The predicted labels are saved in "outputs/predictions" directory and the fine-tuned models are saved in the "outputs/models" directory.
```bash
mkdir outputs
mkdir outputs/predictions
mkdir outputs/models
```

Fine-tune the transformers using RTE training split and evaluate on the RTE dev split:  
```bash
sh rte-train.sh
```
Fine-tune the transformers using SNLI training split and evaluate on the SNLI dev split:
```bash
sh snli-train.sh
```
Fine-tune the transformers using MNLI training split and evaluate on the MNLI dev split:
```bash
sh mnli-train.sh
```

## Evaluate the fine-tuned systems on new benchmarks containing negations
Evaluate on new RTE benchmark  
```bash
sh rte-evaluate.sh
```
Evaluate on new SNLI benchmark  
```bash
sh snli-evaluate.sh
```
Evaluate on new MNLI benchmark  
```bash
sh mnli-evaluate.sh
```



## Results
Results (Table 7 of our paper) can be achieved by the below script.
```bash
  python evaluate.py --corpus corpus_name
```
  + Arguments:
	  --corpus: Name of the corpus (e.g., rte or snli or mnli)
  

## New NLI benchmarks containing negation
The annotation files of the new NLI benchmarks containing negation are given below.  
RTE: ./data/new_benchmarks/clean_data/RTE.txt  
SNLI: ./data/new_benchmarks/clean_data/SNLI.txt  
MNLI: ./data/new_benchmarks/clean_data/MNLI.txt  


## Citation

Please cite our paper if the paper (and code) is useful.   
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
