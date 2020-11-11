#!/bin/sh

# remove old dev files 
rm -rf ./data/GLUE/MNLI/cached_dev_*
rm -rf ./data/GLUE/MNLI/dev*

# copy the new benchmark and save to appropriate location 
cp ./data/new_benchmarks/processed_for_run/MNLI/dev_matched.tsv ./data/GLUE/MNLI/dev_matched.tsv
# We haven't worked with mismatched genres. Just just made a duplicate copy of dev_matched.tsv file and named as dev_mismatched.tsv for running the systems successfully.
cp ./data/new_benchmarks/processed_for_run/MNLI/dev_mismatched.tsv ./data/GLUE/MNLI/dev_mismatched.tsv

# create directories that contain predicted labels (as well as actual labels)
mkdir outputs/predictions/MNLI/RoBERTa/new_dev
mkdir outputs/predictions/MNLI/XLNet/new_dev
mkdir outputs/predictions/MNLI/BERT/new_dev

# export task directory and name
export GLUE_DIR=./data/GLUE/
export TASK_NAME=MNLI



# Using RoBERTa 

# Evaluate on the new pairs containing negation
export PRED_DIR=./outputs/predictions/MNLI/RoBERTa/new_dev/
python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path roberta-base --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/


# Using XLNet
# Evaluate on the new pairs containing negation
export PRED_DIR=./outputs/predictions/MNLI/XLNet/new_dev/
python ./transformers/examples/run_glue.py --model_type xlnet --model_name_or_path xlnet-base-cased --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/XLNet/


# Using BERT
# Evaluate on the new pairs containing negation
export PRED_DIR=./outputs/predictions/MNLI/BERT/new_dev/
python ./transformers/examples/run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/BERT/

