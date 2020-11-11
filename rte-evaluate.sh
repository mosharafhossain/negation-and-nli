#!/bin/sh

# remove old dev files 
rm -rf ./data/GLUE/RTE/cached_dev_*
rm -rf ./data/GLUE/RTE/dev*

# copy the new benchmark and save to appropriate location 
cp ./data/new_benchmarks/processed_for_run/RTE/dev.tsv ./data/GLUE/RTE/dev.tsv

# create directories that contain predicted labels (as well as actual labels)
mkdir outputs/predictions/RTE/RoBERTa/new_dev
mkdir outputs/predictions/RTE/XLNet/new_dev
mkdir outputs/predictions/RTE/BERT/new_dev

# export task directory and name
export GLUE_DIR=./data/GLUE/
export TASK_NAME=RTE


# Using RoBERTa 
# Evaluate on the new pairs containing negation
export PRED_DIR=./outputs/predictions/RTE/RoBERTa/new_dev/
python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path roberta-base --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 16 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 10.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/


# Using XLNet
# Evaluate on the new pairs containing negation
export PRED_DIR=./outputs/predictions/RTE/XLNet/new_dev/
python ./transformers/examples/run_glue.py --model_type xlnet --model_name_or_path xlnet-base-cased --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 8 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 50.0 --output_dir ./outputs/models/$TASK_NAME/XLNet/


# Using BERT
# Evaluate on the new pairs containing negation
export PRED_DIR=./outputs/predictions/RTE/BERT/new_dev/
python ./transformers/examples/run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 8 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 50.0 --output_dir ./outputs/models/$TASK_NAME/BERT/
