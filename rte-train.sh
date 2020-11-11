#!/bin/sh

# create directories that contain predicted labels (as well as actual labels)
mkdir outputs/predictions/RTE
mkdir outputs/predictions/RTE/RoBERTa
mkdir outputs/predictions/RTE/XLNet
mkdir outputs/predictions/RTE/BERT
mkdir outputs/predictions/RTE/RoBERTa/original_dev
mkdir outputs/predictions/RTE/XLNet/original_dev
mkdir outputs/predictions/RTE/BERT/original_dev

# create directories that contain fine-tuned models
mkdir outputs/models/RTE
mkdir outputs/models/RTE/RoBERTa
mkdir outputs/models/RTE/XLNet
mkdir outputs/models/RTE/BERT

# export task directory and name
export GLUE_DIR=./data/GLUE/
export TASK_NAME=RTE



# Using RoBERTa 
#Train on the original training split and evaluate on the original dev split:  
export PRED_DIR=./outputs/predictions/RTE/RoBERTa/original_dev/
python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path roberta-base --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 16 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 10.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/


# Using XLNet
#Train on the original training split and evaluate on the original dev split:  
export PRED_DIR=./outputs/predictions/RTE/XLNet/original_dev/
python ./transformers/examples/run_glue.py --model_type xlnet --model_name_or_path xlnet-base-cased --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 8 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 50.0 --output_dir ./outputs/models/$TASK_NAME/XLNet/


# Using BERT
#Train on the original training split and evaluate on the original dev split:  
export PRED_DIR=./outputs/predictions/RTE/BERT/original_dev/
python ./transformers/examples/run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 8 --save_steps 20000 --learning_rate 2e-5 --num_train_epochs 50.0 --output_dir ./outputs/models/$TASK_NAME/BERT/

