#!/bin/sh

# create directories that contain predicted labels (as well as actual labels)
mkdir outputs/predictions/SNLI
mkdir outputs/predictions/SNLI/RoBERTa
mkdir outputs/predictions/SNLI/XLNet
mkdir outputs/predictions/SNLI/BERT
mkdir outputs/predictions/SNLI/RoBERTa/original_dev
mkdir outputs/predictions/SNLI/XLNet/original_dev
mkdir outputs/predictions/SNLI/BERT/original_dev

# create directories that contain fine-tuned models
mkdir outputs/models/SNLI
mkdir outputs/models/SNLI/RoBERTa
mkdir outputs/models/SNLI/XLNet
mkdir outputs/models/SNLI/BERT

# export task directory and name
export GLUE_DIR=./data/GLUE/
export TASK_NAME=SNLI


# Using RoBERTa 
#Train on the original training split and evaluate on the original dev split:  
export PRED_DIR=./outputs/predictions/SNLI/RoBERTa/original_dev/
python ./transformers/examples/run_glue.py --model_type roberta --model_name_or_path roberta-base --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 1e-5 --weight_decay 0.1 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/RoBERTa/


# Using XLNet
#Train on the original training split and evaluate on the original dev split:  
export PRED_DIR=./outputs/predictions/SNLI/XLNet/original_dev/ 
python ./transformers/examples/run_glue.py --model_type xlnet --model_name_or_path xlnet-base-cased --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 1e-5 --weight_decay 0.1 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/XLNet/


# Using BERT
#Train on the original training split and evaluate on the original dev split:  
export PRED_DIR=./outputs/predictions/SNLI/BERT/original_dev/
python ./transformers/examples/run_glue.py --model_type bert --model_name_or_path bert-base-cased --task_name $TASK_NAME --do_train --do_eval --do_lower_case --data_dir $GLUE_DIR/$TASK_NAME --max_seq_length 128 --per_gpu_train_batch_size 32 --save_steps 20000 --learning_rate 1e-5 --weight_decay 0.1 --num_train_epochs 3.0 --output_dir ./outputs/models/$TASK_NAME/BERT/
