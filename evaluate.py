# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 00:17:16 2019

@author: Mosharaf
"""
import pickle
import pandas as pd
from collections import Counter
import argparse

class data_structure(): 
    def __init__(self, line, indx_sent1, indx_sent2, indx_label):
        tokens = line.split("\t")              
        self.sentence1 = tokens[indx_sent1]
        self.sentence2 = tokens[indx_sent2]
        self.label = tokens[indx_label][0:-1]


class data_preparation():
    def data_load(self, file, indx_sent1, indx_sent2, indx_label, file_type = "text"):
        reader = open(file, "r", encoding="utf8")
        obj_list = []
        isFirstLine = True # set isFirstLine = True if need to ignore the first line, because first line might contain column name.
        for line in reader:
            if not isFirstLine: # not reading the first line. becuase this line reads the column names.
                if file_type == "text":
                    obj = data_structure(line, indx_sent1, indx_sent2, indx_label)
                obj_list.append(obj)
            else:
                isFirstLine = False
        reader.close()
        return obj_list
    
    def write_data(self, file_name, inst_list, indices):
        file_obj = open(file_name, "w", encoding="utf-8")
        line = ''
        delim = "\t"
        for i in indices:
            line = inst_list[i].sentence1 + delim + inst_list[i].sentence2 + delim +inst_list[i].label                        
            file_obj.write(line)
            file_obj.write("\n")
        file_obj.close()
        print("File is created at {}".format(file_name))



class negation_cues():
    def isCuePresent(self, sent):
        for token in sent:
            if token == "PAD":
                return False
            if token not in ( "N_C", "PAD"):
                return True
        return False
        
    def get_cue_indices(self, cues_dict):
        sent1_pred = cues_dict["pred_sent1"]
        sent2_pred = cues_dict["pred_sent2"]
        cues_only_sent1 = []
        cues_only_sent2 = []
        cues_both_sent  = []
        size = len(sent1_pred)
        for i in range(size):
            if self.isCuePresent(sent1_pred[i]) and self.isCuePresent(sent2_pred[i]):
                cues_both_sent.append(i)
            elif self.isCuePresent(sent1_pred[i]):
                cues_only_sent1.append(i)
            elif self.isCuePresent(sent2_pred[i]):
                cues_only_sent2.append(i)
        cues_all = sorted( list(set(cues_only_sent1+cues_only_sent2+cues_both_sent)) ) #instances with any cues in any sentences.
        no_cues  = sorted( list( set(list(range(size)))-set(cues_all) ) ) # instances with no cues in any sentence
                
        return cues_only_sent1, cues_only_sent2, cues_both_sent, cues_all, no_cues
    

class evaluation():
    def accuracy(self, actual_df, pred_df, indicies):
        size = len(indicies)
        correct = 0
        for idx in indicies:
            if actual_df.iloc[idx,0] == pred_df.iloc[idx,0]:
                correct += 1
        return correct*100.0/size if size>0 else -1
    
    def majority_baseline(self, actual_df, indicies):
        size = len(indicies)
        if size == 0:
            return -1, -1
        orig_labels = []
        for idx in indicies:
            orig_labels.append(actual_df.iloc[idx,0])
        label_dist = Counter(orig_labels)
        
        
        
        max_val = float("-inf")        
        for key, val in label_dist.items():
            if val > max_val:
                max_val = val
        majority_bl = round( (max_val*100.0)/size,2)
        
        return majority_bl, label_dist
        
    
    def accuracy_org_corpus(self, model_name, actual_file, pred_file, cues_indices_dict):
        cues_only_sent1  = cues_indices_dict["cues_only_sent1"] 
        cues_only_sent2  = cues_indices_dict["cues_only_sent2"] 
        cues_both_sent   = cues_indices_dict["cues_both_sent"] 
        cues_all         = cues_indices_dict["cues_all"] 
        no_cues          = cues_indices_dict["no_cues"]
        #print("cues_only_sent1:{}, cues_only_sent2: {}, cues_both_sent: {} cues_all: {}".format(len(cues_only_sent1), len(cues_only_sent2), len(cues_both_sent), len(cues_all), len(no_cues) ))
        
        
        print("Accuracy by " + model_name + "______________")
        actual_df = pd.read_csv(actual_file, header = None) 
        pred_df = pd.read_csv(pred_file, header = None) 
        accuracy = self.accuracy(actual_df, pred_df, list(range(pred_df.size)))
        majority_bl, label_dist = self.majority_baseline(actual_df, list(range(pred_df.size)))
        print("Accuracy on Dev:   {}".format( accuracy ))
        print("Majority baseline: {}\n".format(majority_bl))
        
        accuracy_sent1 = self.accuracy(actual_df, pred_df, cues_only_sent1)
        majority_bl, label_dist = self.majority_baseline(actual_df, cues_only_sent1)
        #print("Accuracy-premise is neagated: {}. Number of instances: {}".format( accuracy_sent1, len(cues_only_sent1)))
        #print("Majority b: {}, label dist. {}".format(majority_bl, label_dist))
        
        accuracy_sent2 = self.accuracy(actual_df, pred_df, cues_only_sent2)
        majority_bl, label_dist = self.majority_baseline(actual_df, cues_only_sent2)
        #print("Accuracy-hypothesis is neagated: {}. Number of instances: {}".format( accuracy_sent2, len(cues_only_sent2)))
        #print("Majority b: {}, label dist. {}".format(majority_bl, label_dist))
        
        accuracy_both = self.accuracy(actual_df, pred_df, cues_both_sent)
        majority_bl, label_dist = self.majority_baseline(actual_df, cues_both_sent)
        #print("Accuracy-both neagated: {}. Number of instances: {}".format( accuracy_both, len(cues_both_sent)))
        #print("Majority b: {}, label dist. {}".format(majority_bl, label_dist))
        
        accuracy_all_cues = self.accuracy(actual_df, pred_df, cues_all)
        majority_bl, label_dist = self.majority_baseline(actual_df, cues_all)
        print("Accuracy on Dev (negations): {}".format( accuracy_all_cues))
        print("Majority baseline: {}\n".format(majority_bl))
        
        accuracy_no_cues = self.accuracy(actual_df, pred_df, no_cues)
        majority_bl, label_dist = self.majority_baseline(actual_df, no_cues)
        #print("Accuracy-with no negation: {}. Number of instances: {}".format( accuracy_no_cues, len(no_cues)))
        #print("Majority b: {}, label dist. {}".format(majority_bl, label_dist))
        
        
    def accuracy_new_corpus(self, model_name, actual_file, pred_file, neg_cues_dict):
        sent1_negated_indices = neg_cues_dict["sent1_negated_indices"]
        sent2_negated_indices = neg_cues_dict["sent2_negated_indices"]
        both_negated_indices  = neg_cues_dict["both_negated_indices"]
        
        
        print("Accuracy by " + model_name +"______________")
        actual_df = pd.read_csv(actual_file, header = None) 
        pred_df = pd.read_csv(pred_file, header = None) 
        accuracy_all = self.accuracy(actual_df, pred_df, list(range(pred_df.size)))
        majority_bl_all, label_dist = self.majority_baseline(actual_df, list(range(pred_df.size)))        
        
        accuracy_sent1 = self.accuracy(actual_df, pred_df, sent1_negated_indices)
        majority_bl, label_dist = self.majority_baseline(actual_df, sent1_negated_indices)
        print("Accuracy on Tneg-H pairs: {}".format( accuracy_sent1))
        print("Majority baseline: {}\n".format(majority_bl))
        
        accuracy_sent2 = self.accuracy(actual_df, pred_df, sent2_negated_indices)
        majority_bl, label_dist = self.majority_baseline(actual_df, sent2_negated_indices)
        print("Accuracy on T-Heng pairs: {}".format( accuracy_sent2))
        print("Majority baseline: {}\n".format(majority_bl))
        
        accuracy_both = self.accuracy(actual_df, pred_df, both_negated_indices)
        majority_bl, label_dist = self.majority_baseline(actual_df, both_negated_indices)
        print("Accuracy on Tneg-Hneg pairs: {}".format( accuracy_both))
        print("Majority baseline: {}\n".format(majority_bl))
        
        print("Accuracy on all new pairs: {}".format( accuracy_all ))
        print("Majority baseline: {}\n".format(majority_bl_all))



if __name__ == "__main__":
    
    #python evaluate.py --corpus rte
    argParser   = argparse.ArgumentParser()
    argParser.add_argument("--corpus", help="Name of the corpus to show results (e.g., rte, snli, mnli)", required=True)      
    args        = argParser.parse_args()
    corpus      = args.corpus
    
    if corpus == "rte":
        #Read negation in original dev split
        file_path = "./data/resources/RTE/negation_indices.pkl"
        with open(file_path, "rb") as file_obj:
            dev_cues = pickle.load(file_obj)  
        cues_only_sent1, cues_only_sent2, cues_both_sent, cues_all, no_cues = negation_cues().get_cue_indices(dev_cues)
        cues_indices_dict = {"cues_only_sent1":cues_only_sent1, "cues_only_sent2":cues_only_sent2, "cues_both_sent":cues_both_sent, "cues_all":cues_all, "no_cues":no_cues}
        
        print("\nStarted: Evaluation on original dev split----------------------------------------------------------")
        # RoBERTa
        model_name    = "RoBERTa"
        actual_file   = "./outputs/predictions/RTE/RoBERTa/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/RTE/RoBERTa/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        
        
        # XLNet
        model_name    = "XLNet"
        actual_file   = "./outputs/predictions/RTE/XLNet/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/RTE/XLNet/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        
        # BERT
        model_name    = "BERT"
        actual_file   = "./outputs/predictions/RTE/BERT/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/RTE/BERT/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        print("Ended: Evaluation on original dev split----------------------------------------------------------")
        
        
        print("\n\nStarted: Evaluation on new pairs containing negation----------------------------------------------------------")
        #____________________________________SNLI(Generated Data)___________________________________________________________________________
        file_path = "./data/new_benchmarks/resources/RTE/negated_indices.pkl"
        with open(file_path, "rb") as file_obj:
            neg_cues_dict = pickle.load(file_obj)  #keys: tr_pred_scope,te_pred_scope
    
        # RoBERTa
        model_name  = "RoBERTa"
        actual_file   = "./outputs/predictions/RTE/RoBERTa/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/RTE/RoBERTa/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        
        
        # XLNet
        model_name  = "XLNet"
        actual_file   = "./outputs/predictions/RTE/XLNet/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/RTE/XLNet/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        
        
        # BERT
        model_name  = "BERT"
        actual_file   = "./outputs/predictions/RTE/BERT/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/RTE/BERT/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        print("Ended: Evaluation on new pairs containing negation----------------------------------------------------------")

    
    elif corpus == "snli":
        #Read negation in original dev split
        file_path = "./data/resources/SNLI/negation_indices.pkl"
        with open(file_path, "rb") as file_obj:
            dev_cues = pickle.load(file_obj)  
        cues_only_sent1, cues_only_sent2, cues_both_sent, cues_all, no_cues = negation_cues().get_cue_indices(dev_cues)
        cues_indices_dict = {"cues_only_sent1":cues_only_sent1, "cues_only_sent2":cues_only_sent2, "cues_both_sent":cues_both_sent, "cues_all":cues_all, "no_cues":no_cues}
        
        print("\nStarted: Evaluation on original dev split----------------------------------------------------------")
        # RoBERTa
        model_name    = "RoBERTa"
        actual_file   = "./outputs/predictions/SNLI/RoBERTa/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/SNLI/RoBERTa/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        
        
        # XLNet
        model_name    = "XLNet"
        actual_file   = "./outputs/predictions/SNLI/XLNet/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/SNLI/XLNet/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        
        # BERT
        model_name    = "BERT"
        actual_file   = "./outputs/predictions/SNLI/BERT/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/SNLI/BERT/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        print("Ended: Evaluation on original dev split----------------------------------------------------------")
        
        
        print("\n\nStarted: Evaluation on new pairs containing negation----------------------------------------------------------")
        #____________________________________SNLI(Generated Data)___________________________________________________________________________
        file_path = "./data/new_benchmarks/resources/SNLI/negated_indices.pkl"
        with open(file_path, "rb") as file_obj:
            neg_cues_dict = pickle.load(file_obj)  #keys: tr_pred_scope,te_pred_scope
    
        # RoBERTa
        model_name  = "RoBERTa"
        actual_file   = "./outputs/predictions/SNLI/RoBERTa/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/SNLI/RoBERTa/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        
        
        # XLNet
        model_name  = "XLNet"
        actual_file   = "./outputs/predictions/SNLI/XLNet/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/SNLI/XLNet/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        
        
        # BERT
        model_name  = "BERT"
        actual_file   = "./outputs/predictions/SNLI/BERT/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/SNLI/BERT/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        print("Ended: Evaluation on new pairs containing negation----------------------------------------------------------")

    elif corpus == "mnli":
        #Read negation in original dev split
        file_path = "./data/resources/MNLI/negation_indices.pkl"
        with open(file_path, "rb") as file_obj:
            dev_cues = pickle.load(file_obj)  
        cues_only_sent1, cues_only_sent2, cues_both_sent, cues_all, no_cues = negation_cues().get_cue_indices(dev_cues)
        cues_indices_dict = {"cues_only_sent1":cues_only_sent1, "cues_only_sent2":cues_only_sent2, "cues_both_sent":cues_both_sent, "cues_all":cues_all, "no_cues":no_cues}
        
        print("\nStarted: Evaluation on original dev split----------------------------------------------------------")
        # RoBERTa
        model_name    = "RoBERTa"
        actual_file   = "./outputs/predictions/MNLI/RoBERTa/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/MNLI/RoBERTa/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        
        
        # XLNet
        model_name    = "XLNet"
        actual_file   = "./outputs/predictions/MNLI/XLNet/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/MNLI/XLNet/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        
        # BERT
        model_name    = "BERT"
        actual_file   = "./outputs/predictions/MNLI/BERT/original_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/MNLI/BERT/original_dev/rte_prediction.csv"
        evaluation().accuracy_org_corpus(model_name, actual_file, pred_file, cues_indices_dict)
        print("Ended: Evaluation on original dev split----------------------------------------------------------")
        
        
        print("\n\nStarted: Evaluation on new pairs containing negation----------------------------------------------------------")
        #____________________________________SNLI(Generated Data)___________________________________________________________________________
        file_path = "./data/new_benchmarks/resources/MNLI/negated_indices.pkl"
        with open(file_path, "rb") as file_obj:
            neg_cues_dict = pickle.load(file_obj)  #keys: tr_pred_scope,te_pred_scope
    
        # RoBERTa
        model_name  = "RoBERTa"
        actual_file   = "./outputs/predictions/MNLI/RoBERTa/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/MNLI/RoBERTa/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        
        
        # XLNet
        model_name  = "XLNet"
        actual_file   = "./outputs/predictions/MNLI/XLNet/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/MNLI/XLNet/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        
        
        # BERT
        model_name  = "BERT"
        actual_file   = "./outputs/predictions/MNLI/BERT/new_dev/rte_actuals.csv"
        pred_file     = "./outputs/predictions/MNLI/BERT/new_dev/rte_prediction.csv"
        evaluation().accuracy_new_corpus(model_name, actual_file, pred_file, neg_cues_dict)
        print("Ended: Evaluation on new pairs containing negation----------------------------------------------------------")


