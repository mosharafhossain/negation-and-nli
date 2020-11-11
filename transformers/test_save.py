# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:18:19 2019

@author: Mosharaf
"""

"""
import numpy as np
pred = np.array([ 1,0,1,0,1,1])
#pred = np.array(["a","b","c"])
task_name = "snli"
data_dir = "./predictions/"
np.savetxt(data_dir+task_name+"_output.csv", pred, fmt='%s')
"""


import numpy as np
#pred = np.array([ 1,0,1,0,1,1])
pred = np.array(["a","b","c"])
task_name = "snli"
data_dir = "/home/mh0826/Research/Bitbucket/benchmarks_negation/benchmarks_negation/code/external/transformers/results/predictions/"
np.savetxt(data_dir+task_name+"_output.csv", pred, fmt='%s')