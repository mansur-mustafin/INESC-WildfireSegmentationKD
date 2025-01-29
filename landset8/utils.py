import os
import random

import numpy as np
import torch

def eval_image(predict,label,num_classes):
    """
    Copiado de Sen2Fire/utils/tools.py
    """
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index]
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)


def init_seeds(seed=1234):
    """
    Copiado de Sen2Fire/Train.py
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


def get_model_paths(dir, model_name):
    """
    Returns all model paths ordered by epoch number
    """
    paths = [
            os.path.join(root, file) 
            for root, _, files in os.walk(dir) 
            for file in files if file.endswith('.pt')
        ]
    
    return sorted(paths, key=lambda name: int(os.path.basename(name)[len(model_name):].split('_')[1]))
