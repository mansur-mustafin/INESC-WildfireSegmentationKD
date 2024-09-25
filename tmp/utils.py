import numpy as np
import torch
import random

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
