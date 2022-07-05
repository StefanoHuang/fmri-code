import os
import re
import scipy.io as sio
import numpy as np
import pandas as pd

def get_label():
    brain_fmri_dir = 'extracted/ROISignals_FunImgARglobalCWF'
    #/oldData/hym/gnn-code/
    subjects_path = os.listdir(brain_fmri_dir)
    subjects_path.sort()
    #print(subjects_path)
    label = []
    for subject in subjects_path:
        #sub.append(subject)
        if subject.split('-')[1] == '1':
            label.append(1)
        if subject.split('-')[1] == '2':
            label.append(0)
    label = np.array(label)
    return label

#print(get_label())