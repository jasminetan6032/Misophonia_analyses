#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 12:35:57 2024

@author: jwt30
"""
import os
import scipy
import pandas

def find_file(search_string,data_dir):
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                
    return file    

subjs2exclude = ['112601','113301']
transcend_dir = '/autofs/space/transcend/MEG/'

run = 'run01_behaviour.mat'
misophones=[]

for subject in subjects:
    if subject not in subjs2exclude:
    
        data_dir = os.path.join(transcend_dir,'Misophonia',subject)
        mat_file   = find_file(run,data_dir)
        mat = scipy.io.loadmat(mat_file)
        misophones.append(mat["misophone"][0])


count = pandas.Series(misophones).value_counts()





