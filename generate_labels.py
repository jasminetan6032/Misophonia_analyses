#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:32:40 2024

@author: jwt30
"""


import mne
import os
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files  

def find_mri_recons(subj_dir,filename):
    participant = os.path.split(filename)[1].split('_')[0]
    possible_directories = []
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                possible_directories.append(dir)
                
    valid_directories = [i for i in range(0, len(possible_directories)) if len(possible_directories[i].split('_')) == 2 and len(possible_directories[i].split('_')[1])==8]
    
    
    meg_date = int(os.path.split(os.path.split(filename)[0])[1].split('_')[1])
    
    if len(valid_directories) == 1:
        subjID_date = possible_directories[valid_directories[0]]
    else:
        date_differences = []
        for i in range(0, len(valid_directories)):
            date=int(possible_directories[valid_directories[i]].split('_')[1])
            date_difference = meg_date-date
            date_differences.append(abs(date_difference))
        correct_file = valid_directories[date_differences.index(min(date_differences))]
        subjID_date = possible_directories[correct_file]
    return subjID_date

def get_diagnosis(csvfile,subject):
    """
    Get relevant behavioral data from redcap csv generated label file. This function requires as csv (labels) file generated from redcap and located in
    the paradigm folder in local_mount (cfg.paradigm_dir). The file should be saved as redcap_info.csv

    Parameters
    ----------
    subject : str
        subject ID
    """

        # diagnosis
    diagnosis = []
    for i in list(csvfile.columns[19:21]):
        this_val = csvfile[i][csvfile['Subject ID:'] == subject].dropna()
        if not this_val.empty:
            diagnosis.append(this_val.values)
    diagnosis = 'misophonia' if 'Yes' in diagnosis else 'td'
    
    return diagnosis

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

def flip_peaks(time_series):
    time_series_abs = np.abs(time_series)
    max_index = np.argmax(time_series_abs)
    if time_series[max_index] <0:
        time_series_output = time_series * -1
    else:
        time_series_output = time_series
    return time_series_output


local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'
paradigm = 'Misophonia'
data_dir = os.path.join(local_dir, paradigm)

labels_list = ['S_central']
conditions = ['Misophone','Novels']
hemispheres = ['lh','rh']

participants_data = []

#load participant list
fname = '/local_mount/space/hypatia/2/users/Jasmine/Misophonia/updated_meg_mri_alignment_20240525.csv'
csvfile = pd.read_csv(fname, sep=',')
csvfile['Subject'] = csvfile['Subject'].astype('string').str.zfill(6)
participants_list = list(set(csvfile['Subject']))
exclude_participants = ['113201','113301','000000','135401','112601','KSU_te']

#load fixation_basic for diagnosis
diagnosis_fname   = os.path.join(data_dir,'behavioral_and_demographics.csv')
diagnosis_file = pd.read_csv(diagnosis_fname, sep=',')
diagnosis_file['Subject ID:'] = diagnosis_file['Subject ID:'].astype('string').str.zfill(6)
fontsize = 20

participants= []
for participant in participants_list:
    if participant not in exclude_participants:
            #find diagnosis
            diagnosis = get_diagnosis(diagnosis_file,participant)
            if diagnosis == 'misophonia' or participant == '118601':
                group_participant = ['Misophonic',participant]
                participants.append(group_participant)
            else: 
                group_participant = ['TD',participant]
                participants.append(group_participant)


fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'
fname_fsaverage_src = os.path.join(fsaverageDir, "fsaverage" , "bem" , "fsaverage-ico-5-src.fif")
src_to = mne.read_source_spaces(fname_fsaverage_src)

report = mne.Report(title="Comparing misophonic and novel peaks")

# Compute a label/ROI based on the peak power between 80 and 120 ms.
# The label bankssts-lh is used for the comparison.
aparc_label_name = "S_central"
tmin, tmax = 0.100, 0.300

# Load data
participant_data = []
participant = participants[11]
data_dir = os.path.join(local_dir,paradigm,participant[1])
section = participant[1]
diagnosis = participant[0]
src_file = find_files('_run01_src.fif',data_dir)[0]
src = mne.read_source_spaces(src_file)


for condition in conditions:
    condition_name = '_' + condition + '-lh.stc'
    load_fname = find_files(condition_name,data_dir)[0]

    subjID_date = find_mri_recons(subj_dir,load_fname)
    stc = mne.read_source_estimate(load_fname,subject=subjID_date).savgol_filter(30)

# Make an STC in the time interval of interest and take the mean
stc_mean = stc.copy().crop(tmin, tmax).mean()

# use the stc_mean to generate a functional label
# region growing is halted at 60% of the peak value within the
# anatomical label / ROI specified by aparc_label_name

for hemi in hemispheres:
    #load_labels
    central_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[0], subjects_dir=subj_dir)

    stc_mean_label = stc_mean.in_label(central_label[0])
    data = np.abs(stc_mean_label.data)
    stc_mean_label.data[data < 0.6 * np.max(data)] = 0.0

# 8.5% of original source space vertices were omitted during forward
# calculation, suppress the warning here with verbose='error'
func_labels, _ = mne.stc_to_label(
    stc_mean_label,
    src=src,
    smooth=True,
    subjects_dir=subj_dir,
    connected=False,
    verbose="error",
)

# take first as func_labels are ordered based on maximum values in stc
func_label = func_labels[0]

brain = mne.viz.Brain(subject = subjID_date,hemi = hemi ,views = 'lateral',subjects_dir = subj_dir,surf='inflated',background='white')
brain.add_label(func_labels[0], hemi = hemi, alpha=1)

for f_label in func_labels:
    brain.add_label(f_label, hemi = hemi, alpha=1)
# extract the anatomical time course for each label
stc_anat_label = stc.in_label(anat_label)
pca_anat = stc.extract_label_time_course(anat_label, src, mode="pca_flip")[0]

stc_func_label = stc.in_label(func_label)
pca_func = stc.extract_label_time_course(func_label, src, mode="pca_flip")[0]

# flip the pca so that the max power between tmin and tmax is positive
pca_anat *= np.sign(pca_anat[np.argmax(np.abs(pca_anat))])
pca_func *= np.sign(pca_func[np.argmax(np.abs(pca_anat))])