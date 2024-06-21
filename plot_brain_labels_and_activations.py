#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 18:04:34 2024

@author: jwt30
"""
import mne
import os
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd

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

local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'
paradigm = 'Misophonia'
data_dir = os.path.join(local_dir, paradigm)


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

participants_miso = []
participants_td = []
for participant in participants_list:
    if participant not in exclude_participants:
        #find diagnosis
        diagnosis = get_diagnosis(diagnosis_file,participant)
        if diagnosis == 'misophonia' or participant == '118601':
            participants_miso.append(participant)
        else: 
            participants_td.append(participant)

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'

participants_cond1 = []
participants_cond2 = []

#brain = mne.viz.Brain(subject = 'fsaverage',hemi = 'lh',views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')
participants_data = []

for participant in participants_td:
    data_dir = os.path.join(local_dir,paradigm,participant)

    #load label
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'miso_ofc_lh.label' in filename:
                label_fname = os.path.join(path,filename)
    
    subjID_date = find_mri_recons(subj_dir,label_fname)

    label = [mne.read_label(label_fname,subject= subjID_date)]

    # morphed_label= mne.morph_labels(label, subject_to='fsaverage', subject_from=subjID_date, subjects_dir=subj_dir, surf_name='inflated')
    # hemi = os.path.split(label_fname)[1].split('_')[2].split('.')[0]
    # print('Plotting participant ' + subjID_date)
    # brain.add_label(morphed_label[0], hemi = hemi, alpha=1)
    # print('Finished plotting participant ' + subjID_date)
 

#plot activations
    load_fname = find_files('_Misophone-lh.stc',data_dir)[0]
    stc_cond1 = mne.read_source_estimate(load_fname,subject=subjID_date)
    stc_cond1_filtered = stc_cond1.savgol_filter(30)

    load_fname = find_files('_Novels-lh.stc',data_dir)[0]
    stc_cond2 = mne.read_source_estimate(load_fname,subject=subjID_date)
    
    src_file = find_files('_run01_src.fif',data_dir)[0]
    src = mne.read_source_spaces(src_file)
    
    data_miso_lh = mne.extract_label_time_course(
        stc_cond1_filtered, label, src, mode="mean_flip", verbose="error"
    )
    data_novel_lh = mne.extract_label_time_course(
        stc_cond2, label, src, mode="mean_flip", verbose="error"
    )

    data_miso_lh_mean = np.mean(data_miso_lh,axis=0)
    data_novel_lh_mean = np.mean(data_novel_lh,axis=0)
    time = stc_cond1.times
    time_idx = (time>=0.08) & (time<=0.18)

    miso_mean = np.mean(data_miso_lh_mean[time_idx])
    novel_mean = np.mean(data_novel_lh_mean[time_idx])

    participant_data_miso = ['TD',participant,'misophonic trigger',miso_mean]
    participant_data_novel = ['TD',participant,'novel',novel_mean]
    
    participants_data.append(participant_data_miso)
    participants_data.append(participant_data_novel)
    
    participants_cond1.append(data_miso_lh.data)
    participants_cond2.append(data_novel_lh.data)

participants_cond1_stc_ave = np.mean(participants_cond1,axis=0)
participants_cond2_stc_ave = np.mean(participants_cond2,axis=0)

fontsize = 20
sns.set(style="white")
sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')

times = np.reshape(time,(1,-1))
sub_ax1.plot(times[0],participants_cond1_stc_ave[0], label='trigger')
sub_ax1.plot(times[0],participants_cond2_stc_ave[0], label='distractor')
sub_ax1.set_xlim([-0.2,0.6])
sub_ax1.set_ylim([0,7])
sub_ax1.tick_params(labelsize=fontsize)
sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
sub_ax1.axvline(x=0, ls='--', color='k')
sub_ax1.set_title('Grand-averaged activation in prefrontal ROIs \n Typically developing',fontsize=24)
sub_ax1.legend(fontsize=fontsize,loc='upper left')
sub_ax1.axvspan(0.38, 0.58, color='black', alpha=.15)

fig_to_save = sub_fig.get_figure()
sub_fig.savefig("/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/miso_activations.tiff",dpi=300)

df = pd.DataFrame(participants_data, columns = ['Group','Participant', 'Sound_type','Activation(AU)']) 

df=df.replace(to_replace='misophonic trigger', value='trigger', regex=True)
df=df.replace(to_replace='novel', value='distractor', regex=True)

df=df.replace(to_replace='Misophonic', value='Misophonic (n=16)', regex=True)
df=df.replace(to_replace='TD', value='TD (n=2)', regex=True)

fig, ax = plt.subplots()
#sns.set(rc={'figure.figsize':(8.27,8.27)})
# sns.axes_style('whitegrid')
# sns.set(font_scale=2)

ax = sns.barplot(x="Group", y="Activation(AU)",
            hue="Sound_type", palette=["#1b699e", "#ca6723"],
            data=df,capsize=.1, errorbar="sd")
ax.set_xlabel("",fontsize = 18)
ax.set_ylabel("Activation(AU)",fontsize=18)

ax = sns.swarmplot(x="Group", y="Activation(AU)", hue="Sound_type",data=df, alpha=.8,dodge=True,size=10)
ax.set_title("Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors",fontsize=18)

fig_to_save = fig.get_figure()
fig.savefig("/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/barplot.tiff",dpi=300)


import scipy
df_miso = df[df['Group'] == 'Misophonic (n=16)']
cat1 = df_miso[df_miso['Sound_type']=='trigger']
cat2 = df_miso[df_miso['Sound_type']=='distractor']
scipy.stats.ttest_rel(cat1['Activation(AU)'], cat2['Activation(AU)'])