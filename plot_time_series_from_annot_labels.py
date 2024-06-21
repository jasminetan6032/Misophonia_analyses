#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:31:59 2024

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

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

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

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'

labels_list = ['S_temporal_transverse','G_temp_sup-G_T_transv','S_central','S_circular_insula_ant','S_front_inf','S_orbital_lateral','G_front_inf-Orbital']
conditions = ['Misophone','Novels']
hemispheres = ['lh','rh']

participants_data = []

for participant in participants:
    data_dir = os.path.join(local_dir,paradigm,participant[1])
    src_file = find_files('_run01_src.fif',data_dir)[0]
    src = mne.read_source_spaces(src_file)

    for condition in conditions:
        condition_name = '_' + condition + '-lh.stc'
        load_fname = find_files(condition_name,data_dir)[0]
        subjID_date = find_mri_recons(subj_dir,load_fname)
        stc = mne.read_source_estimate(load_fname,subject=subjID_date)
        stc_filtered = stc.savgol_filter(30)        
            
        for hemi in hemispheres:
            #load_labels
            labels = []
            auditory_label1 = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[0], subjects_dir=subj_dir)
            auditory_label2 = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[1], subjects_dir=subj_dir)
            auditory_label = auditory_label1[0] + auditory_label2[0]
            name_label = ['auditory',auditory_label]
            labels.append(name_label)
            central_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[2], subjects_dir=subj_dir)
            name_label = ['central',central_label[0]]
            labels.append(name_label)
            insula_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[3], subjects_dir=subj_dir)
            name_label = ['insula',insula_label[0]]
            labels.append(name_label)
            frontal_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[4], subjects_dir=subj_dir)
            name_label = ['frontal',frontal_label[0]]
            labels.append(name_label)
            ofc_label1 = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[5], subjects_dir=subj_dir)
            ofc_label2 = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[6], subjects_dir=subj_dir)
            ofc_label = ofc_label1[0] + ofc_label2[0]
            name_label = ['orbitofrontal',ofc_label]
            labels.append(name_label)
    
            for label in labels:
                ROI = label[0]
                stc_label= mne.extract_label_time_course(
                    stc_filtered, label[1], src, mode="mean_flip", verbose="error"
                )
    
                data = [participant[0],participant[1],condition,hemi,ROI,stc_label]
                participants_data.append(data)


df = pd.DataFrame(participants_data, columns = ['Diagnosis','Participant','Condition','hemisphere','label','peak_activation']) 

df=df.replace(to_replace='Misophone', value='trigger', regex=True)
df=df.replace(to_replace='Novels', value='distractor', regex=True)

df=df.replace(to_replace='Misophonic', value='Misophonic (n=16)', regex=True)
df=df.replace(to_replace='TD', value='TD (n=2)', regex=True)


df.to_csv('ROI_time_series_labels.csv',index=False)

df_miso = df[df["Diagnosis"]=="Misophonic (n=16)"]
df_outlier_removed = df_miso[df_miso["Participant"]!= 118801]

time = stc.times
fontsize=20

for hemi in hemispheres:
    for label in labels:
        df_to_plot = df_outlier_removed.loc[(df_outlier_removed['hemisphere']== hemi) & (df_outlier_removed['label']== label[0])]
        
        df_to_plot_miso = df_to_plot.loc[df_to_plot['Condition']=='trigger']
        df_to_plot_novel = df_to_plot.loc[df_to_plot['Condition']=='distractor']
        time_series_to_plot_miso = df_to_plot_miso['peak_activation'].mean()
        time_series_to_plot_novel = df_to_plot['peak_activation'].mean()
        
        sns.set(style="white")
        sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')

        times = np.reshape(time,(1,-1))
        sub_ax1.plot(times[0],time_series_to_plot_miso[0], label='trigger')
        sub_ax1.plot(times[0],time_series_to_plot_novel[0], label='distractor')
        sub_ax1.set_xlim([-0.2,0.6])
        sub_ax1.set_ylim([0,15])
        sub_ax1.tick_params(labelsize=fontsize)
        sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
        sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
        sub_ax1.axvline(x=0, ls='--', color='k')
        
        if hemi == 'lh': 
            title = "Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n " + label[0] + " Left Hemisphere"
        else:
            title = "Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n " + label[0] + " Right Hemisphere"
        
        sub_ax1.set_title(title,fontsize=24)
        sub_ax1.legend(fontsize=fontsize,loc='upper left')
        
        fig_to_save = sub_fig.get_figure()
        savename = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_labels/" + title + ".tiff"
        sub_fig.savefig(savename,dpi=300)
        
        #sub_ax1.axvspan(0.38, 0.58, color='black', alpha=.15)
# for hemi in hemispheres:
#     for time_window in time_windows:
#         df_to_plot = df_outlier_removed.loc[(df_outlier_removed['time_window']==time_window[0]) & (df_outlier_removed['hemisphere']== hemi)]
#         fig, ax = plt.subplots()
#         #sns.set(rc={'figure.figsize':(8.27,8.27)})
#         # sns.axes_style('whitegrid')
#         # sns.set(font_scale=2)

#         ax = sns.barplot(x="label", y="peak_activation",
#                     hue="Condition", palette=["#1b699e", "#ca6723"],
#                     data=df_to_plot,capsize=.1, errorbar="sd")
#         ax.set_xlabel("",fontsize = 18)
#         ax.set_ylabel("Peak Activation(AU)",fontsize=18)

#         ax = sns.swarmplot(x="label", y="peak_activation", hue="Condition",data=df_to_plot, alpha=.8,dodge=True,size=5)
#         ax.set_xlabel("",fontsize = 18)
#         if hemi == 'lh': 
#             title = "Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n " + str(time_window[1]) + "s -" + str(time_window[2]) + "s " + "Left Hemisphere"
#         else:
#             title = "Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n " + str(time_window[1]) + "s -" + str(time_window[2]) + "s " + "Right Hemisphere"

#         ax.set_title(title,fontsize=12)
#         legend_without_duplicate_labels(ax)




# labels = []
# hemi = 'rh'
# auditory_label1 = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[0], subjects_dir=fsaverageDir)
# auditory_label2 = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[1], subjects_dir=fsaverageDir)
# auditory_label = auditory_label1[0] + auditory_label2[0]
# name_label = ['auditory',auditory_label]
# labels.append(name_label)
# central_label = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[2], subjects_dir=fsaverageDir)
# name_label = ['central',central_label[0]]
# labels.append(name_label)
# insula_label = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[3], subjects_dir=fsaverageDir)
# name_label = ['insula',insula_label[0]]
# labels.append(name_label)
# frontal_label = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[4], subjects_dir=fsaverageDir)
# name_label = ['frontal',frontal_label[0]]
# labels.append(name_label)
# ofc_label1 = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[5], subjects_dir=fsaverageDir)
# ofc_label2 = mne.read_labels_from_annot(subject = 'fsaverage', parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[6], subjects_dir=fsaverageDir)
# ofc_label = ofc_label1[0] + ofc_label2[0]
# name_label = ['ofc',ofc_label]
# labels.append(name_label)

# from matplotlib.pyplot import cm
# color = cm.rainbow(np.linspace(0, 1, 5))
# for i, c in enumerate(color):
#     color[i,3] = 0.5
    
# for i in range(len(labels)):
#     labels[i].append(color[i])
    
# brain = mne.viz.Brain(subject = 'fsaverage',hemi = hemi,views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')
# for label in labels:
#     brain.add_label(label[1], hemi = hemi, color = label[2])

    