#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:32:40 2024

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

conditions = ['Misophone','Novels']
hemispheres = ['lh','rh']
fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'
local_dir = '/local_mount/space/hypatia/2/users/Jasmine/'
paradigm = 'Misophonia'
data_dir = os.path.join(local_dir, paradigm)
participant = '130101'


df = pd.read_pickle('/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/ROI_time_series_peaksVsLabels.pkl')

df_miso = df[df["Diagnosis"]=='Misophonic (n=16)']
df_outlier_removed = df_miso[df_miso["Participant"]!= 118801]

data_dir = os.path.join(local_dir,paradigm,participant)
condition_name = 'Misophone-lh.stc'
load_fname = find_files(condition_name,data_dir)[0]

subjID_date = find_mri_recons(subj_dir,load_fname)
stc = mne.read_source_estimate(load_fname,subject=subjID_date).savgol_filter(30)
label_name = 'aud_lh.label'
label_fname = find_files(label_name,data_dir)[0]
drawn_label = [mne.read_label(label_fname,subject= subjID_date)]
stc_label  = stc.in_label(drawn_label[0])
time = stc_label.times
fontsize=20

report = mne.Report(title="Comparing peak vertex and hand drawn labels")
#report  = mne.open_report(os.path.join("/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/","miso_peak_vertex_and_labels.html"))

#labels = ['auditory_peak_vertex','auditory_drawn_label']
labels = ['auditory_drawn_label']
for hemi in hemispheres:
    for label in labels:
        df_to_plot = df_outlier_removed.loc[(df_outlier_removed['hemisphere']== hemi) & (df_outlier_removed['label']== label)]
        
        df_to_plot_miso = df_to_plot.loc[df_to_plot['Condition']=='trigger']
        df_to_plot_novel = df_to_plot.loc[df_to_plot['Condition']=='distractor']

        time_series_to_plot_miso = df_to_plot_miso['peak_activation'].mean()
        time_series_to_plot_novel = df_to_plot_novel['peak_activation'].mean()
        
        sns.set(style="white")
        sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')
    
        #times = np.reshape(time,(1,-1))
        sub_ax1.plot(time,time_series_to_plot_miso, label='trigger')
        sub_ax1.plot(time,time_series_to_plot_novel, label='distractor')
        sub_ax1.set_xlim([-0.2,0.6])
        sub_ax1.set_ylim([0,40])
        sub_ax1.tick_params(labelsize=fontsize)
        sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
        sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
        sub_ax1.axvline(x=0, ls='--', color='k')
        
        if hemi == 'lh': 
            title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n Left Hemisphere'
        else:
            title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n Right Hemisphere'
        
        sub_ax1.set_title(title,fontsize=24)
        sub_ax1.legend(fontsize=fontsize,loc='upper left')
        
        fig_to_save = sub_fig.get_figure()
        savename = '/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/Grand_average_activations_' + label + '_' + hemi + '_.tiff'
        sub_fig.savefig(savename,dpi=300)
        
        if label == 'auditory_drawn_label':
            brain = mne.viz.Brain(subject = 'fsaverage',hemi = hemi ,views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')
            drawn_labels = df_to_plot_miso["drawn_label"]
            for info in drawn_labels:
                morphed_label= mne.morph_labels([info[0]], subject_to='fsaverage', subject_from=info[0].subject, subjects_dir=subj_dir, surf_name='inflated')
                brain.add_label(morphed_label[0], hemi = hemi, alpha=1)
            brain_image_name = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/brain"  + label + '_' + hemi + ".tiff"
            brain.save_image(filename=brain_image_name, mode='rgb')
        else:
            brain = mne.viz.Brain(subject = 'fsaverage',hemi = hemi ,views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')
            triggers_foci = df_to_plot_miso["fs_peak"]
            novels_foci = df_to_plot_novel["fs_peak"]
            for peak_vertex_surf_fs in triggers_foci:
                brain.add_foci(peak_vertex_surf_fs, coords_as_verts=True, hemi=hemi, color='#1f77b4',alpha = 0.8,scale_factor = 0.75)
            for peak_vertex_surf_fs in novels_foci:
                brain.add_foci(peak_vertex_surf_fs, coords_as_verts=True, hemi=hemi, color='#e377c2',alpha = 0.8,scale_factor = 0.75)
                    
            brain.add_text(0.1, 0.9, "Pink = novel, blue = misophonic trigger", "title", font_size=14)
            brain_image_name = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/brain.tiff"
            brain.save_image(filename=brain_image_name, mode='rgb')
    
        fig = plt.figure(figsize=(18,6), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig) 
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax1.imshow(plt.imread(brain_image_name))
        ax1.axis('off')
        ax2.imshow(plt.imread(savename))
        ax2.axis('off')
        
        section = 'grand_average'
        report.add_figure(fig=fig, title='Grand-averaged activations', section=section, tags=['grand_average',hemi,label])


#report.save("/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/grand_averaged_miso_peak_vertex_and_labels.html", overwrite=True)

            
    