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

report = mne.Report(title="Comparing peak vertex and hand drawn labels")


labels_list = ['S_temporal_transverse','G_temp_sup-G_T_transv','S_central']
time_windows = [['tw1',0.1,0.3],['tw2',0.35,0.5]]
conditions = ['Misophone','Novels']
hemispheres = ['lh','rh']

participants_data = []

for participant in participants:
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
        
        morph = mne.compute_source_morph(
                                            stc,
                                            subject_from=subjID_date,
                                            subject_to="fsaverage",
                                            src_to=src_to,
                                            subjects_dir=subj_dir,
                                        )
        stc_fsaverage = morph.apply(stc)
        
        for hemi in hemispheres:
            #load_labels
            label_name = 'aud_' + hemi + '.label'
            label_fname = find_files(label_name,data_dir)[0]
            drawn_label = [mne.read_label(label_fname,subject= subjID_date)]

            auditory_label1 = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[0], subjects_dir=subj_dir)
            auditory_label2 = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[1], subjects_dir=subj_dir)
            auditory_label = auditory_label1[0] + auditory_label2[0]
            morphed_auditory_label= mne.morph_labels([auditory_label], subject_to='fsaverage', subject_from=auditory_label.subject, subjects_dir=subj_dir, surf_name='inflated')

            stc_label  = stc.in_label(auditory_label)
            stc_fs_label  = stc_fsaverage.in_label(morphed_auditory_label[0])
            peak_vertex,peak_time = stc_label.get_peak(hemi = hemi, tmin = 0.1,tmax = 0.3,vert_as_index = True, time_as_index = True)
            peak_vertex_fs,peak_time_fs = stc_fs_label.get_peak(hemi = hemi, tmin = 0.1,tmax = 0.3,vert_as_index = True, time_as_index = True)
            
            if hemi == 'lh':
                peak_vertex_stc = stc_label.lh_data[peak_vertex,:]
            else:
                peak_vertex_stc = stc_label.rh_data[peak_vertex,:]
            
            peak_vertex_abs = flip_peaks(peak_vertex_stc)
            
            data_drawn_label = mne.extract_label_time_course(
                    stc, drawn_label, src, mode="mean_flip", verbose="error"
                )

            data_drawn_abs = flip_peaks(data_drawn_label[0])
            
            sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')
            time = stc_label.times
            times = np.reshape(time,(1,-1))
            sub_ax1.plot(time,peak_vertex_abs, label='peak_vertex')
            sub_ax1.plot(time,data_drawn_abs, label='drawn_labels')
            sub_ax1.set_xlim([-0.2,0.6])
            sub_ax1.set_ylim([-10,55])
            sub_ax1.tick_params(labelsize=fontsize)
            sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
            sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
            sub_ax1.axvline(x=0, ls='--', color='k')
            title = condition + '_' + hemi + '_activations'
            sub_ax1.set_title(title,fontsize=24)
            sub_ax1.legend(fontsize=fontsize,loc='upper left')
            
            fig_to_save = sub_fig.get_figure()
            savename = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/" + title + ".tiff"
            sub_fig.savefig(savename,dpi=300)
            
            initial_time = time[peak_time]
            brain = stc.plot(
                subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
                hemi='both',
                initial_time=initial_time,
                clim=dict(kind="percent", lims=[99.5, 99.7, 99.9]),
                smoothing_steps=7,
                time_viewer = False,
                brain_kwargs = dict(show=False)
            )
            
            
            if hemi == 'lh':
                peak_vertex_surf = stc_label.lh_vertno[peak_vertex]
                peak_vertex_surf_fs = stc_fs_label.lh_vertno[peak_vertex_fs]
            else:
                peak_vertex_surf = stc_label.rh_vertno[peak_vertex]
                peak_vertex_surf_fs = stc_fs_label.rh_vertno[peak_vertex_fs]
            
            brain.add_foci(peak_vertex_surf, coords_as_verts=True, hemi=hemi, color="c",alpha = 0.8,scale_factor = 0.75)
            
            brain.add_label(drawn_label[0],borders = False, color = "m")
            
            brain_image_name = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/brain.tiff"
            brain.save_image(filename=brain_image_name, mode='rgb')
            brain.close()
            
            fig = plt.figure(figsize=(18,6), layout='constrained')
            gs  = GridSpec(1, 2, figure=fig) 
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax1.imshow(plt.imread(brain_image_name))
            ax1.axis('off')
            ax2.imshow(plt.imread(savename))
            ax2.axis('off')
            
            report.add_figure(fig=fig, title=title, section=section, tags=[condition,hemi,diagnosis], replace=True)

            data_peak = [participant[0],participant[1],condition,hemi,'auditory_peak_vertex',peak_vertex_abs,drawn_label,peak_vertex_surf_fs]
            data_label = [participant[0],participant[1],condition,hemi,'auditory_drawn_label',data_drawn_abs,drawn_label,peak_vertex_surf_fs]
            participants_data.append(data_peak)
            participants_data.append(data_label)

df = pd.DataFrame(participants_data, columns = ['Diagnosis','Participant','Condition','hemisphere','label','peak_activation','drawn_label','fs_peak']) 

df=df.replace(to_replace='Misophone', value='trigger', regex=True)
df=df.replace(to_replace='Novels', value='distractor', regex=True)

df=df.replace(to_replace='Misophonic', value='Misophonic (n=16)', regex=True)
df=df.replace(to_replace='TD', value='TD (n=2)', regex=True)


df.to_pickle('/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/ROI_time_series_peaksVsLabels.pkl')

df_miso = df[df["Diagnosis"]=="Misophonic (n=16)"]
df_outlier_removed = df_miso[df_miso["Participant"]!= 118801]

labels = ['auditory_peak_vertex','auditory_drawn_label']
for hemi in hemispheres:
    for label in labels:
        df_to_plot = df_outlier_removed.loc[(df_outlier_removed['hemisphere']== hemi) & (df_outlier_removed['label']== label)]
        
        df_to_plot_miso = df_to_plot.loc[df_to_plot['Condition']=='trigger']
        df_to_plot_novel = df_to_plot.loc[df_to_plot['Condition']=='distractor']
        time_series_to_plot_miso = df_to_plot_miso['peak_activation'].mean()
        time_series_to_plot_novel = df_to_plot_novel['peak_activation'].mean()
        
        sns.set(style="white")
        sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')
        sub_ax1.plot(time,time_series_to_plot_miso, label='trigger')
        sub_ax1.plot(time,time_series_to_plot_novel, label='distractor')
        sub_ax1.set_xlim([-0.2,0.6])
        sub_ax1.set_ylim([0,35])
        sub_ax1.tick_params(labelsize=fontsize)
        sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
        sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
        sub_ax1.axvline(x=0, ls='--', color='k')
        
        if hemi == 'lh': 
            title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n' + label + ' Left Hemisphere'
        else:
            title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n' + label + ' Right Hemisphere'
        
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
            brain.close()
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
            brain.close()
    
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


report.save("/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/miso_peak_vertex_and_labels.html", overwrite=True)