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
savedir = os.path.join(data_dir,'Poster_images','Jun2024','central_sulcus')
data_savename = os.path.join(savedir,'central_sulcus_misoVsNovels.pkl')
report_savename = os.path.join(savedir,'central_sulcus_labels_miso_vs_novel_labels_vs_peaks.html')


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


labels_list = ['S_central']
conditions = ['Misophone','Novels']
hemispheres = ['lh','rh']
label_compare = ['drawn_label']

participants_data = []

participants_to_study = [el[1] for el in participants]

for participant in participants:
    if participant[1] in participants_to_study:

        participant_data = []
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
                label_name = 'central_' + hemi + '.label'
                label_fname = find_files(label_name,data_dir)[0]
                drawn_label = [mne.read_label(label_fname,subject= subjID_date)]
                
                central_label = mne.read_labels_from_annot(subjID_date, parc = 'aparc.a2009s',hemi = hemi, surf_name = 'white', regexp = labels_list[0], subjects_dir=subj_dir)
                morphed_central_label= mne.morph_labels(central_label, subject_to='fsaverage', subject_from=subjID_date, subjects_dir=subj_dir, surf_name='inflated')
    
                stc_label  = stc.in_label(central_label[0])
                stc_fs_label  = stc_fsaverage.in_label(morphed_central_label[0])
                peak_vertex,peak_time = stc_label.get_peak(hemi = hemi, tmin = 0.1,tmax = 0.3,vert_as_index = True, time_as_index = True)
                peak_vertex_fs,peak_time_fs = stc_fs_label.get_peak(hemi = hemi, tmin = 0.1,tmax = 0.3,vert_as_index = True, time_as_index = True)
                
                if hemi == 'lh':
                    peak_vertex_stc = stc_label.lh_data[peak_vertex,:]
                else:
                    peak_vertex_stc = stc_label.rh_data[peak_vertex,:]
                
                peak_vertex_abs = flip_peaks(peak_vertex_stc)
                time = stc_label.times
    
                if hemi == 'lh':
                    peak_vertex_surf = stc_label.lh_vertno[peak_vertex]
                    peak_vertex_surf_fs = stc_fs_label.lh_vertno[peak_vertex_fs]
                else:
                    peak_vertex_surf = stc_label.rh_vertno[peak_vertex]
                    peak_vertex_surf_fs = stc_fs_label.rh_vertno[peak_vertex_fs]
    
                data_drawn_label = mne.extract_label_time_course(
                        stc, drawn_label, src, mode="mean_flip", verbose="error"
                    )
    
                data_drawn_abs = flip_peaks(data_drawn_label[0])
    
                data_peak = [participant[0],participant[1],condition,hemi,'peak_vertex',peak_vertex_abs,peak_vertex_surf,peak_time,peak_vertex_surf_fs,stc,time]
                data_label = [participant[0],participant[1],condition,hemi,'drawn_label',data_drawn_abs,drawn_label,peak_time,peak_vertex_surf_fs,stc,time]
                participants_data.append(data_peak)
                participant_data.append(data_peak)
                participants_data.append(data_label)
                participant_data.append(data_label)
    
        df_participant = pd.DataFrame(participant_data, columns = ['Diagnosis','Participant','Condition','hemisphere','label','peak_activation','peak','peak_time','fs_peak','stc','time']) 
        df_participant=df_participant.replace(to_replace='Misophone', value='trigger', regex=True)
        df_participant=df_participant.replace(to_replace='Novels', value='distractor', regex=True)
    
        df_participant=df_participant.replace(to_replace='Misophonic', value='Misophonic (n=16)', regex=True)
        df_participant=df_participant.replace(to_replace='TD', value='TD (n=2)', regex=True)
        if participant[1] == '118801':
            ylims = [0,100]
        else:
            ylims = [0,50]
    
        for label in label_compare:
            for hemi in hemispheres:
                df_to_plot = df_participant.loc[(df_participant['hemisphere']== hemi) & (df_participant['label']== label)]
    
                df_to_plot_miso = df_to_plot.loc[df_to_plot['Condition']=='trigger']
                df_to_plot_novel = df_to_plot.loc[df_to_plot['Condition']=='distractor']
                time_series_to_plot_miso = df_to_plot_miso['peak_activation'].mean()
                time_series_to_plot_novel = df_to_plot_novel['peak_activation'].mean()
    
                sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')
    
                time = df_to_plot_miso['time'].values[0]
                sub_ax1.plot(time,time_series_to_plot_miso, label='trigger')
                sub_ax1.plot(time,time_series_to_plot_novel, label='distractor')
                sub_ax1.set_xlim([-0.2,0.6])
                sub_ax1.set_ylim(ylims)
                sub_ax1.tick_params(labelsize=fontsize)
                sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
                sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
                sub_ax1.axvline(x=0, ls='--', color='k')
                title = 'central' + '_' + hemi + '_' +label + '_activations'
                sub_ax1.set_title(title,fontsize=24)
                sub_ax1.legend(fontsize=fontsize,loc='upper left')
                
                fig_to_save = sub_fig.get_figure()
                savename = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/" + title + ".tiff"
                sub_fig.savefig(savename,dpi=300)
                
                peak_time_miso = df_to_plot_miso['peak_time'].values[0]
                peak_time_novel = df_to_plot_novel['peak_time'].values[0]
    
                initial_time = time[peak_time_miso]
                stc = df_to_plot_miso['stc'].values[0]
                brain = stc.plot(
                    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
                    hemi=hemi,
                    initial_time=initial_time,
                    clim=dict(kind="percent", lims=[99.5, 99.7, 99.9]),
                    smoothing_steps=7,
                    time_viewer = False
                )
                
                if label == 'peak_vertex':
                    peak_vertex_surf_miso = df_to_plot_miso['peak'].values[0]
                    brain.add_foci(peak_vertex_surf_miso, coords_as_verts=True, hemi=hemi, color="b",alpha = 0.8,scale_factor = 0.75)
                else:
                    brain.add_label(df_to_plot_miso['peak'].values[0][0],hemi = hemi, alpha=1,color = 'tab:cyan')
    
                brain_image_name = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/brain.tiff"
                brain.save_image(filename=brain_image_name, mode='rgb')
                brain.close()
    
                initial_time = time[peak_time_novel]
                stc = df_to_plot_novel['stc'].values[0]
                brain1 = stc.plot(
                    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
                    hemi=hemi,
                    initial_time=initial_time,
                    clim=dict(kind="percent", lims=[99.5, 99.7, 99.9]),
                    smoothing_steps=7,
                    time_viewer = False
                )
                
                if label == 'peak_vertex':
                    peak_vertex_surf_novel = df_to_plot_novel['peak'].values[0]
                    brain1.add_foci(peak_vertex_surf_novel, coords_as_verts=True, hemi=hemi, color="m",alpha = 0.8,scale_factor = 0.75)
                else:
                    brain1.add_label(df_to_plot_novel['peak'].values[0][0],hemi = hemi, alpha=1,color = 'tab:cyan')
                
                brain1_image_name = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/brain1.tiff"
                brain1.save_image(filename=brain1_image_name, mode='rgb')
                brain1.close()
                
                
                brain2 = mne.viz.Brain(subject = subjID_date,hemi = hemi ,views = 'lateral',subjects_dir = subj_dir,surf='inflated',background='white')
                if label == 'peak_vertex':
                    brain2.add_foci(peak_vertex_surf_miso, coords_as_verts=True, hemi=hemi, color='#1f77b4',alpha = 0.8,scale_factor = 0.75)
                    brain2.add_foci(peak_vertex_surf_novel, coords_as_verts=True, hemi=hemi, color='#e377c2',alpha = 0.8,scale_factor = 0.75)
                else:
                    brain2.add_label(df_to_plot_novel['peak'].values[0][0],hemi = hemi, alpha=1,color = 'tab:cyan')
                
                brain2_image_name = "/local_mount/space/hypatia/2/users/Jasmine/Misophonia/Poster_images/Jun2024/roi_time_series_abs/brain2.tiff"
                brain2.save_image(filename=brain2_image_name, mode='rgb')
                brain2.close()
    
    
                fig1 = plt.figure(figsize=(12,6), layout='constrained')
                gs  = GridSpec(1, 2, figure=fig1) 
                ax1 = fig1.add_subplot(gs[0,0])
                ax2 = fig1.add_subplot(gs[0,1])
                ax1.imshow(plt.imread(brain_image_name))
                ax1.axis('off')
                ax2.imshow(plt.imread(brain1_image_name))
                ax2.axis('off')
                
                report.add_figure(fig=fig1, title=title, section=section, tags=[hemi,diagnosis,label])
    
                fig2 = plt.figure(figsize=(18,6), layout='constrained')
                gs  = GridSpec(1, 2, figure=fig2) 
                ax1 = fig2.add_subplot(gs[0,0])
                ax2 = fig2.add_subplot(gs[0,1])
                ax1.imshow(plt.imread(brain2_image_name))
                ax1.axis('off')
                ax2.imshow(plt.imread(savename))
                ax2.axis('off')
                
                report.add_figure(fig=fig2, title=title, section=section, tags=[hemi,diagnosis,label])


df= pd.DataFrame(participants_data, columns = ['Diagnosis','Participant','Condition','hemisphere','label','peak_activation','peak','peak_time','fs_peak','stc','time']) 

df=df.replace(to_replace='Misophone', value='trigger', regex=True)
df=df.replace(to_replace='Novels', value='distractor', regex=True)

df=df.replace(to_replace='Misophonic', value='Misophonic (n=16)', regex=True)
df=df.replace(to_replace='TD', value='TD (n=2)', regex=True)


df.to_pickle(data_savename)

#df_miso = df[df["Participant"]=="129301"]
fontsize=20

savedir = os.path.join(data_dir,'Poster_images','Jun2024','central_sulcus','all_participants')
df = pd.read_pickle(data_savename)
report = mne.Report(title="Comparing misophonic and novel peaks")
for participant in participants:
    df_subject = df[df["Participant"]==participant[1]]
    for label in label_compare:
        for hemi in hemispheres:
            df_to_plot = df_subject.loc[(df_subject['hemisphere']== hemi) & (df_subject['label']== label)]
            
            df_to_plot_miso = df_to_plot.loc[df_to_plot['Condition']=='trigger']
            df_to_plot_novel = df_to_plot.loc[df_to_plot['Condition']=='distractor']
            time_series_to_plot_miso = df_to_plot_miso['peak_activation'].mean()
            time_series_to_plot_novel = df_to_plot['peak_activation'].mean()
            
            sns.set(style="white")
            sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')
    
            time = df_to_plot_miso['time'].values[0]
            sub_ax1.plot(time,time_series_to_plot_miso, label='trigger')
            sub_ax1.plot(time,time_series_to_plot_novel, label='distractor')
            sub_ax1.set_xlim([-0.2,0.6])
            if participant[1] == '118801':
                ylims = [0,100]
            else:
                ylims = [0,50]
            sub_ax1.set_ylim(ylims)
            sub_ax1.tick_params(labelsize=fontsize)
            sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
            sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
            sub_ax1.axvline(x=0, ls='--', color='k')
            
            if hemi == 'lh': 
                title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n' + 'Central' + ' Left Hemisphere'
            else:
                title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n' + 'Central' + ' Right Hemisphere'
            
            sub_ax1.set_title(title,fontsize=24)
            sub_ax1.legend(fontsize=fontsize,loc='upper left')
            
            fig_to_save = sub_fig.get_figure()
            fig_name = '_'.join([participant[1], 'Central Sulcus', hemi + '.tiff'])
            savename = os.path.join(savedir,fig_name)
            sub_fig.savefig(savename,dpi=300)
            
            brain = mne.viz.Brain(subject = 'fsaverage',hemi = hemi ,views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')
            if label == 'peak_vertex':
                triggers_foci = df_to_plot_miso["fs_peak"]
                novels_foci = df_to_plot_novel["fs_peak"]
                for peak_vertex_surf_fs in triggers_foci:
                    brain.add_foci(peak_vertex_surf_fs, coords_as_verts=True, hemi=hemi, color='#1f77b4',alpha = 0.8,scale_factor = 0.75)
                for peak_vertex_surf_fs in novels_foci:
                    brain.add_foci(peak_vertex_surf_fs, coords_as_verts=True, hemi=hemi, color='#e377c2',alpha = 0.8,scale_factor = 0.75)
            else:
                drawn_labels = df_to_plot_miso["peak"]
                for info in drawn_labels:
                    morphed_label= mne.morph_labels([info[0]], subject_to='fsaverage', subject_from=info[0].subject, subjects_dir=subj_dir, surf_name='inflated')
                    brain.add_label(morphed_label[0], hemi = hemi, alpha=1)
                        
            #brain.add_text(0.1, 0.9, "Pink = novel, blue = misophonic trigger", "title", font_size=14)
            brain_fig_name = '_'.join([participant[1], 'Central Sulcus', hemi + '_brain.tiff'])
            brain_image_name = os.path.join(savedir,brain_fig_name)
            brain.save_image(filename=brain_image_name, mode='rgb')
    
            fig = plt.figure(figsize=(18,6), layout='constrained')
            gs  = GridSpec(1, 2, figure=fig) 
            ax1 = fig.add_subplot(gs[0,0])
            ax2 = fig.add_subplot(gs[0,1])
            ax1.imshow(plt.imread(brain_image_name))
            ax1.axis('off')
            ax2.imshow(plt.imread(savename))
            ax2.axis('off')
            
            section = participant[1]
            report.add_figure(fig=fig, title='Grand-averaged activations', section=section, tags=['grand_average',hemi,label])


df_miso = df[df["Diagnosis"]=="Misophonic (n=16)"]
for label in label_compare:
    for hemi in hemispheres:

        df_to_plot = df_miso.loc[(df_miso['hemisphere']== hemi) & (df_miso['label']== label)]
        
        df_to_plot_miso = df_to_plot.loc[df_to_plot['Condition']=='trigger']
        df_to_plot_novel = df_to_plot.loc[df_to_plot['Condition']=='distractor']
        time_series_to_plot_miso = df_to_plot_miso['peak_activation'].mean()
        time_series_to_plot_novel = df_to_plot['peak_activation'].mean()
        
        sns.set(style="white")
        sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')

        time = df_to_plot_miso['time'].values[0]
        sub_ax1.plot(time,time_series_to_plot_miso, label='trigger')
        sub_ax1.plot(time,time_series_to_plot_novel, label='distractor')
        sub_ax1.set_xlim([-0.2,0.6])
        sub_ax1.set_ylim([0,30])
        sub_ax1.tick_params(labelsize=fontsize)
        sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
        sub_ax1.set_ylabel('dSPM activation (AU)',fontsize=fontsize)
        sub_ax1.axvline(x=0, ls='--', color='k')
        
        if hemi == 'lh': 
            title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n' + 'Central' + ' Left Hemisphere'
        else:
            title = 'Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors \n' + 'Central' + ' Right Hemisphere'
        
        sub_ax1.set_title(title,fontsize=24)
        sub_ax1.legend(fontsize=fontsize,loc='upper left')
        
        fig_to_save = sub_fig.get_figure()
        fig_name = 'all_participants_Central Sulcus' + '_' + hemi + '_.tiff'
        savename = os.path.join(savedir,fig_name)
        sub_fig.savefig(savename,dpi=300)
        
        brain = mne.viz.Brain(subject = 'fsaverage',hemi = hemi ,views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')
        if label == 'peak_vertex':
            triggers_foci = df_to_plot_miso["fs_peak"]
            novels_foci = df_to_plot_novel["fs_peak"]
            for peak_vertex_surf_fs in triggers_foci:
                brain.add_foci(peak_vertex_surf_fs, coords_as_verts=True, hemi=hemi, color='#1f77b4',alpha = 0.8,scale_factor = 0.75)
            for peak_vertex_surf_fs in novels_foci:
                brain.add_foci(peak_vertex_surf_fs, coords_as_verts=True, hemi=hemi, color='#e377c2',alpha = 0.8,scale_factor = 0.75)
        else:
            drawn_labels = df_to_plot_miso["peak"]
            for info in drawn_labels:
                morphed_label= mne.morph_labels([info[0]], subject_to='fsaverage', subject_from=info[0].subject, subjects_dir=subj_dir, surf_name='inflated')
                brain.add_label(morphed_label[0], hemi = hemi, alpha=1)
                    
        #brain.add_text(0.1, 0.9, "Pink = novel, blue = misophonic trigger", "title", font_size=14)
        brain_fig_name = '_'.join(['all_participants_Central Sulcus', hemi + '_brain.tiff'])
        brain_image_name = os.path.join(savedir,brain_fig_name)
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


report.save(report_savename, overwrite=True)

            
    