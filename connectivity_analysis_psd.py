#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:25:10 2024

@author: jwt30
"""

import mne
import os
import pandas as pd
from autoreject import get_rejection_threshold
from mne_connectivity import spectral_connectivity_epochs, seed_target_indices, seed_target_multivariate_indices
from mne.minimum_norm import apply_inverse_epochs,source_induced_power
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def find_files(search_string,data_dir):
    files = []
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if search_string in filename:
                file = os.path.join(path,filename)
                files.append(file)
                
    return files  

def get_condition_epochs(epochs,stimuli):
    condition_epochs = epochs[stimuli]
    reject = get_rejection_threshold(condition_epochs, ch_types=['mag','grad'], decim=2)
    condition_epochs.drop_bad(reject=reject)
    #evoked = condition_epochs.average()
    return condition_epochs

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
subj_dir = '/autofs/space/transcend/MRI/WMA/recons/'
savedir = os.path.join(data_dir,'test_analyses','power')

#get stcs for each epoch
method = "dSPM"
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr**2


condition = ['Misophone','Novels']
hemisphere = ['lh','rh']
labels_to_compare = ['aud']
diagnoses = ['misophonia','td']

#connectivity settings                    
tmin_con = 0.0
tmax_con = 1.2
freq_min = 6
freq_max = 81
con_method = "dSPM"
con_n_cycles = 4

#plotting settings
vmin = 0.0
vmax = 1.0

connectivity_output_dir = os.path.join(savedir,'_'.join([con_method]+labels_to_compare))
if not os.path.exists(connectivity_output_dir):
    os.makedirs(connectivity_output_dir)

save_fname = '_'.join(condition + labels_to_compare + [con_method])
data_fname = save_fname + '.pkl'
data_savename = os.path.join(connectivity_output_dir,data_fname)

report_title = '_'.join(condition + labels_to_compare)
report = mne.Report(title=report_title+con_method)

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

participants_to_study = participants_list

all_participants = []

for sub_id in participants_list:
    if sub_id in participants_to_study:
        if sub_id not in exclude_participants:
            participant = sub_id
            participant_dir = os.path.join(local_dir,paradigm,str(participant))
            
            #find diagnosis            
            if sub_id == '118601':
                diagnosis = 'misophonia'
            else:
               diagnosis = get_diagnosis(diagnosis_file,sub_id) 

            #load epochs
            load_fname = find_files('Misophone_epo.fif',participant_dir)[0]
            subjID_date = find_mri_recons(subj_dir, load_fname)
            info = mne.io.read_info(load_fname)
            sfreq = info["sfreq"]  # the sampling frequency)
            
            #load inverse operator
            fwd_path = find_files('_fwd.fif',participant_dir)[0]
            inv_fname = fwd_path.replace('_fwd.fif', '_inv.fif')
            if os.path.isfile(inv_fname):
                inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fname)
            else:
                fwd   = mne.read_forward_solution(fwd_path, verbose=False)
                covfname = find_files('_cov.fif',participant_dir)[0]
                noise_cov = mne.read_cov(covfname)
                inverse_operator = mne.minimum_norm.make_inverse_operator(
                    info, fwd, noise_cov, loose=0.2, depth=0.8
                )
                mne.minimum_norm.write_inverse_operator(inv_fname,inverse_operator)
            
            #select epochs
            for condition_type in condition:
                condition_name = '_' + condition_type + '_epo.fif'
                epoch_name = find_files(condition_name,participant_dir)[0]
                print("Analysing " + epoch_name)
                epochs = mne.read_epochs(epoch_name)
                for label in labels_to_compare:
                    for hemi in hemisphere:
                        label_name = label + '_' + hemi +'.label'
                        fname_label = find_files(label_name,participant_dir)[0]
                        label_to_analyse = mne.read_label(fname_label)

                        src = inverse_operator["src"]  # the source space used

                        #calculate connectivity (coherence)                    
                        power, itc = source_induced_power(
                            epochs,
                            inverse_operator,
                            method= con_method,
                            freqs = np.arange(freq_min,freq_max,1),
                            label = [label_to_analyse],
                            baseline = (-0.2,0),
                            baseline_mode = 'percent',
                            n_cycles = 4,
                            n_jobs = None,
                        )
                        
                        norm_power = (power - np.min(power)) / (np.max(power) - np.min(power))
                        norm_itc = (itc - np.min(itc)) / (np.max(itc) - np.min(itc))

                        participant_data = [diagnosis,participant,condition_type,label,hemi,np.squeeze(norm_power),np.squeeze(np.mean(norm_itc,axis=0)),epochs.times]
                        all_participants.append(participant_data)

                        #plot time-frequency plot
                        sub_fig,sub_ax1 = plt.subplots(figsize=(6.4,4.8), layout='constrained')
                        pc = sub_ax1.pcolormesh(epochs.times,np.arange(freq_min,freq_max,1),np.squeeze(norm_power),vmin = vmin, vmax = vmax)
                        #save temporary plot for report
                        if hemi == 'lh': 
                            title = 'Power for ' + condition_type + ' \n Left Hemisphere'
                        else:
                            title = 'Power for ' + condition_type + ' \n Right Hemisphere'
                        
                        sub_ax1.set_title(title,fontsize=16)
                        sub_fig.colorbar(pc)
                        
                        fig_to_save = sub_fig.get_figure()
                        savetitle = '_'.join([condition_type,hemi,'power','plot'])
                        savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
                        sub_fig.savefig(savename,dpi=300)
                        plt.close()

                        #plot itc plot
                        sub_fig,sub_ax1 = plt.subplots(figsize=(6.4,4.8), layout='constrained')
                        pc = sub_ax1.pcolormesh(epochs.times,np.arange(freq_min,freq_max,1),np.squeeze(np.mean(norm_itc,axis=0)),vmin = vmin, vmax = vmax)
                        #save temporary plot for report
                        if hemi == 'lh': 
                            title = 'ITC for ' + condition_type + ' \n Left Hemisphere'
                        else:
                            title = 'ITC for ' + condition_type + ' \n Right Hemisphere'
                        
                        sub_ax1.set_title(title,fontsize=16)
                        sub_fig.colorbar(pc)
                        
                        fig_to_save = sub_fig.get_figure()
                        savetitle = '_'.join([condition_type,hemi,'itc','plot'])
                        savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
                        sub_fig.savefig(savename,dpi=300)
                        plt.close()

                    #combine plots for each hemisphere
                    fig1 = plt.figure(figsize=(12.8,4.8), layout='constrained')
                    gs  = GridSpec(1, 2, figure=fig1) 
                    ax1 = fig1.add_subplot(gs[0,0])
                    ax2 = fig1.add_subplot(gs[0,1])
                    savetitle = '_'.join([condition_type,'lh','power','plot'])
                    savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
                    ax1.imshow(plt.imread(savename))
                    ax1.axis('off')
                    savetitle = '_'.join([condition_type,'rh','power','plot'])
                    savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
                    ax2.imshow(plt.imread(savename))
                    ax2.axis('off')
                    #add to report
                    title = '_'.join([participant,condition_type,'power'])
                    report.add_figure(fig=fig1, title=title, section=participant, tags=[diagnosis,condition_type,'power'])

                    #combine plots for each hemisphere
                    fig1 = plt.figure(figsize=(12.8,4.8), layout='constrained')
                    gs  = GridSpec(1, 2, figure=fig1) 
                    ax1 = fig1.add_subplot(gs[0,0])
                    ax2 = fig1.add_subplot(gs[0,1])
                    savetitle = '_'.join([condition_type,'lh','itc','plot'])
                    savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
                    ax1.imshow(plt.imread(savename))
                    ax1.axis('off')
                    savetitle = '_'.join([condition_type,'rh','itc','plot'])
                    savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
                    ax2.imshow(plt.imread(savename))
                    ax2.axis('off')
                    #add to report
                    title = '_'.join([participant,condition_type,'itc'])
                    report.add_figure(fig=fig1, title=title, section=participant, tags=[diagnosis,condition_type,'itc'])
                    
df = pd.DataFrame(all_participants, columns = ['Diagnosis','Participant','Condition','label','hemisphere','power','itc','time']) 
df.to_pickle(data_savename)

df_miso = df[df["Diagnosis"]=='misophonia']

for diagnosis in diagnoses:
    for condition_type in condition:
        for hemi in hemisphere:
            df_to_plot = df_miso.loc[(df_miso['hemisphere']== hemi) & (df_miso['Condition']==condition_type)]
            data_to_plot = df_to_plot['power'].mean()
            #plot time-frequency plot
            sub_fig,sub_ax1 = plt.subplots(figsize=(6.4,4.8), layout='constrained')
            pc = sub_ax1.pcolormesh(epochs.times,np.arange(freq_min,freq_max,1),np.squeeze(data_to_plot),vmin = vmin, vmax = vmax)
            #save temporary plot for report
            if hemi == 'lh': 
                title = 'Time-frequency plot for ' + condition_type + ' \n Left Hemisphere Grand Average'
            else:
                title = 'Time-frequency plot for ' + condition_type + ' \n Right Hemisphere Grand Average'
            sub_ax1.set_title(title,fontsize=16)
            sub_fig.colorbar(pc)
            
            fig_to_save = sub_fig.get_figure()
            savetitle = condition_type + '_' + hemi + '_' + 'gavg_power_plot'
            savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
            sub_fig.savefig(savename,dpi=300)

            data_to_plot = df_to_plot['itc'].mean()
            #plot time-frequency plot
            sub_fig,sub_ax1 = plt.subplots(figsize=(6.4,4.8), layout='constrained')
            pc = sub_ax1.pcolormesh(epochs.times,np.arange(freq_min,freq_max,1),np.squeeze(data_to_plot),vmin = vmin, vmax = vmax)
            #save temporary plot for report
            if hemi == 'lh': 
                title = 'Time-frequency plot for ' + condition_type + ' \n Left Hemisphere Grand Average'
            else:
                title = 'Time-frequency plot for ' + condition_type + ' \n Right Hemisphere Grand Average'
            sub_ax1.set_title(title,fontsize=16)
            sub_fig.colorbar(pc)
            
            fig_to_save = sub_fig.get_figure()
            savetitle = condition_type + '_' + hemi + '_' + 'gavg_itc_plot'
            savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
            sub_fig.savefig(savename,dpi=300)

        #combine plots for each hemisphere
        fig1 = plt.figure(figsize=(12.8,4.8), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig1) 
        ax1 = fig1.add_subplot(gs[0,0])
        ax2 = fig1.add_subplot(gs[0,1])
        savetitle = condition_type + '_lh_' + 'gavg_power_plot'
        savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
        ax1.imshow(plt.imread(savename))
        ax1.axis('off')
        savetitle = condition_type + '_rh_' + 'gavg_power_plot'
        savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
        ax2.imshow(plt.imread(savename))
        ax2.axis('off')
        #add to report
        title = '_'.join(['grand-average',condition_type,'gavg_power_plot'])
        report.add_figure(fig=fig1, title=title, section='grand-average', tags=[diagnosis,condition_type,'power'])

        #combine plots for each hemisphere
        fig1 = plt.figure(figsize=(12.8,4.8), layout='constrained')
        gs  = GridSpec(1, 2, figure=fig1) 
        ax1 = fig1.add_subplot(gs[0,0])
        ax2 = fig1.add_subplot(gs[0,1])
        savetitle = condition_type + '_lh_' + 'gavg_itc_plot'
        savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
        ax1.imshow(plt.imread(savename))
        ax1.axis('off')
        savetitle = condition_type + '_rh_' + 'gavg_itc_plot'
        savename = os.path.join(connectivity_output_dir, savetitle + ".tiff")
        ax2.imshow(plt.imread(savename))
        ax2.axis('off')
        #add to report
        title = '_'.join(['grand-average',condition_type,'gavg_itc_plot'])
        report.add_figure(fig=fig1, title=title, section='grand-average', tags=[diagnosis,condition_type,'itc'])

report_savename = os.path.join(connectivity_output_dir,save_fname + '.html')
report.save(report_savename, overwrite=True)