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
from mne.minimum_norm import apply_inverse_epochs
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
savedir = os.path.join(data_dir,'Poster_images','Jun2024','central_sulcus')

#get stcs for each epoch
method = "dSPM"
snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr**2


condition = ['Misophone','Novels']
hemisphere = ['lh','rh']
labels_to_compare = ['aud','central']
#freq_bands = ['beta','gamma']
freq_bands = ['theta','alpha','beta','low_gamma','broad_gamma','high_gamma']

save_fname = '_'.join(condition + labels_to_compare + freq_bands)
#connectivity settings                    
fmin = (4.0,8.0,15.0,35.0,35.0,60.0)
fmax = (8.0,12.0,30.0,55.0,80.0,80.0)
tmin_con = 0.0



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
                epochs = mne.read_epochs(epoch_name)
                for hemi in hemisphere:
                    seed_target_labels = {}
                    label_name = labels_to_compare[0] + '_' + hemi +'.label'
                    fname_label = find_files(label_name,participant_dir)[0]
                    seed_target_labels['seed'] = mne.read_label(fname_label)
                    label_name = labels_to_compare[1] + '_' + hemi +'.label'
                    fname_label = find_files(label_name,participant_dir)[0]
                    seed_target_labels['target'] = mne.read_label(fname_label)
                    src = inverse_operator["src"]  # the source space used
                    #extract stc from relevant epochs
                    stcs = apply_inverse_epochs(
                        epochs, inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
                    )
                    
                    #get label time series for visual labels
                    seed_ts = mne.extract_label_time_course(
                        stcs, seed_target_labels['seed'], src, mode="mean_flip", verbose="error"
                    )
                    
                    #get label time series for intraparietal labels
                    stcs = apply_inverse_epochs(
                        epochs, inverse_operator, lambda2, method, pick_ori="normal", return_generator=True
                    )
                    target_ts = mne.extract_label_time_course(
                        stcs, seed_target_labels['target'], src, mode="mean_flip", verbose="error"
                    )
                    
                    #combine stcs
                    comb_ts = list(zip(seed_ts,target_ts))
                    
                    #set up indices
                    indices = seed_target_indices([0],[1])
                    
                    #calculate connectivity (coherence)                    
                    con = spectral_connectivity_epochs(
                        comb_ts,
                        indices=indices,
                        method="coh",
                        fmin = fmin,
                        fmax = fmax,
                        tmin = tmin_con,
                        faverage = True, 
                        sfreq=sfreq,
                        n_jobs=1,
                        gc_n_lags = 40
                    )
                    
                    participant_data = [participant,diagnosis,condition_type,hemi,con.get_data()[0][0],con.get_data()[0][1],con.get_data()[0][2],con.get_data()[0][3],con.get_data()[0][4],con.get_data()[0][5]]
                    all_participants.append(participant_data)

df= pd.DataFrame(all_participants, columns = ['Participant','Diagnosis','Condition','hemisphere'] + freq_bands) 

df=df.replace(to_replace='Misophone', value='trigger', regex=True)
df=df.replace(to_replace='Novels', value='distractor', regex=True)

df=df.replace(to_replace='misophonia', value='Misophonic (n=16)', regex=True)
df=df.replace(to_replace='td', value='TD (n=2)', regex=True)

data_savename = os.path.join(savedir,save_fname + '.csv')
df.to_csv(data_savename,index=False)

#df = pd.read_csv(data_savename)

report_title = '_'.join(condition + labels_to_compare + freq_bands)
report = mne.Report(title=report_title+'coherence')

group = ["Misophonic (n=16)",'TD (n=2)']

for diagnosis in group:
    df_group = df[df["Diagnosis"]==diagnosis]

    for hemi in hemisphere:
        df_to_plot = df_group.loc[df_group['hemisphere']== hemi]
        for freq in freq_bands:

            sns.set_style("whitegrid")
            fig, ax = plt.subplots()

            ax = sns.pointplot(data=df_to_plot, x="Condition", y=freq, hue = "Condition",estimator = "median", linestyles="-")
            plt.setp(ax.collections, sizes=[100])
            if hemi == 'lh':
                title = freq + " connectivity in left hemisphere"
            else:
                title = freq + " connectivity in right hemisphere"
            ax = sns.swarmplot(x="Condition", y=freq, hue = "Condition", data=df_to_plot,alpha=.35,legend=False,size = 10).set(title = title)

            fig_to_save = fig.get_figure()
            savename= os.path.join(savedir, freq + "_" + hemi +".tiff")
            fig.savefig(savename,dpi=300)

            section = 'grand_average'
            if diagnosis == "Misophonic (n=16)":
                group_name = 'Misophonic'
            else:
                group_name = 'TD'
            report.add_figure(fig=fig, title='Grand-averaged coherence', section=section, tags=[freq,hemi,group_name])


report_savename = os.path.join(savedir,save_fname + 'median.html')
report.save(report_savename, overwrite=True)