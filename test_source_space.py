#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 14:41:26 2023

@author: jwt30
"""

import mne 
import os
from autoreject import get_rejection_threshold

def get_evoked(epochs,stimuli):
    condition_epochs = epochs[stimuli]
    reject = get_rejection_threshold(condition_epochs, ch_types=['mag','grad'], decim=2)
    condition_epochs.drop_bad(reject=reject)
    evoked = condition_epochs.average()
    return evoked



paradigm = 'Misophonia'
# # participants = {}
# # participants['miso'] = ['118601','118801','119001','125101','124201']
# # participants['TD'] = ['116201','114601']

participant = '129901'

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant)

for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if 'run' in filename:
            continue
        if '_epo.fif' in filename:
            epo_file = os.path.join(path,filename)
            epochs = mne.read_epochs(epo_file)

            evoked = get_evoked(epochs,'misophone')

            


# noise_cov = mne.compute_covariance(
#     epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
# )

#fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, evoked1.info)

#evoked1.plot(time_unit="s")

#mne.bem.make_watershed_bem('118601_20230822', subjects_dir='/autofs/space/transcend/MRI/WMA/recons/')

participant = '102201'

transcend_dir = '/autofs/space/transcend/MEG/'

data_dir = os.path.join(transcend_dir,paradigm,participant)

for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if '_run01_raw.fif' in filename:
            raw_fname = os.path.join(path,filename)

subj_dir='/autofs/space/transcend/MRI/WMA/recons/'

for path, directory_names, filenames in os.walk(subj_dir):
    for dir in directory_names:
        if participant + '_' in dir:
            subjID_date = dir

#mne.gui.coregistration(subject = subjID_date, subjects_dir=subj_dir,inst=raw_fname)

src = mne.setup_source_space(
    subject = subjID_date, spacing="oct6", add_dist="patch", subjects_dir=subj_dir
)

#mne.viz.plot_bem(subject = subjID_date, subjects_dir=subj_dir, src=src)

conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject = subjID_date, ico=4, conductivity=conductivity, subjects_dir=subj_dir
)
bem = mne.make_bem_solution(model)

data_dir = os.path.join(subj_dir,subjID_date)

for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if '_trans.fif' in filename:
            trans_fname = os.path.join(path,filename)

fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans_fname,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant)

for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if '_erm_raw_sss.fif' in filename:
            raw_erm_fname = os.path.join(path,filename)

raw_erm = mne.io.read_raw_fif(raw_erm_fname)

noise_cov =  mne.compute_raw_covariance(raw_erm, tmin=0, method='auto', rank=None, tmax=None)
#noise_cov.plot(raw_erm.info, proj=True)

for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if 'run' in filename:
            continue
        if '_epo.fif' in filename:
            epo_file = os.path.join(path,filename)
            epochs = mne.read_epochs(epo_file)
            epochs_filt = epochs.filter(1, 30)

            evoked1 = get_evoked(epochs_filt,'misophone')
            evoked2 = get_evoked(epochs_filt,'novel')


inverse_operator = mne.minimum_norm.make_inverse_operator(
    evoked1.info, fwd, noise_cov, loose=0.2, depth=0.8
)

method = "sLORETA"
snr = 3.0
lambda2 = 1.0 / snr**2

stc1, residual = mne.minimum_norm.apply_inverse(
    evoked1,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)


initial_time = 0.0
brain = stc1.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

stc2, residual = mne.minimum_norm.apply_inverse(
    evoked2,
    inverse_operator,
    lambda2,
    method=method,
    pick_ori=None,
    return_residual=True,
    verbose=True,
)


initial_time = 0.0
brain = stc2.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant)
fname1 = os.path.join(data_dir,participant + '_misophones')
stc1.save(fname1)

fname2 = os.path.join(data_dir,participant + '_novels')
stc2.save(fname2,overwrite=True)

participant = '124201'
subj_dir='/autofs/space/transcend/MRI/WMA/recons/'

for path, directory_names, filenames in os.walk(subj_dir):
    for dir in directory_names:
        if participant + '_' in dir:
            subjID_date = dir

local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(local_dir,paradigm,participant)

load_fname1_lh = os.path.join(data_dir,participant + '_misophones-lh.stc')
load_fname1_rh = os.path.join(data_dir,participant + '_misophones-rh.stc')

load_fname2_lh = os.path.join(data_dir,participant + '_novels-lh.stc')
load_fname2_rh = os.path.join(data_dir,participant + '_novels-rh.stc')

stc_misophones_lh = mne.read_source_estimate(load_fname1_lh,subject=subjID_date)
stc_misophones_rh = mne.read_source_estimate(load_fname1_rh,subject=subjID_date)

stc_novels_lh = mne.read_source_estimate(load_fname2_lh,subject=subjID_date)
stc_novels_rh = mne.read_source_estimate(load_fname2_rh,subject=subjID_date)

load_fname = os.path.join(data_dir,participant + '_misophones-lh.stc')
stc_miso = mne.read_source_estimate(load_fname,subject=subjID_date)

initial_time = 0.0
brain = stc_miso.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='both',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

initial_time = 0.0
brain = stc_misophones_lh.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='lh',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

initial_time = 0.0
brain = stc_misophones_rh.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='rh',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

initial_time = 0.0
brain = stc_novels_lh.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='lh',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

initial_time = 0.0
brain = stc_novels_rh.plot(
    subjects_dir='/autofs/space/transcend/MRI/WMA/recons/',
    hemi='rh',
    initial_time=initial_time,
    clim=dict(kind="value", lims=[3, 6, 9]),
    smoothing_steps=7,
)

import mne
import os
import pandas as pd
import seaborn as sns, matplotlib.pyplot as plt
import numpy as np

paradigm = 'Misophonia'
participants = {}
participants['miso'] = ['118601','118801','119001','125101','124201']
participants['TD'] = ['116201','114601']

subj_dir='/autofs/space/transcend/MRI/WMA/recons/'

participants_data = []

for participant in participants['miso']:
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                subjID_date = dir

    local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
    data_dir = os.path.join(local_dir,paradigm,participant)

    #load label
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'miso_ofc-lh.label' in filename:
                label_fname = os.path.join(path,filename)
    label = [mne.read_label(label_fname)]

    load_fname = os.path.join(data_dir,participant + '_misophones-lh.stc')
    stc_miso = mne.read_source_estimate(load_fname,subject=subjID_date)

    load_fname = os.path.join(data_dir,participant + '_novels-lh.stc')
    stc_novel = mne.read_source_estimate(load_fname,subject=subjID_date)


    stc_label_miso_lh   = stc_miso.in_label(label[0])
    stc_label_novel_lh   = stc_novel.in_label(label[0])

    data_miso_lh = np.abs(stc_label_miso_lh.data)
    data_novel_lh = np.abs(stc_label_novel_lh.data)

    data_miso_lh_mean = np.mean(data_miso_lh,axis=0)
    data_novel_lh_mean = np.mean(data_novel_lh,axis=0)
    time = stc_label_miso_lh.times
    time_idx = (time>=0.35) & (time<=0.55)

    miso_mean = np.mean(data_miso_lh_mean[time_idx])
    novel_mean = np.mean(data_novel_lh_mean[time_idx])

    participant_data_miso = ['Misophonic',participant,'misophonic trigger',miso_mean]
    participant_data_novel = ['Misophonic',participant,'novel',novel_mean]
    
    participants_data.append(participant_data_miso)
    participants_data.append(participant_data_novel)


for participant in participants['TD']:
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                subjID_date = dir

    local_dir = '/local_mount/space/hypatia/2/users/Jasmine'
    data_dir = os.path.join(local_dir,paradigm,participant)

    #load label
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'miso_ofc-lh.label' in filename:
                label_fname = os.path.join(path,filename)
    label = [mne.read_label(label_fname)]

    load_fname = os.path.join(data_dir,participant + '_misophones-lh.stc')
    stc_miso = mne.read_source_estimate(load_fname,subject=subjID_date)

    load_fname = os.path.join(data_dir,participant + '_novels-lh.stc')
    stc_novel = mne.read_source_estimate(load_fname,subject=subjID_date)


    stc_label_miso_lh   = stc_miso.in_label(label[0])
    stc_label_novel_lh   = stc_novel.in_label(label[0])

    data_miso_lh = np.abs(stc_label_miso_lh.data)
    data_novel_lh = np.abs(stc_label_novel_lh.data)

    data_miso_lh_mean = np.mean(data_miso_lh,axis=0)
    data_novel_lh_mean = np.mean(data_novel_lh,axis=0)
    time = stc_label_miso_lh.times
    time_idx = (time>=0.38) & (time<=0.48)

    miso_mean = np.mean(data_miso_lh_mean[time_idx])
    novel_mean = np.mean(data_novel_lh_mean[time_idx])

    participant_data_miso = ['TD',participant,'misophonic trigger',miso_mean]
    participant_data_novel = ['TD',participant,'novel',novel_mean]
    
    participants_data.append(participant_data_miso)
    participants_data.append(participant_data_novel)

df = pd.DataFrame(participants_data, columns = ['Group','Participant', 'Sound_type','Activation(AU)']) 

df=df.replace(to_replace='misophonic trigger', value='trigger', regex=True)
df=df.replace(to_replace='novel', value='distractor', regex=True)

df=df.replace(to_replace='Misophonic', value='Misophonic (n=5)', regex=True)
df=df.replace(to_replace='TD', value='TD (n=2)', regex=True)

fig, ax = plt.subplots()
# sns.set(rc={'figure.figsize':(13.7,8.27)})
# sns.axes_style('whitegrid')
# sns.set(font_scale=3)

ax = sns.barplot(x="Group", y="Activation(AU)",
            hue="Sound_type", palette=["#1b699e", "#ca6723"],
            data=df,capsize=.1, errorbar="sd")
ax.set_xlabel("")
ax.set_ylabel("Activation(AU)")

ax = sns.swarmplot(x="Group", y="Activation(AU)", hue="Sound_type",data=df, alpha=.8,dodge=True,size=20).set(title = "Activations for Misophonic Triggers \n compared to Non-Misophonic Distractors")


from matplotlib.pyplot import cm
color = cm.rainbow(np.linspace(0, 1, 7))
for i, c in enumerate(color):
    color[i,3] = 0.5

participants_list = participants['miso']+participants['TD']
participants_color = dict.fromkeys(participants_list)

for i in range(0,7):
    participants_color[participants_list[i]]=color[i]


fsaverageDir = '/local_mount/space/hypatia/2/users/Jasmine/MNE-sample-data/subjects/'
local_dir = '/local_mount/space/hypatia/2/users/Jasmine'

brain = mne.viz.Brain(subject = 'fsaverage',hemi = 'lh',views = 'lateral',subjects_dir = fsaverageDir,surf='inflated',background='white')

for participant in participants_list:
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                subjID_date = dir

    
    data_dir = os.path.join(local_dir,paradigm,participant)

    #load label
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'miso_ofc-lh.label' in filename:
                label_fname = os.path.join(path,filename)
    label = [mne.read_label(label_fname,subject= subjID_date)]

    morphed_label_lh = mne.morph_labels(label, subject_to='fsaverage', subject_from=subjID_date, subjects_dir=subj_dir, surf_name='inflated')
    brain.add_label(morphed_label_lh[0], hemi = 'lh', alpha=1)



paradigm = 'Misophonia'
participants = {}
participants['miso'] = ['118601','118801','119001','125101','124201']
participants['TD'] = ['116201','114601']

subj_dir='/autofs/space/transcend/MRI/WMA/recons/'
local_dir = '/local_mount/space/hypatia/2/users/Jasmine'

participants_miso = []
participants_novel = []

for participant in participants['miso']:
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                subjID_date = dir

    
    data_dir = os.path.join(local_dir,paradigm,participant)

    #load label
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'miso_ofc-lh.label' in filename:
                label_fname = os.path.join(path,filename)
    label = [mne.read_label(label_fname)]

    load_fname = os.path.join(data_dir,participant + '_misophones-lh.stc')
    stc_miso = mne.read_source_estimate(load_fname,subject=subjID_date)

    load_fname = os.path.join(data_dir,participant + '_novels-lh.stc')
    stc_novel = mne.read_source_estimate(load_fname,subject=subjID_date)


    stc_label_miso_lh   = stc_miso.in_label(label[0])
    stc_label_novel_lh   = stc_novel.in_label(label[0])

    participants_miso.append(stc_label_miso_lh.data[0])
    participants_novel.append(stc_label_novel_lh.data[0])

miso_stc_ave = np.mean(participants_miso,axis=0)
novel_stc_ave = np.mean(participants_novel,axis=0)

fontsize = 20
sns.set(style="white")
sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')

sub_ax1.plot(stc_label_miso_lh.times,miso_stc_ave, label='trigger')
sub_ax1.plot(stc_label_novel_lh.times,novel_stc_ave, label='distractor')
sub_ax1.set_xlim([-0.2,0.6])
sub_ax1.set_ylim([0,50])
sub_ax1.tick_params(labelsize=fontsize)
sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
sub_ax1.set_ylabel('sLORETA activation (AU)',fontsize=fontsize)
sub_ax1.axvline(x=0, ls='--', color='k')
sub_ax1.set_title('Grand-averaged activation in prefrontal ROIs \n Misophonia',fontsize=24)
sub_ax1.legend(fontsize=fontsize)
sub_ax1.axvspan(0.38, 0.48, color='black', alpha=.15)

participants_miso = []
participants_novel = []

for participant in participants['TD']:
    for path, directory_names, filenames in os.walk(subj_dir):
        for dir in directory_names:
            if participant + '_' in dir:
                subjID_date = dir

    
    data_dir = os.path.join(local_dir,paradigm,participant)

    #load label
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'miso_ofc-lh.label' in filename:
                label_fname = os.path.join(path,filename)
    label = [mne.read_label(label_fname)]

    load_fname = os.path.join(data_dir,participant + '_misophones-lh.stc')
    stc_miso = mne.read_source_estimate(load_fname,subject=subjID_date)

    load_fname = os.path.join(data_dir,participant + '_novels-lh.stc')
    stc_novel = mne.read_source_estimate(load_fname,subject=subjID_date)


    stc_label_miso_lh   = stc_miso.in_label(label[0])
    stc_label_novel_lh   = stc_novel.in_label(label[0])

    participants_miso.append(stc_label_miso_lh.data[0])
    participants_novel.append(stc_label_novel_lh.data[0])

miso_stc_ave = np.mean(participants_miso,axis=0)
novel_stc_ave = np.mean(participants_novel,axis=0)


sns.set(style="white")
sub_fig,sub_ax1 = plt.subplots(figsize=(10,4), layout='constrained')
# time = stc_label_miso_lh.times
# time_idx = (time>=-0.2) & (time<=0)
# data = miso_stc_ave - np.mean(miso_stc_ave[time_idx])
sub_ax1.plot(stc_label_miso_lh.times,miso_stc_ave, label='trigger')
sub_ax1.plot(stc_label_novel_lh.times,novel_stc_ave, label='distractor')
sub_ax1.set_xlim([-0.2,0.6])
sub_ax1.set_ylim([0,50])
sub_ax1.tick_params(labelsize=fontsize)
sub_ax1.set_xlabel('Time (s)',fontsize=fontsize)
sub_ax1.set_ylabel('sLORETA activation (AU)',fontsize=fontsize)
sub_ax1.axvline(x=0, ls='--', color='k')
sub_ax1.set_title('Typically developing',fontsize=24)
sub_ax1.legend(fontsize=fontsize,loc='upper left')
sub_ax1.axvspan(0.38, 0.48, color='black', alpha=.15)

report = mne.Report(title="STCs: Misophones vs Novels")
report.add_image(
    image=image_path, title="Participant", caption="Misophones"
)
report.save("report_custom_image.html", overwrite=True)