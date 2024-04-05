#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:13:09 2023

@author: jwt30
"""

import os
import mne
from autoreject import get_rejection_threshold
import matplotlib.pyplot as plt
import numpy
 
paradigm = 'Misophonia'
participant = '113201'
date = 20230804

left_auditory = ['MEG0131','MEG0141','MEG1511','MEG1541','MEG1521', 'MEG1611','MEG1621', 'MEG0231','MEG0241','MEG0211']
right_auditory = ['MEG1441','MEG1321','MEG1341','MEG2411','MEG2421', 'MEG2641','MEG2621', 'MEG2611','MEG1331','MEG1431']
central = ['MEG0711','MEG1821','MEG1831','MEG2241','MEG2211','MEG0721','MEG0741','MEG0731']
frontal = ['MEG0811', 'MEG0521','MEG0511','MEG0311','MEG0541','MEG0611','MEG0531','MEG1011','MEG0821','MEG0941','MEG1021','MEG0931', 'MEG0921','MEG1211','MEG0911']

#MEG_dir = '/autofs/space/megraid_research/MEG/tal'
transcend_dir = '/local_mount/space/hypatia/2/users/Jasmine'

data_dir = os.path.join(transcend_dir,paradigm,participant,'visit_' + str(date) + '/')

epo_file = os.path.join(data_dir, participant + '_' + paradigm + '_' + str(date) + '_epo.fif')
epochs = mne.read_epochs(epo_file)

stimuli = "misophone"
cond1 = epochs[stimuli]

reject = get_rejection_threshold(cond1, ch_types=['mag','grad'], decim=2)
cond1.drop_bad(reject = reject)

cond1.average().savgol_filter(30).plot_joint()
miso_ave = cond1.average().savgol_filter(10)
mne.viz.plot_evoked(miso_ave, picks = 'mag',ylim=dict(mag=[-600,750]),spatial_colors= 'auto')

# out_fname = participant + '_' + paradigm + '_' + str(date) + '_' + stimuli +'_ave.fif'
# cond1.save(os.path.join(data_dir,out_fname))

stimuli = "standard"
cond2 = epochs[stimuli]

reject = get_rejection_threshold(cond2, ch_types=['mag','grad'], decim=2)
cond2.drop_bad(reject = reject)

cond2.average().savgol_filter(30).plot_joint()
standard_ave = cond2.average().savgol_filter(10)
mne.viz.plot_evoked(standard_ave, picks = 'mag',ylim=dict(mag=[-600,750]),spatial_colors= 'auto')

# out_fname = participant + '_' + paradigm + '_' + str(date) + '_' + stimuli +'_ave.fif'
# cond2.save(os.path.join(data_dir,out_fname))


stimuli = "novel"
cond3 = epochs[stimuli]

reject = get_rejection_threshold(cond3, ch_types=['mag','grad'], decim=2)
cond3.drop_bad(reject = reject)

cond3.average().savgol_filter(30).plot_joint()
nov_ave = cond3.average().savgol_filter(10)
mne.viz.plot_evoked(nov_ave, picks = 'mag',ylim=dict(mag=[-300,300]),spatial_colors='auto')

misodata = miso_ave.get_data(picks = central,units = dict(grad='fT/cm',mag = 'fT'),tmin = 0.1,tmax = 0.4)
novdata = nov_ave.get_data(picks = central,units = dict(grad='fT/cm',mag = 'fT'),tmin = 0.1,tmax = 0.4)
names = ['Misophonic Triggers', 'Novels']
values = [numpy.average(misodata),numpy.average(novdata)]
plt.bar(names,values)
plt.suptitle('Amplitude differences between conditions \n in central sensors (fT)')

# out_fname = participant + '_' + paradigm + '_' + str(date) + '_' + stimuli +'_ave.fif'
# cond3.save(os.path.join(data_dir,out_fname))


plot_epochs = {'misophone':cond1,
               'novel': cond3}

mne.viz.plot_compare_evokeds(plot_epochs, picks= right_auditory, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= left_auditory, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True)

participants = {
    '113201':{'misophone':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/113201/visit_20230804/113201_Misophonia_20230804_misophone_ave.fif',
              'novel':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/113201/visit_20230804/113201_Misophonia_20230804_novel_ave.fif'},
    '113301':{'misophone': '/local_mount/space/hypatia/2/users/Jasmine/Misophonia/113301/visit_20230619/113301_Misophonia_20230619_misophone_ave.fif',
              'novel':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/113301/visit_20230619/113301_Misophonia_20230619_novel_ave.fif'},
    '118601':{'misophone':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/118601/visit_20230822/118601_Misophonia_20230822_misophone_ave.fif',
              'novel':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/118601/visit_20230822/118601_Misophonia_20230822_novel_ave.fif'},
    '116201':{'misophone':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/116201/visit_20230719/116201_Misophonia_20230719_misophone_ave.fif',
              'novel':'/local_mount/space/hypatia/2/users/Jasmine/Misophonia/116201/visit_20230719/116201_Misophonia_20230719_novel_ave.fif'},
    '114601':{'misophone': '/local_mount/space/hypatia/2/users/Jasmine/Misophonia/114601/visit_20230818/114601_Misophonia_20230818_misophone_ave.fif',
              'novel': '/local_mount/space/hypatia/2/users/Jasmine/Misophonia/114601/visit_20230818/114601_Misophonia_20230818_novel_ave.fif'}}

stimuli = 'misophone'

miso = mne.read_evokeds(participants['113201'][stimuli])
miso1 = mne.read_evokeds(participants['118601'][stimuli])
grand_average_miso = mne.grand_average([miso[0], miso1[0]])
grand_average_miso.savgol_filter(10)

stimuli = 'novel'

miso = mne.read_evokeds(participants['113201'][stimuli])
miso1 = mne.read_evokeds(participants['113301'][stimuli])
miso2 = mne.read_evokeds(participants['118601'][stimuli])
grand_average_novel = mne.grand_average([miso[0], miso1[0],miso2[0]])
grand_average_novel.savgol_filter(10)

plot_epochs = {'misophone':grand_average_miso,
               'novel': grand_average_novel}


mne.viz.plot_compare_evokeds(plot_epochs, picks= right_auditory, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= left_auditory, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True)


stimuli = 'misophone'
td = mne.read_evokeds(participants['116201'][stimuli])
td1 = mne.read_evokeds(participants['114601'][stimuli])
grand_average_miso = mne.grand_average([td[0], td1[0]])
grand_average_miso.savgol_filter(10)

stimuli = 'novel'
td = mne.read_evokeds(participants['116201'][stimuli])
td1 = mne.read_evokeds(participants['114601'][stimuli])
grand_average_novel = mne.grand_average([td[0], td1[0]])
grand_average_novel.savgol_filter(10)

plot_epochs = {'misophone':grand_average_miso,
               'novel': grand_average_novel}


mne.viz.plot_compare_evokeds(plot_epochs, picks= right_auditory, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= left_auditory, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True)
mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True)


paradigm = 'Misophonia'
participant = '118601'
date = 20230822
transcend_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(transcend_dir,paradigm,participant,'visit_' + str(date) + '/')

#plot butterfly plots
fname = participant + '_' + paradigm + '_' + str(date) + '_misophone_ave.fif'
miso = mne.read_evokeds(os.path.join(data_dir,fname))
miso[0].savgol_filter(30).plot_joint()

fname = participant + '_' + paradigm + '_' + str(date) + '_novel_ave.fif'
novel = mne.read_evokeds(os.path.join(data_dir,fname))
novel[0].savgol_filter(30).plot_joint()

difference1 = mne.combine_evoked([cond1,cond2],weights = [1,-1])
difference2 = mne.combine_evoked([cond3,cond2],weights = [1,-1])

plot_difference = {'misophone':difference1,
               'novel': difference2}

mne.viz.plot_compare_evokeds(plot_difference, picks= right_auditory, combine = 'mean')

#Plot by time
paradigm = 'Misophonia'
miso = ['113201','118601']
TD = ['116201','114601']

transcend_dir = '/local_mount/space/hypatia/2/users/Jasmine'
data_dir = os.path.join(transcend_dir,paradigm,participant,'visit_' + str(date) + '/')

for participant in TD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'run' in filename:
                continue
            if '_epo.fif' in filename:
                epo_file = os.path.join(path,filename)
                epochs = mne.read_epochs(epo_file)
                stimuli = "novel"
                cond1 = epochs[stimuli]
                reject = get_rejection_threshold(cond1, ch_types=['mag','grad'], decim=2)
                cond1.drop_bad(reject = reject)

                def split(a, n):
                    k, m = divmod(len(a), n)
                    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

                time_windows = list(split(range(numpy.size(cond1,0)), 3))

                window_erps =[]
                for index, item in enumerate(time_windows):
                    window_erp = cond1[time_windows[index]].average()
                    window_erp.comment = 'Time Window ' + str(index+1) + ' in experiment'
                    window_erps.append(window_erp.savgol_filter(10))

                mne.viz.plot_compare_evokeds(window_erps, picks= right_auditory, combine = 'mean',show_sensors=True,
                                             title = 'Evoked responses from right auditory areas \n Participant ' + participant)
                mne.viz.plot_compare_evokeds(window_erps, picks= left_auditory, combine = 'mean',show_sensors=True,
                                             title = 'Evoked responses from left auditory areas \n Participant ' + participant)
                mne.viz.plot_compare_evokeds(window_erps, picks= central, combine = 'mean',show_sensors=True,
                                             title = 'Evoked responses from central areas \n Participant ' + participant)
                mne.viz.plot_compare_evokeds(window_erps, picks= frontal, combine = 'mean',show_sensors=True,
                                             title = 'Evoked responses from frontal areas \n Participant ' + participant)




#load and concatenate tsss
paradigm = 'Misophonia'
participant = '113201'
date = 20230804

transcend_dir = '/local_mount/space/hypatia/2/users/Jasmine'

data_dir = os.path.join(transcend_dir,paradigm,participant,'visit_' + str(date) + '/')

filename = participant + '_' + paradigm + '_run01_raw_tsss.fif'

raw_sss = mne.io.read_raw_fif(os.path.join(data_dir,filename), preload=True, verbose=False)

out_fname = os.path.join(data_dir, participant + '_' + paradigm + '_' + str(date) + '_tsss_nobadseg_epo.fif')

all_epochs = []

event_dict = {'attendRight/standard/high/right': 1,
                     'attendRight/standard/low/right': 3,
                     'attendRight/target/high/right': 11,
                     'attendRight/target/low/right': 13,
                     'attendRight/beep/low/left': 5,
                     'attendRight/beep/high/left': 7,
                     'attendRight/dev/low/left': 35,
                     'attendRight/dev/high/left': 37,
                     'attendRight/novel/low/left': 25,
                     'attendRight/novel/high/left': 27,
                     'attendRight/misophone/low/left': 45,
                     'attendRight/misophone/high/left': 47,
                     'attendLeft/standard/high/left': 2,
                     'attendLeft/standard/low/left': 4,
                     'attendLeft/target/high/left': 12,
                     'attendLeft/target/low/left': 14,
                     'attendLeft/beep/low/right': 6,
                     'attendLeft/beep/high/right': 8,
                     'attendLeft/dev/low/right': 36,
                     'attendLeft/dev/high/right': 38,
                     'attendLeft/novel/low/right': 26,
                     'attendLeft/novel/high/right': 28,
                     'attendLeft/misophone/low/right': 46,
                     'attendLeft/misophone/high/right': 48
                     }

# first_run_file = os.path.join(data_dir,participant + '_' + paradigm + '_run01_raw.fif')
# first_run = mne.io.read_raw_fif(first_run_file)
# first_run_info = first_run.info['dev_head_t']

for path, directory_names, filenames in os.walk(data_dir):
    for filename in filenames:
        if '_tsss.fif' in filename: 
            raw_sss = mne.io.read_raw_fif(os.path.join(data_dir,filename), preload=True, verbose=False)
            # raw_sss.info['dev_head_t'] = first_run_info
            events = mne.find_events(raw_sss, stim_channel='STI101', shortest_event=1)
            epochs = mne.Epochs(raw_sss, events=events, tmin=-0.2, tmax=1, baseline = (-0.2,0.0),event_id=event_dict, on_missing='ignore',reject=None)
            
            all_epochs.append(epochs)

all_epochs  = mne.concatenate_epochs(all_epochs)

all_epochs.save(os.path.join(data_dir,out_fname))
                
raw_file = os.path.join(data_dir, participant + '_' + paradigm + '_run01_raw_tsss.fif')
raw = mne.io.read_raw_fif(raw_file)
events = mne.find_events(raw, stim_channel='STI101',uint_cast=True)

import os
import mne
from autoreject import get_rejection_threshold
import matplotlib.pyplot as plt

paradigm = 'Misophonia'
miso = ['118601','118801','119001','125101']
TD = ['116201','114601']

transcend_dir = '/local_mount/space/hypatia/2/users/Jasmine'

#butterfly plots
condition_ave = []

for participant in TD:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'run' in filename:
                continue
            if '_epo.fif' in filename:
                epo_file = os.path.join(path,filename)
                epochs = mne.read_epochs(epo_file)
                stimuli = "novel"
                cond1 = epochs[stimuli]
                reject = get_rejection_threshold(cond1, ch_types=['mag','grad'], decim=2)
                cond1.drop_bad(reject = reject)
                participant_ave = cond1.average()
                participant_ave.savgol_filter(30).plot_joint(ts_args = dict(ylim = dict(mag=[-700, 700],grad = [-200,300]))) #ts_args = dict(ylim = dict(mag=[-700, 700],grad = [-200,300]))
                condition_ave.append(participant_ave)
                
grand_average = mne.grand_average(condition_ave)                
                
grand_average.savgol_filter(30).plot_joint(ts_args = dict(ylim = dict(mag=[-400, 400],grad = [-100,100])))    #ts_args = dict(ylim = dict(mag=[-700, 700],grad = [-200,300]))

#central ERPs
left_auditory = ['MEG0131','MEG0141','MEG1511','MEG1541','MEG1521', 'MEG1611','MEG1621', 'MEG0231','MEG0241','MEG0211']
right_auditory = ['MEG1441','MEG1321','MEG1341','MEG2411','MEG2421', 'MEG2641','MEG2621', 'MEG2611','MEG1331','MEG1431']
central = ['MEG0711','MEG1821','MEG1831','MEG2241','MEG2211','MEG0721','MEG0741','MEG0731']
frontal = ['MEG0811', 'MEG0521','MEG0511','MEG0311','MEG0541','MEG0611','MEG0531','MEG1011','MEG0821','MEG0941','MEG1021','MEG0931', 'MEG0921','MEG1211','MEG0911']

condition1_ave = []
condition2_ave = []


for participant in miso:
    
    data_dir = os.path.join(transcend_dir,paradigm,participant + '/')
    
    for path, directory_names, filenames in os.walk(data_dir):
        for filename in filenames:
            if 'run' in filename:
                continue
            if '_epo.fif' in filename:
                epo_file = os.path.join(path,filename)
                epochs = mne.read_epochs(epo_file)
                
                stimuli = "novel"
                cond1 = epochs[stimuli]
                reject = get_rejection_threshold(cond1, ch_types=['mag','grad'], decim=2)
                cond1.drop_bad(reject = reject)
                cond1_ave = cond1.average().savgol_filter(10)

                condition1_ave.append(cond1_ave)
                
                stimuli = "misophone"
                cond2 = epochs[stimuli]
                reject = get_rejection_threshold(cond2, ch_types=['mag','grad'], decim=2)
                cond2.drop_bad(reject = reject)
                cond2_ave = cond2.average().savgol_filter(10)

                condition2_ave.append(cond2_ave)
                
                plot_epochs = {'misophone':cond2_ave,
                               'novel': cond1_ave}
                
                mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=False,
                                 title = 'Participant ' + participant)
                plt.savefig('Participant'+ participant + '_misoVsNovel_evoked.tiff',dpi=300)





grand_average_cond1 = mne.grand_average(condition1_ave)  

grand_average_cond2 = mne.grand_average(condition2_ave)  

plot_epochs = {'misophone':grand_average_cond2,
               'novel': grand_average_cond1}               
                
mne.viz.plot_compare_evokeds(plot_epochs, picks= right_auditory, combine = 'mean',show_sensors=True,
                             title = 'Evoked responses from right auditory areas \n (Grand-averaged response)')
mne.viz.plot_compare_evokeds(plot_epochs, picks= left_auditory, combine = 'mean',show_sensors=True,
                             title = 'Evoked responses from left auditory areas \n (Grand-averaged response)')
mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=True,
                             title = 'Evoked responses from central areas \n (Grand-averaged response)')
mne.viz.plot_compare_evokeds(plot_epochs, picks= frontal, combine = 'mean',show_sensors=True,
                             title = 'Evoked responses from frontal areas \n (Grand-averaged response)')


fig, axs = plt.subplots(1, len(miso),figsize = [12.8,9.6],sharex=True,sharey=True,dpi=80)

for n in range(len(miso)):
                                            
    cond1_ave = condition1_ave[n]
    cond2_ave = condition2_ave[n]
    
    
    plot_epochs = {'misophone':cond2_ave,
                   'novel': cond1_ave}
    

    mne.viz.plot_compare_evokeds(plot_epochs, picks= central, combine = 'mean',show_sensors=False, axes = axs[n],
                                 title = miso[n])
