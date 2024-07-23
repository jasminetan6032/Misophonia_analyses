import mne
import pandas as pd
import numpy as np
import ASSR_config as cfg
import matplotlib.pyplot as plt
from os.path import join, split, exists
from glob import glob

# load alignment file
ASSR_df       = pd.read_csv(join(cfg.paradigm_dir,'alignment_file.csv'), index_col=0, dtype=str)
subject_info  = pd.read_csv(join(cfg.paradigm_dir,'ASSR_subject_info.csv'), dtype=str)

# get subject list
subjects      = list(ASSR_df['subject']) 
subs2exclude  = []
source_method = 'dSPM'
freqs         = np.arange(4,60,1)

# create dictionary for different conditions
pow_jump_asd_lh,pow_jump_asd_rh = [],[]
pow_stay_asd_lh,pow_stay_asd_rh = [],[]
pow_jump_td_lh,pow_jump_td_rh   = [],[]
pow_stay_td_lh,pow_stay_td_rh   = [],[]
itc_jump_asd_lh,itc_jump_asd_rh = [],[]
itc_stay_asd_lh,itc_stay_asd_rh = [],[]
itc_jump_td_lh,itc_jump_td_rh   = [],[]
itc_stay_td_lh,itc_stay_td_rh   = [],[]

# open a report if it already exists or create one if it doesn't
report  = mne.Report(title=source_method+'_'+cfg.report_name) if not exists(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name)) else mne.open_report(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name))

for sub_i in subjects:
    if sub_i not in subs2exclude:

        # set subject paths
        sss_path    = list(ASSR_df['sss path'][ASSR_df['subject'] == sub_i])[0]
        info        = mne.io.read_info(sss_path)
        this_visit  = [i for i in sss_path.split('/') if 'visit' in i][0]
        recons_path = list(ASSR_df['recons path'][ASSR_df['subject'] == sub_i])[0]
        output_dir  = join(cfg.paradigm_dir,sub_i,this_visit)
        subject = split(recons_path)[-1]
        subjects_dir = cfg.recons_dir
        if sub_i in ['082802','082601','082501']:
            subject      = 'fsaverage'
            subjects_dir = '/local_mount/space/tapputi/1/users/sergio/MNE-sample-data/subjects'

        # get diagnosis
        diagnosis   = list(subject_info[subject_info['Subj_ID'] == sub_i]['diagnosis'])[0]

        # load src
        src_fname = glob(join(output_dir,'*src.fif'))[0]
        src = mne.read_source_spaces(src_fname)
        
        # load stc data and pre-defined labels
        stc_fname  = split(sss_path)[1].split('0hp')[0]+'01_120hz.stc'

        # read epochs
        epochs_fname = stc_fname.replace('.stc','_epo.fif')   
        all_epochs  = mne.read_epochs(join(output_dir,epochs_fname),  preload=True)
        if info['sfreq'] > 1000: 
            all_epochs = all_epochs.resample(1000)
        # downsampled data is 1 sample longer. Trim those files
        if len(all_epochs.times) == 2001:
            all_epochs = all_epochs.crop(tmax=all_epochs.times[-2]) 
        jump_epochs = all_epochs['jump'] 
        stay_epochs = all_epochs['stay']

        # get the inverse operator
        cov_path  = glob(join(cfg.transcend_dir,sub_i,this_visit,'epoched','*cov*'))[0]
        noise_cov = mne.read_cov(cov_path)

        # compute the inverse operator
        fwd_fname     = stc_fname.split('_01')[0]+'_fwd.fif'
        fwd           = mne.read_forward_solution(join(output_dir,fwd_fname), verbose=False)
        inv_operator  = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info')

        # load label
        try:
            func_label_rh = mne.read_label(join(output_dir,'corrected_labels',source_method+'_rh.label'))
        except:
            func_label_rh = mne.read_label(join(output_dir,source_method+'_rh.label'))
            
        try:
            func_label_lh = mne.read_label(join(output_dir,'corrected_labels',source_method+'_lh.label'))
        except:    
            func_label_lh = mne.read_label(join(output_dir,source_method+'_lh.label'))  

        # get power estimate from source space using the functional labels
        pow_jump_lh, itc_jump_lh = mne.minimum_norm.source_induced_power(jump_epochs, inv_operator, freqs = freqs, baseline=(-.5,-.05), baseline_mode='percent', label=func_label_lh, method=source_method)
        pow_jump_rh, itc_jump_rh = mne.minimum_norm.source_induced_power(jump_epochs, inv_operator, freqs = freqs, baseline=(-.5,-.05), baseline_mode='percent', label=func_label_rh, method=source_method)
        pow_stay_lh, itc_stay_lh = mne.minimum_norm.source_induced_power(stay_epochs, inv_operator, freqs = freqs, baseline=(-.5,-.05), baseline_mode='percent', label=func_label_lh, method=source_method)
        pow_stay_rh, itc_stay_rh = mne.minimum_norm.source_induced_power(stay_epochs, inv_operator, freqs = freqs, baseline=(-.5,-.05), baseline_mode='percent', label=func_label_rh, method=source_method)     

        pow_jump_lh = (pow_jump_lh - np.min(pow_jump_lh)) / (np.max(pow_jump_lh) - np.min(pow_jump_lh))
        pow_jump_rh = (pow_jump_rh - np.min(pow_jump_rh)) / (np.max(pow_jump_rh) - np.min(pow_jump_rh))
        pow_stay_lh = (pow_stay_lh - np.min(pow_stay_lh)) / (np.max(pow_stay_lh) - np.min(pow_stay_lh))
        pow_stay_rh = (pow_stay_rh - np.min(pow_stay_rh)) / (np.max(pow_stay_rh) - np.min(pow_stay_rh))
        itc_jump_lh = (itc_jump_lh - np.min(itc_jump_lh)) / (np.max(itc_jump_lh) - np.min(itc_jump_lh))
        itc_jump_rh = (itc_jump_rh - np.min(itc_jump_rh)) / (np.max(itc_jump_rh) - np.min(itc_jump_rh))
        itc_stay_lh = (itc_stay_lh - np.min(itc_stay_lh)) / (np.max(itc_stay_lh) - np.min(itc_stay_lh))
        itc_stay_rh = (itc_stay_rh - np.min(itc_stay_rh)) / (np.max(itc_stay_rh) - np.min(itc_stay_rh))

        # get time vector
        times    = jump_epochs.times
        pow_lims = (0,1)

        # plot subject TFR and add to the paradimg report
        fig_pow, axs = plt.subplots(2,2, figsize=(11,7))
        tf1 = axs[0,0].imshow(np.mean(pow_jump_lh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[0,0].set_title('jump - lh')
        axs[0,0].set_ylabel('freqs')
        axs[0,0].set_xticks([])
        axs[0,0].axvline(x=0, ls='--', color='gray')
        axs[0,0].axvline(x=.55, ls='--', color='gray')
        axs[0,0].set_xlim(-.2,1.2)
        tf2 = axs[0,1].imshow(np.mean(pow_jump_rh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[0,1].set_title('jump - rh')
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,1].axvline(x=0, ls='--', color='gray')
        axs[0,1].axvline(x=.55, ls='--', color='gray')
        axs[0,1].set_xlim(-.2,1.2)
        tf3 = axs[1,0].imshow(np.mean(pow_stay_lh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[1,0].set_title('stay - lh')
        axs[1,0].set_ylabel('freqs')
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].axvline(x=0, ls='--', color='gray')
        axs[1,0].axvline(x=.55, ls='--', color='gray')        
        axs[1,0].set_xlim(-.2,1.2)
        tf4 = axs[1,1].imshow(np.mean(pow_stay_rh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[1,1].set_title('stay - rh')
        axs[1,1].set_xlabel('Time (s)')
        axs[1,1].set_yticks([])
        axs[1,1].axvline(x=0, ls='--', color='gray')
        axs[1,1].axvline(x=.55, ls='--', color='gray')
        axs[1,1].set_xlim(-.2,1.2)        
        fig_pow.colorbar(tf4, ax=axs, shrink=0.6, location='right', label= 'power (perc. change)')
        
        # plot subject ITC and add to the paradimg report
        fig_itc, axs = plt.subplots(2,2, figsize=(11,7))
        tf1 = axs[0,0].imshow(np.mean(itc_jump_lh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[0,0].set_title('jump - lh')
        axs[0,0].set_ylabel('freqs')
        axs[0,0].set_xticks([])
        axs[0,0].axvline(x=0, ls='--', color='gray')
        axs[0,0].axvline(x=.55, ls='--', color='gray')
        axs[0,0].set_xlim(-.2,1.2)
        tf2 = axs[0,1].imshow(np.mean(itc_jump_rh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[0,1].set_title('jump - rh')
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks([])
        axs[0,1].axvline(x=0, ls='--', color='gray')
        axs[0,1].axvline(x=.55, ls='--', color='gray')
        axs[0,1].set_xlim(-.2,1.2)
        tf3 = axs[1,0].imshow(np.mean(itc_stay_lh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[1,0].set_title('stay - lh')
        axs[1,0].set_ylabel('freqs')
        axs[1,0].set_xlabel('Time (s)')
        axs[1,0].axvline(x=0, ls='--', color='gray')
        axs[1,0].axvline(x=.55, ls='--', color='gray')        
        axs[1,0].set_xlim(-.2,1.2)
        tf4 = axs[1,1].imshow(np.mean(itc_stay_rh, axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
        axs[1,1].set_title('stay - rh')
        axs[1,1].set_xlabel('Time (s)')
        axs[1,1].set_yticks([])
        axs[1,1].axvline(x=0, ls='--', color='gray')
        axs[1,1].axvline(x=.55, ls='--', color='gray')
        axs[1,1].set_xlim(-.2,1.2)        
        fig_itc.colorbar(tf4, ax=axs, shrink=0.6, location='right', label='ITC')

        # add figures to report
        report.add_figure(fig=fig_pow, title='tf', section=sub_i, tags=(['tf',diagnosis]), replace=True) 
        report.add_figure(fig=fig_itc, title='itc', section=sub_i, tags=(['itc',diagnosis]), replace=True) 

        plt.close('all')

    # add to TD or ASD list
    if 'TD' in diagnosis:
        pow_jump_td_lh.extend([np.mean(pow_jump_lh, axis=0)])
        pow_stay_td_lh.extend([np.mean(pow_stay_lh, axis=0)])
        pow_jump_td_rh.extend([np.mean(pow_jump_rh, axis=0)])
        pow_stay_td_rh.extend([np.mean(pow_stay_rh, axis=0)])
        itc_jump_td_lh.extend([np.mean(itc_jump_lh, axis=0)])
        itc_stay_td_lh.extend([np.mean(itc_stay_lh, axis=0)])
        itc_jump_td_rh.extend([np.mean(itc_jump_rh, axis=0)])
        itc_stay_td_rh.extend([np.mean(itc_stay_rh, axis=0)])        
    else: 
        pow_jump_asd_lh.extend([np.mean(pow_jump_lh, axis=0)])
        pow_stay_asd_lh.extend([np.mean(pow_stay_lh, axis=0)])
        pow_jump_asd_rh.extend([np.mean(pow_jump_rh, axis=0)])
        pow_stay_asd_rh.extend([np.mean(pow_stay_rh, axis=0)]) 
        itc_jump_asd_lh.extend([np.mean(itc_jump_lh, axis=0)])
        itc_stay_asd_lh.extend([np.mean(itc_stay_lh, axis=0)])
        itc_jump_asd_rh.extend([np.mean(itc_jump_rh, axis=0)])
        itc_stay_asd_rh.extend([np.mean(itc_stay_rh, axis=0)])        

# # convert lists to arrays and save in case we need this
# np.save(join(cfg.paradigm_dir,'tf_jump_td_lh.npy'), np.array(pow_jump_td_lh))
# np.save(join(cfg.paradigm_dir,'tf_jump_td_rh.npy'), np.array(pow_jump_td_rh))
# np.save(join(cfg.paradigm_dir,'tf_stay_td_lh.npy'), np.array(pow_stay_td_lh))
# np.save(join(cfg.paradigm_dir,'tf_stay_td_rh.npy'), np.array(pow_stay_td_rh))
# np.save(join(cfg.paradigm_dir,'tf_jump_asd_lh.npy'), np.array(pow_jump_asd_lh))
# np.save(join(cfg.paradigm_dir,'tf_jump_asd_rh.npy'), np.array(pow_jump_asd_rh))
# np.save(join(cfg.paradigm_dir,'tf_stay_asd_lh.npy'), np.array(pow_stay_asd_lh))
# np.save(join(cfg.paradigm_dir,'tf_stay_asd_rh.npy'), np.array(pow_stay_asd_rh))
# np.save(join(cfg.paradigm_dir,'itc_jump_td_lh.npy'), np.array(itc_jump_td_lh))
# np.save(join(cfg.paradigm_dir,'itc_jump_td_rh.npy'), np.array(itc_jump_td_rh))
# np.save(join(cfg.paradigm_dir,'itc_stay_td_lh.npy'), np.array(itc_stay_td_lh))
# np.save(join(cfg.paradigm_dir,'itc_stay_td_rh.npy'), np.array(itc_stay_td_rh))
# np.save(join(cfg.paradigm_dir,'itc_jump_asd_lh.npy'), np.array(itc_jump_asd_lh))
# np.save(join(cfg.paradigm_dir,'itc_jump_asd_rh.npy'), np.array(itc_jump_asd_rh))
# np.save(join(cfg.paradigm_dir,'itc_stay_asd_lh.npy'), np.array(itc_stay_asd_lh))
# np.save(join(cfg.paradigm_dir,'itc_stay_asd_rh.npy'), np.array(itc_stay_asd_rh))

# convert lists to arrays and save in case we need this
np.save(join(cfg.paradigm_dir,'tf_jump_td_lh_norm.npy'), np.array(pow_jump_td_lh))
np.save(join(cfg.paradigm_dir,'tf_jump_td_rh_norm.npy'), np.array(pow_jump_td_rh))
np.save(join(cfg.paradigm_dir,'tf_stay_td_lh_norm.npy'), np.array(pow_stay_td_lh))
np.save(join(cfg.paradigm_dir,'tf_stay_td_rh_norm.npy'), np.array(pow_stay_td_rh))
np.save(join(cfg.paradigm_dir,'tf_jump_asd_lh_norm.npy'), np.array(pow_jump_asd_lh))
np.save(join(cfg.paradigm_dir,'tf_jump_asd_rh_norm.npy'), np.array(pow_jump_asd_rh))
np.save(join(cfg.paradigm_dir,'tf_stay_asd_lh_norm.npy'), np.array(pow_stay_asd_lh))
np.save(join(cfg.paradigm_dir,'tf_stay_asd_rh_norm.npy'), np.array(pow_stay_asd_rh))
np.save(join(cfg.paradigm_dir,'itc_jump_td_lh_norm.npy'), np.array(itc_jump_td_lh))
np.save(join(cfg.paradigm_dir,'itc_jump_td_rh_norm.npy'), np.array(itc_jump_td_rh))
np.save(join(cfg.paradigm_dir,'itc_stay_td_lh_norm.npy'), np.array(itc_stay_td_lh))
np.save(join(cfg.paradigm_dir,'itc_stay_td_rh_norm.npy'), np.array(itc_stay_td_rh))
np.save(join(cfg.paradigm_dir,'itc_jump_asd_lh_norm.npy'), np.array(itc_jump_asd_lh))
np.save(join(cfg.paradigm_dir,'itc_jump_asd_rh_norm.npy'), np.array(itc_jump_asd_rh))
np.save(join(cfg.paradigm_dir,'itc_stay_asd_lh_norm.npy'), np.array(itc_stay_asd_lh))
np.save(join(cfg.paradigm_dir,'itc_stay_asd_rh_norm.npy'), np.array(itc_stay_asd_rh))

# # load data
# pow_jump_td_lh = np.load(join(cfg.paradigm_dir,'tf_jump_td_lh.npy'))
# pow_jump_td_rh = np.load(join(cfg.paradigm_dir,'tf_jump_td_rh.npy'))
# pow_stay_td_lh = np.load(join(cfg.paradigm_dir,'tf_stay_td_lh.npy'))
# pow_stay_td_rh = np.load(join(cfg.paradigm_dir,'tf_stay_td_rh.npy'))
# pow_jump_asd_lh = np.load(join(cfg.paradigm_dir,'tf_jump_asd_lh.npy'))
# pow_jump_asd_rh = np.load(join(cfg.paradigm_dir,'tf_jump_asd_rh.npy'))
# pow_stay_asd_lh = np.load(join(cfg.paradigm_dir,'tf_stay_asd_lh.npy'))
# pow_stay_asd_rh = np.load(join(cfg.paradigm_dir,'tf_stay_asd_rh.npy'))
# itc_jump_td_lh = np.load(join(cfg.paradigm_dir,'itc_jump_td_lh.npy'))
# itc_jump_td_rh = np.load(join(cfg.paradigm_dir,'itc_jump_td_rh.npy'))
# itc_stay_td_lh = np.load(join(cfg.paradigm_dir,'itc_stay_td_lh.npy'))
# itc_stay_td_rh = np.load(join(cfg.paradigm_dir,'itc_stay_td_rh.npy'))
# itc_jump_asd_lh = np.load(join(cfg.paradigm_dir,'itc_jump_asd_lh.npy'))
# itc_jump_asd_rh = np.load(join(cfg.paradigm_dir,'itc_jump_asd_rh.npy'))
# itc_stay_asd_lh = np.load(join(cfg.paradigm_dir,'itc_stay_asd_lh.npy'))
# itc_stay_asd_rh = np.load(join(cfg.paradigm_dir,'itc_stay_asd_rh.npy'))

# load data
pow_jump_td_lh = np.load(join(cfg.paradigm_dir,'tf_jump_td_lh_norm.npy'))
pow_jump_td_rh = np.load(join(cfg.paradigm_dir,'tf_jump_td_rh_norm.npy'))
pow_stay_td_lh = np.load(join(cfg.paradigm_dir,'tf_stay_td_lh_norm.npy'))
pow_stay_td_rh = np.load(join(cfg.paradigm_dir,'tf_stay_td_rh_norm.npy'))
pow_jump_asd_lh = np.load(join(cfg.paradigm_dir,'tf_jump_asd_lh_norm.npy'))
pow_jump_asd_rh = np.load(join(cfg.paradigm_dir,'tf_jump_asd_rh_norm.npy'))
pow_stay_asd_lh = np.load(join(cfg.paradigm_dir,'tf_stay_asd_lh_norm.npy'))
pow_stay_asd_rh = np.load(join(cfg.paradigm_dir,'tf_stay_asd_rh_norm.npy'))
itc_jump_td_lh = np.load(join(cfg.paradigm_dir,'itc_jump_td_lh_norm.npy'))
itc_jump_td_rh = np.load(join(cfg.paradigm_dir,'itc_jump_td_rh_norm.npy'))
itc_stay_td_lh = np.load(join(cfg.paradigm_dir,'itc_stay_td_lh_norm.npy'))
itc_stay_td_rh = np.load(join(cfg.paradigm_dir,'itc_stay_td_rh_norm.npy'))
itc_jump_asd_lh = np.load(join(cfg.paradigm_dir,'itc_jump_asd_lh_norm.npy'))
itc_jump_asd_rh = np.load(join(cfg.paradigm_dir,'itc_jump_asd_rh_norm.npy'))
itc_stay_asd_lh = np.load(join(cfg.paradigm_dir,'itc_stay_asd_lh_norm.npy'))
itc_stay_asd_rh = np.load(join(cfg.paradigm_dir,'itc_stay_asd_rh_norm.npy'))

# normalize power and ITCs
# minpow_td = np.min(np.concatenate((pow_jump_td_lh,pow_jump_td_rh,pow_stay_td_lh,pow_stay_td_rh), axis=0))
# maxpow_td = np.max(np.concatenate((pow_jump_td_lh,pow_jump_td_rh,pow_stay_td_lh,pow_stay_td_rh), axis=0))

# minpow_asd = np.min(np.concatenate((pow_jump_asd_lh,pow_jump_asd_rh,pow_stay_asd_lh,pow_stay_asd_rh), axis=0))
# maxpow_asd = np.max(np.concatenate((pow_jump_asd_lh,pow_jump_asd_rh,pow_stay_asd_lh,pow_stay_asd_rh), axis=0))


# pow_jump_td_lh = (pow_jump_td_lh - minpow_td) / (maxpow_td - minpow_td)
# pow_jump_td_rh = (pow_jump_td_rh - minpow_td) / (maxpow_td - minpow_td)
# pow_stay_td_lh = (pow_stay_td_lh - minpow_td) / (maxpow_td - minpow_td)
# pow_stay_td_rh = (pow_stay_td_rh - minpow_td) / (maxpow_td - minpow_td)
# pow_jump_asd_lh = (pow_jump_asd_lh- minpow_asd) / (maxpow_asd - minpow_asd)
# pow_jump_asd_rh = np.load(join(cfg.paradigm_dir,'tf_jump_asd_rh.npy'))
# pow_stay_asd_lh = np.load(join(cfg.paradigm_dir,'tf_stay_asd_lh.npy'))
# pow_stay_asd_rh = np.load(join(cfg.paradigm_dir,'tf_stay_asd_rh.npy'))
# itc_jump_td_lh = np.load(join(cfg.paradigm_dir,'itc_jump_td_lh.npy'))
# itc_jump_td_rh = np.load(join(cfg.paradigm_dir,'itc_jump_td_rh.npy'))
# itc_stay_td_lh = np.load(join(cfg.paradigm_dir,'itc_stay_td_lh.npy'))
# itc_stay_td_rh = np.load(join(cfg.paradigm_dir,'itc_stay_td_rh.npy'))
# itc_jump_asd_lh = np.load(join(cfg.paradigm_dir,'itc_jump_asd_lh.npy'))
# itc_jump_asd_rh = np.load(join(cfg.paradigm_dir,'itc_jump_asd_rh.npy'))
# itc_stay_asd_lh = np.load(join(cfg.paradigm_dir,'itc_stay_asd_lh.npy'))
# itc_stay_asd_rh = np.load(join(cfg.paradigm_dir,'itc_stay_asd_rh.npy'))

# Now, plot grand averages
fig_dir = join(cfg.paradigm_dir,'figures')
pow_lims = (0,.7)

fig_td, axs = plt.subplots(2,2, figsize=(11,7))
tf1 = axs[0,0].imshow(np.mean(np.array(pow_jump_td_lh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,0].set_title('jump - lh')
axs[0,0].set_ylabel('freqs')
axs[0,0].set_xticks([])
axs[0,0].axvline(x=0, ls='--', color='gray')
axs[0,0].axvline(x=.55, ls='--', color='gray')
axs[0,0].set_xlim(-.2,1.2)
tf2 = axs[0,1].imshow(np.mean(np.array(pow_jump_td_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,1].set_title('jump - rh')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].axvline(x=0, ls='--', color='gray')
axs[0,1].axvline(x=.55, ls='--', color='gray')
axs[0,1].set_xlim(-.2,1.2)
tf3 = axs[1,0].imshow(np.mean(np.array(pow_stay_td_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,0].set_title('stay - lh')
axs[1,0].set_ylabel('freqs')
axs[1,0].set_xlabel('Time (s)')
axs[1,0].axvline(x=0, ls='--', color='gray')
axs[1,0].axvline(x=.55, ls='--', color='gray')        
axs[1,0].set_xlim(-.2,1.2)
tf4 = axs[1,1].imshow(np.mean(np.array(pow_stay_td_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,1].set_title('stay - rh')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_yticks([])
axs[1,1].axvline(x=0, ls='--', color='gray')
axs[1,1].axvline(x=.55, ls='--', color='gray')
axs[1,1].set_xlim(-.2,1.2)        
fig_td.colorbar(tf4, ax=axs, shrink=0.6, location='right', label='power (perc. change)')
fig_td.suptitle('TD')

# save GrandAve for TDs
fig_td.savefig(join(fig_dir,'ASSRnew_Jumps_GrandAverage_SourceTimeFreq_TD.tiff'), dpi=300)

fig_asd, axs = plt.subplots(2,2, figsize=(11,7))
tf1 = axs[0,0].imshow(np.mean(np.array(pow_jump_asd_lh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,0].set_title('jump - lh')
axs[0,0].set_ylabel('freqs')
axs[0,0].set_xticks([])
axs[0,0].axvline(x=0, ls='--', color='gray')
axs[0,0].axvline(x=.55, ls='--', color='gray')
axs[0,0].set_xlim(-.2,1.2)
tf2 = axs[0,1].imshow(np.mean(np.array(pow_jump_asd_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,1].set_title('jump - rh')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].axvline(x=0, ls='--', color='gray')
axs[0,1].axvline(x=.55, ls='--', color='gray')
axs[0,1].set_xlim(-.2,1.2)
tf3 = axs[1,0].imshow(np.mean(np.array(pow_stay_asd_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,0].set_title('stay - lh')
axs[1,0].set_ylabel('freqs')
axs[1,0].set_xlabel('Time (s)')
axs[1,0].axvline(x=0, ls='--', color='gray')
axs[1,0].axvline(x=.55, ls='--', color='gray')        
axs[1,0].set_xlim(-.2,1.2)
tf4 = axs[1,1].imshow(np.mean(np.array(pow_stay_asd_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,1].set_title('stay - rh')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_yticks([])
axs[1,1].axvline(x=0, ls='--', color='gray')
axs[1,1].axvline(x=.55, ls='--', color='gray')
axs[1,1].set_xlim(-.2,1.2)        
fig_asd.colorbar(tf4, ax=axs, shrink=0.6, location='right', label='power (perc. change)')
fig_asd.suptitle('ASD')

# save GrandAve for ASDs
fig_asd.savefig(join(fig_dir,'ASSRnew_Jumps_GrandAverage_SourceTimeFreq_ASD.tiff'), dpi=300)
plt.close('all')

# now plot ITCS
fig_td, axs = plt.subplots(2,2, figsize=(11,7))
pow_lims = (0,.5)
tf1 = axs[0,0].imshow(np.mean(np.array(itc_jump_td_lh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,0].set_title('jump - lh')
axs[0,0].set_ylabel('freqs')
axs[0,0].set_xticks([])
axs[0,0].axvline(x=0, ls='--', color='gray')
axs[0,0].axvline(x=.55, ls='--', color='gray')
axs[0,0].set_xlim(-.2,1.2)
tf2 = axs[0,1].imshow(np.mean(np.array(itc_jump_td_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,1].set_title('jump - rh')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].axvline(x=0, ls='--', color='gray')
axs[0,1].axvline(x=.55, ls='--', color='gray')
axs[0,1].set_xlim(-.2,1.2)
tf3 = axs[1,0].imshow(np.mean(np.array(itc_stay_td_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,0].set_title('stay - lh')
axs[1,0].set_ylabel('freqs')
axs[1,0].set_xlabel('Time (s)')
axs[1,0].axvline(x=0, ls='--', color='gray')
axs[1,0].axvline(x=.55, ls='--', color='gray')        
axs[1,0].set_xlim(-.2,1.2)
tf4 = axs[1,1].imshow(np.mean(np.array(itc_stay_td_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,1].set_title('stay - rh')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_yticks([])
axs[1,1].axvline(x=0, ls='--', color='gray')
axs[1,1].axvline(x=.55, ls='--', color='gray')
axs[1,1].set_xlim(-.2,1.2)        
fig_td.colorbar(tf4, ax=axs, shrink=0.6, location='right', label="ITC")
fig_td.suptitle('TD')

# save GrandAve for TDs
fig_td.savefig(join(fig_dir,'ASSRnew_Jumps_GrandAverage_SourceITC_TD.tiff'), dpi=300)

fig_asd, axs = plt.subplots(2,2, figsize=(11,7))
tf1 = axs[0,0].imshow(np.mean(np.array(itc_jump_asd_lh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,0].set_title('jump - lh')
axs[0,0].set_ylabel('freqs')
axs[0,0].set_xticks([])
axs[0,0].axvline(x=0, ls='--', color='gray')
axs[0,0].axvline(x=.55, ls='--', color='gray')
axs[0,0].set_xlim(-.2,1.2)
tf2 = axs[0,1].imshow(np.mean(np.array(itc_jump_asd_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[0,1].set_title('jump - rh')
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])
axs[0,1].axvline(x=0, ls='--', color='gray')
axs[0,1].axvline(x=.55, ls='--', color='gray')
axs[0,1].set_xlim(-.2,1.2)
tf3 = axs[1,0].imshow(np.mean(np.array(itc_stay_asd_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,0].set_title('stay - lh')
axs[1,0].set_ylabel('freqs')
axs[1,0].set_xlabel('Time (s)')
axs[1,0].axvline(x=0, ls='--', color='gray')
axs[1,0].axvline(x=.55, ls='--', color='gray')        
axs[1,0].set_xlim(-.2,1.2)
tf4 = axs[1,1].imshow(np.mean(np.array(itc_stay_asd_rh), axis=0), extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', origin='lower', vmin=pow_lims[0], vmax=pow_lims[1], cmap="RdBu_r")
axs[1,1].set_title('stay - rh')
axs[1,1].set_xlabel('Time (s)')
axs[1,1].set_yticks([])
axs[1,1].axvline(x=0, ls='--', color='gray')
axs[1,1].axvline(x=.55, ls='--', color='gray')
axs[1,1].set_xlim(-.2,1.2)        
fig_asd.colorbar(tf4, ax=axs, shrink=0.6, location='right', label='ITC')
fig_asd.suptitle('ASD')

# save GrandAve for ASDs
fig_asd.savefig(join(fig_dir,'ASSRnew_Jumps_GrandAverage_SourceITC_ASD.tiff'), dpi=300)

# save report
report.save(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name), verbose=False, overwrite=True)
report.save(join(cfg.paradigm_dir,source_method+'_'+cfg.report_name.replace('.hdf5','.html')),verbose=False,overwrite=True)