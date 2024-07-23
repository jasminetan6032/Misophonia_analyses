import mne
import mne_connectivity
import ASSR_config as cfg
import numpy as np
import pandas as pd
import scipy.stats as st
from glob import glob
import matplotlib.pyplot as plt
import random
from os.path import join, split, exists

# load alignment file
ASSR_df       = pd.read_csv(join(cfg.paradigm_dir,'alignment_file.csv'), index_col=0, dtype=str)
subject_info  = pd.read_csv(join(cfg.paradigm_dir,'ASSR_subject_info.csv'), dtype=str)
source_method = 'dSPM'

# new sample frequency 
sfreq = 250

redo = True
normalize_gc = True

#time to analize
tmin    = .6
tmax    = 1.2
fmin    = 8
fmax    = 30
padding = .29
cwt_freqs = np.arange(fmin, fmax, 1)
cwt_n_cycles = 3

# get subject list
subjects = list(ASSR_df['subject'])

src_to = mne.read_source_spaces('/local_mount/space/tapputi/1/users/sergio/MNE-sample-data/subjects/fsaverage/bem/fsaverage-ico-5-src.fif')

# get labels for GC analyses
labels = glob(join(cfg.paradigm_dir,'connectivity_analyses','*cluster*.label'))
labels.sort()
# get rid of non significant label
labels = [i for i in labels if 'lh_seed_cluster_2_rh' not in i]

# containers for feedback connectivity
from_PFCA2AC_td_jump,from_PFCA2AC_asd_jump = [],[]
from_MT2AC_td_jump,from_MT2AC_asd_jump = [],[]
from_PFCB2AC_td_jump,from_PFCB2AC_asd_jump = [],[]
from_PFCA2AC_td_stay,from_PFCA2AC_asd_stay = [],[]
from_MT2AC_td_stay,from_MT2AC_asd_stay = [],[]
from_PFCB2AC_td_stay,from_PFCB2AC_asd_stay = [],[]

# containers for feedforward coonecitivy
from_AC2PFCA_td_jump,from_AC2PFCA_asd_jump = [],[]
from_AC2MT_td_jump,from_AC2MT_asd_jump = [],[]
from_AC2PFCB_td_jump,from_AC2PFCB_asd_jump = [],[]
from_AC2PFCA_td_stay,from_AC2PFCA_asd_stay = [],[]
from_AC2MT_td_stay,from_AC2MT_asd_stay = [],[]
from_AC2PFCB_td_stay,from_AC2PFCB_asd_stay = [],[]

jump_td_epoch_count  = [] 
stay_td_epoch_count  = []
jump_asd_epoch_count = [] 
stay_asd_epoch_count = []

if not exists(join(cfg.paradigm_dir,'connectivity_analyses','granger_jump_feedback_6-30hz.npz')) or redo:
    for counter,sub_i in enumerate(subjects):
            # print counter
            print('\n\n >>>>> Estimating FC for subject %d / %d <<<<< \n\n' %(counter+1,len(subjects)))

            # set subject paths
            sss_path    = list(ASSR_df['sss path'][ASSR_df['subject'] == sub_i])[0]
            info        = mne.io.read_info(sss_path)
            this_visit  = [i for i in sss_path.split('/') if 'visit' in i][0]
            recons_path = list(ASSR_df['recons path'][ASSR_df['subject'] == sub_i])[0]
            output_dir  = join(cfg.paradigm_dir,sub_i,this_visit)
            subject     = split(recons_path)[-1]
            subjects_dir = cfg.recons_dir

            # some subjects don't have good MRIs
            if sub_i in ['082802','082601','082501']:
                subject      = 'fsaverage'
                subjects_dir = '/local_mount/space/tapputi/1/users/sergio/MNE-sample-data/subjects'

            # get diagnosis
            diagnosis   = list(subject_info[subject_info['Subj_ID'] == sub_i]['diagnosis'])[0]

            # load src
            src_fname = glob(join(output_dir,'*src.fif'))[0]
            src       = mne.read_source_spaces(src_fname)
            
            # load stc data and pre-defined labels
            stc_fname  = split(sss_path)[1].split('0hp')[0]+'01_120hz.stc'

            # read epochs
            epochs_fname = stc_fname.replace('.stc','_epo.fif')   
            all_epochs   = mne.read_epochs(join(output_dir,epochs_fname),  preload=True)
            all_epochs   = all_epochs.resample(sfreq)

            # downsampled data is 1 sample longer. Trim those files
            # if len(all_epochs.times) == 2001: all_epochs = all_epochs.crop(tmax=all_epochs.times[-2]) 
            jump_epochs = all_epochs['jump'] 
            stay_epochs = all_epochs['stay']

            # get the inverse operator
            cov_path  = glob(join(cfg.transcend_dir,sub_i,this_visit,'epoched','*cov*'))[0]
            noise_cov = mne.read_cov(cov_path)

            # compute the inverse operator
            fwd_fname     = stc_fname.split('_01')[0]+'_fwd.fif'
            fwd           = mne.read_forward_solution(join(output_dir,fwd_fname), verbose=False)
            inv_operator  = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8, rank='info')

            # get source time courses
            snr     = 3
            lambda2 = 1.0 / snr**2

            # apply inverse to epochs
            stcs_jump = mne.minimum_norm.apply_inverse_epochs(jump_epochs, inv_operator, lambda2, source_method, pick_ori="normal", verbose=False)
            stcs_stay = mne.minimum_norm.apply_inverse_epochs(stay_epochs, inv_operator, lambda2, source_method, pick_ori="normal", verbose=False)
            stcs_jump = [i.crop(tmin=tmin, tmax=tmax) for i in stcs_jump]
            stcs_stay = [i.crop(tmin=tmin, tmax=tmax) for i in stcs_stay]

            # load label
            try:
                func_label_rh = mne.read_label(join(output_dir,'corrected_labels',source_method+'_rh.label'))
            except:
                func_label_rh = mne.read_label(join(output_dir,source_method+'_rh.label'))
                
            try:
                func_label_lh = mne.read_label(join(output_dir,'corrected_labels',source_method+'_lh.label'))
            except:    
                func_label_lh = mne.read_label(join(output_dir,source_method+'_lh.label')) 
                
            # now we need to morph data to fsaverage
            fwd  = mne.read_forward_solution(glob(join(output_dir,'*fwd.fif'))[0])
            src_from = fwd['src']

            # extract signals from labels
            stc_jump_lh = mne.extract_label_time_course(stcs_jump, func_label_lh, src, mode='mean_flip', verbose=False)
            stc_jump_rh = mne.extract_label_time_course(stcs_jump, func_label_rh, src, mode='mean_flip', verbose=False)
            stc_stay_lh = mne.extract_label_time_course(stcs_stay, func_label_lh, src, mode='mean_flip', verbose=False)
            stc_stay_rh = mne.extract_label_time_course(stcs_stay, func_label_rh, src, mode='mean_flip', verbose=False)            

            morphed_surf  = mne.compute_source_morph(src_from, subject_from=subject, subject_to='fsaverage',src_to=src_to, subjects_dir=subjects_dir) 
            stcs_jump   = [morphed_surf.apply(i) for i in stcs_jump]
            stcs_stay   = [morphed_surf.apply(i) for i in stcs_stay]

            # lets get the data from the labels we are interested in
            data4gc_jump = []
            data4gc_stay = []
            for label_i in labels:
                label = mne.read_label(label_i)
                data4gc_jump.append(np.array([np.mean(i.in_label(label).data, axis=0) for i in stcs_jump]))
                data4gc_stay.append(np.array([np.mean(i.in_label(label).data, axis=0) for i in stcs_stay]))

            n_labels,n_trials,n_time = np.array(data4gc_jump).shape
            data4gc_jump = np.concatenate((np.array(data4gc_jump), np.array(stc_jump_lh).reshape(1,n_trials,n_time), np.array(stc_jump_rh).reshape(1,n_trials,n_time)), axis=0)
            n_labels,n_trials,n_time = np.array(data4gc_stay).shape
            data4gc_stay = np.concatenate((np.array(data4gc_stay), np.array(stc_stay_lh).reshape(1,n_trials,n_time), np.array(stc_stay_rh).reshape(1,n_trials,n_time)), axis=0)

            # now let's reshape arrays to match mne python GC connectivity requirements
            n_labels,n_trials,n_time = np.array(data4gc_jump).shape
            data4gc_jump = data4gc_jump.reshape(n_trials,n_labels,n_time)
            n_labels,n_trials,n_time = np.array(data4gc_stay).shape
            data4gc_stay = data4gc_stay.reshape(n_trials,n_labels,n_time)

            # data4gc_jump and data4gc_stay are 3D arrays (trials,rois,time)
            # [:,0,:] -> PFC1
            # [:,1,:] -> MT
            # [:,2,:] -> PFC2

            # ---- first, let's compute feedback connectivity ---- #
            indices = ([[0], [0]],  # seeds
                    [[3], [4]])     # targets
            
            gc_pfcA2ac_jump = mne_connectivity.spectral_connectivity_time(
                data4gc_jump,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                mode='cwt_morlet',
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                n_cycles=cwt_n_cycles,
                fmin=fmin,
                fmax=fmax,
                verbose=False)  

            gc_pfcA2ac_stay = mne_connectivity.spectral_connectivity_time(
                data4gc_stay,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False)  
                    
            indices = ([[1], [1]],  # seeds
                [[3], [4]])  # targets
            
            gc_mt2ac_jump = mne_connectivity.spectral_connectivity_time(
                data4gc_jump,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 

            gc_mt2ac_stay = mne_connectivity.spectral_connectivity_time(
                data4gc_stay,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 
            
            indices = ([[2], [2]],  # seeds
                    [[3], [4]])  # targets
            
            gc_pfcB2ac_jump = mne_connectivity.spectral_connectivity_time(
                data4gc_jump,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 

            gc_pfcB2ac_stay = mne_connectivity.spectral_connectivity_time(
                data4gc_stay,
                method="gc",
                indices=indices,
                n_cycles=cwt_n_cycles,
                sfreq=sfreq,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 
                            
            freqs = gc_pfcB2ac_jump.freqs

            if 'TD' in diagnosis:
                from_PFCA2AC_td_jump.append(np.mean(gc_pfcA2ac_jump.get_data(), axis=0))
                from_MT2AC_td_jump.append(np.mean(gc_mt2ac_jump.get_data(), axis=0))
                from_PFCB2AC_td_jump.append(np.mean(gc_pfcB2ac_jump.get_data(), axis=0))    
                from_PFCA2AC_td_stay.append(np.mean(gc_pfcA2ac_stay.get_data(), axis=0))
                from_MT2AC_td_stay.append(np.mean(gc_mt2ac_stay.get_data(), axis=0))
                from_PFCB2AC_td_stay.append(np.mean(gc_pfcB2ac_stay.get_data(), axis=0)) 
                jump_td_epoch_count.append(len(stcs_jump))
                stay_td_epoch_count.append(len(stcs_stay))   
            else:
                from_PFCA2AC_asd_jump.append(np.mean(gc_pfcA2ac_jump.get_data(), axis=0))
                from_MT2AC_asd_jump.append(np.mean(gc_mt2ac_jump.get_data(), axis=0))
                from_PFCB2AC_asd_jump.append(np.mean(gc_pfcB2ac_jump.get_data(), axis=0))    
                from_PFCA2AC_asd_stay.append(np.mean(gc_pfcA2ac_stay.get_data(), axis=0))
                from_MT2AC_asd_stay.append(np.mean(gc_mt2ac_stay.get_data(), axis=0))
                from_PFCB2AC_asd_stay.append(np.mean(gc_pfcB2ac_stay.get_data(), axis=0))
                jump_asd_epoch_count.append(len(stcs_jump))
                stay_asd_epoch_count.append(len(stcs_stay))    

            #  ---- now let's compute feedfordward connectivity ---- #
    
            indices = ([[3], [4]],  # seeds
                    [[0], [0]])  # targets
            
            gc_ac2pfcA_jump = mne_connectivity.spectral_connectivity_time(
                data4gc_jump,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                n_cycles=cwt_n_cycles,
                fmin=fmin,
                fmax=fmax,
                verbose=False)  

            gc_ac2pfcA_stay = mne_connectivity.spectral_connectivity_time(
                data4gc_stay,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,                
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False)  
                    
            indices = ([[3], [4]],  # seeds
                [[1], [1]])  # targets
            
            gc_ac2mt_jump = mne_connectivity.spectral_connectivity_time(
                data4gc_jump,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 

            gc_ac2mt_stay = mne_connectivity.spectral_connectivity_time(
                data4gc_stay,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 
            
            indices = ([[3], [4]],  # seeds
                    [[2], [2]])  # targets
            
            gc_ac2pfcB_jump = mne_connectivity.spectral_connectivity_time(
                data4gc_jump,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 

            gc_ac2pfcB_stay = mne_connectivity.spectral_connectivity_time(
                data4gc_stay,
                method="gc",
                indices=indices,
                sfreq=sfreq,
                n_cycles=cwt_n_cycles,
                gc_n_lags=20,
                padding=padding,
                freqs=cwt_freqs,
                fmin=fmin,
                fmax=fmax,
                verbose=False) 

            if 'TD' in diagnosis:
                from_AC2PFCA_td_jump.append(np.mean(gc_ac2pfcA_jump.get_data(), axis=0))
                from_AC2MT_td_jump.append(np.mean(gc_ac2mt_jump.get_data(), axis=0))
                from_AC2PFCB_td_jump.append(np.mean(gc_ac2pfcB_jump.get_data(), axis=0))    
                from_AC2PFCA_td_stay.append(np.mean(gc_ac2pfcA_stay.get_data(), axis=0))
                from_AC2MT_td_stay.append(np.mean(gc_ac2mt_stay.get_data(), axis=0))
                from_AC2PFCB_td_stay.append(np.mean(gc_ac2pfcB_stay.get_data(), axis=0))    
            else:
                from_AC2PFCA_asd_jump.append(np.mean(gc_ac2pfcA_jump.get_data(), axis=0))
                from_AC2MT_asd_jump.append(np.mean(gc_ac2mt_jump.get_data(), axis=0))
                from_AC2PFCB_asd_jump.append(np.mean(gc_ac2pfcB_jump.get_data(), axis=0))    
                from_AC2PFCA_asd_stay.append(np.mean(gc_ac2pfcA_stay.get_data(), axis=0))
                from_AC2MT_asd_stay.append(np.mean(gc_ac2mt_stay.get_data(), axis=0))
                from_AC2PFCB_asd_stay.append(np.mean(gc_ac2pfcB_stay.get_data(), axis=0)) 

    np.savez(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_jump_feedback_{fmin}-{fmax}hz_{tmin}-{tmax}s"), 
             from_SG2AC_td_jump   = from_PFCA2AC_td_jump, 
             from_SP2AC_td_jump   = from_MT2AC_td_jump, 
             from_PFC2AC_td_jump  = from_PFCB2AC_td_jump, 
             from_SG2AC_asd_jump  = from_PFCA2AC_asd_jump, 
             from_SP2AC_asd_jump  = from_MT2AC_asd_jump, 
             from_PFC2AC_asd_jump = from_PFCB2AC_asd_jump, 
             jump_asd_epoch_count = jump_asd_epoch_count,
             stay_asd_epoch_count = stay_asd_epoch_count,
             jump_td_epoch_count  = jump_td_epoch_count,
             stay_td_epoch_count  = stay_td_epoch_count,
             freqs=freqs)
    np.savez(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_stay_feedback_{fmin}-{fmax}hz_{tmin}-{tmax}s"), 
             from_SG2AC_td_stay   = from_PFCA2AC_td_stay, 
             from_SP2AC_td_stay   = from_MT2AC_td_stay, 
             from_PFC2AC_td_stay  = from_PFCB2AC_td_stay, 
             from_SG2AC_asd_stay  = from_PFCA2AC_asd_stay, 
             from_SP2AC_asd_stay  = from_MT2AC_asd_stay, 
             from_PFC2AC_asd_stay = from_PFCB2AC_asd_stay, 
             jump_asd_epoch_count = jump_asd_epoch_count,
             stay_asd_epoch_count = stay_asd_epoch_count,
             jump_td_epoch_count  = jump_td_epoch_count,
             stay_td_epoch_count  = stay_td_epoch_count,             
             freqs=freqs)
    np.savez(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_jump_feedforward_{fmin}-{fmax}hz_{tmin}-{tmax}s"), 
             from_AC2SG_td_jump   = from_AC2PFCA_td_jump, 
             from_AC2SP_td_jump   = from_AC2MT_td_jump, 
             from_AC2PFC_td_jump  = from_AC2PFCB_td_jump, 
             from_AC2SG_asd_jump  = from_AC2PFCA_asd_jump, 
             from_AC2SP_asd_jump  = from_AC2MT_asd_jump, 
             from_AC2PFC_asd_jump = from_AC2PFCB_asd_jump,
             jump_asd_epoch_count = jump_asd_epoch_count,
             stay_asd_epoch_count = stay_asd_epoch_count,
             jump_td_epoch_count  = jump_td_epoch_count,
             stay_td_epoch_count  = stay_td_epoch_count,             
             freqs=freqs)
    np.savez(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_stay_feedforward_{fmin}-{fmax}hz_{tmin}-{tmax}s"), 
             from_AC2SG_td_stay   = from_AC2PFCA_td_stay, 
             from_AC2SP_td_stay   = from_AC2MT_td_stay, 
             from_AC2PFC_td_stay  = from_AC2PFCB_td_stay, 
             from_AC2SG_asd_stay  = from_AC2PFCA_asd_stay, 
             from_AC2SP_asd_stay  = from_AC2MT_asd_stay, 
             from_AC2PFC_asd_stay = from_AC2PFCB_asd_stay, 
             jump_asd_epoch_count = jump_asd_epoch_count,
             stay_asd_epoch_count = stay_asd_epoch_count,
             jump_td_epoch_count  = jump_td_epoch_count,
             stay_td_epoch_count  = stay_td_epoch_count,           
             freqs=freqs)
else:

    GC_jump_fb = np.load(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_jump_feedback_{fmin}-{fmax}hz_{tmin}-{tmax}s.npz"))
    from_PFCA2AC_td_jump  = GC_jump_fb['from_PFCA2AC_td_jump']
    from_MT2AC_td_jump    = GC_jump_fb['from_MT2AC_td_jump']
    from_PFCB2AC_td_jump  = GC_jump_fb['from_PFCB2AC_td_jump']
    from_PFCA2AC_asd_jump = GC_jump_fb['from_PFCA2AC_asd_jump']
    from_MT2AC_asd_jump   = GC_jump_fb['from_MT2AC_asd_jump']
    from_PFCB2AC_asd_jump = GC_jump_fb['from_PFCB2AC_asd_jump']
    jump_asd_epoch_count = np.expand_dims(np.expand_dims(GC_jump_fb['jump_asd_epoch_count'], axis=1), axis=2)
    stay_asd_epoch_count = np.expand_dims(np.expand_dims(GC_jump_fb['stay_asd_epoch_count'], axis=1), axis=2)
    jump_td_epoch_count  = np.expand_dims(np.expand_dims(GC_jump_fb['jump_td_epoch_count'], axis=1), axis=2)
    stay_td_epoch_count  = np.expand_dims(np.expand_dims(GC_jump_fb['stay_td_epoch_count'], axis=1), axis=2)

    GC_stay_fb = np.load(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_stay_feedback_{fmin}-{fmax}hz_{tmin}-{tmax}s.npz"))
    from_PFCA2AC_td_stay  = GC_stay_fb['from_PFCA2AC_td_stay']
    from_MT2AC_td_stay    = GC_stay_fb['from_MT2AC_td_stay']
    from_PFCB2AC_td_stay  = GC_stay_fb['from_PFCB2AC_td_stay']
    from_PFCA2AC_asd_stay = GC_stay_fb['from_PFCA2AC_asd_stay']
    from_MT2AC_asd_stay   = GC_stay_fb['from_MT2AC_asd_stay']
    from_PFCB2AC_asd_stay = GC_stay_fb['from_PFCB2AC_asd_stay']  

    GC_jump_ff = np.load(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_jump_feedforward_{fmin}-{fmax}hz_{tmin}-{tmax}s.npz"))
    from_AC2PFCA_td_jump  = GC_jump_ff['from_AC2PFCA_td_jump']
    from_AC2MT_td_jump    = GC_jump_ff['from_AC2MT_td_jump']
    from_AC2PFCB_td_jump  = GC_jump_ff['from_AC2PFCB_td_jump']
    from_AC2PFCA_asd_jump = GC_jump_ff['from_AC2PFCA_asd_jump']
    from_AC2MT_asd_jump   = GC_jump_ff['from_AC2MT_asd_jump']
    from_AC2PFCB_asd_jump = GC_jump_ff['from_AC2PFCB_asd_jump']

    GC_stay_ff = np.load(join(cfg.paradigm_dir,'connectivity_analyses',f"granger_stay_feedforward_{fmin}-{fmax}hz_{tmin}-{tmax}s.npz"))
    from_AC2PFCA_td_stay  = GC_stay_ff['from_AC2PFCA_td_stay']
    from_AC2MT_td_stay    = GC_stay_ff['from_AC2MT_td_stay']
    from_AC2PFCB_td_stay  = GC_stay_ff['from_AC2PFCB_td_stay']
    from_AC2PFCA_asd_stay = GC_stay_ff['from_AC2PFCA_asd_stay']
    from_AC2MT_asd_stay   = GC_stay_ff['from_AC2MT_asd_stay']
    from_AC2PFCB_asd_stay = GC_stay_ff['from_AC2PFCB_asd_stay']    
    freqs = GC_stay_ff['freqs']

# # normalize GC estimates 
# if normalize_gc:
#     from_PFCA2AC_td_jump  = ((np.emath.arctanh(from_PFCA2AC_td_jump)  - (1 / jump_td_epoch_count))  - (np.emath.arctanh(from_PFCA2AC_td_stay)  - (1 / stay_td_epoch_count))) / np.sqrt((1/jump_td_epoch_count)+(1/stay_td_epoch_count))
#     from_MT2AC_td_jump  = ((np.emath.arctanh(from_MT2AC_td_jump)  - (1 / jump_td_epoch_count))  - (np.emath.arctanh(from_MT2AC_td_stay)  - (1 / stay_td_epoch_count))) / np.sqrt((1/jump_td_epoch_count)+(1/stay_td_epoch_count))
#     from_PFCB2AC_td_jump = ((np.emath.arctanh(from_PFCB2AC_td_jump) - (1 / jump_td_epoch_count))  - (np.emath.arctanh(from_PFCB2AC_td_stay)  - (1 / stay_td_epoch_count))) / np.sqrt((1/jump_td_epoch_count)+(1/stay_td_epoch_count))

#     from_PFCA2AC_asd_jump  = ((np.emath.arctanh(from_PFCA2AC_asd_jump)  - (1 / jump_asd_epoch_count))  - (np.emath.arctanh(from_PFCA2AC_asd_stay)  - (1 / stay_asd_epoch_count))) / np.sqrt((1/jump_asd_epoch_count)+(1/stay_asd_epoch_count))
#     from_MT2AC_asd_jump  = ((np.emath.arctanh(from_MT2AC_asd_jump)  - (1 / jump_asd_epoch_count))  - (np.emath.arctanh(from_MT2AC_asd_stay)  - (1 / stay_asd_epoch_count))) / np.sqrt((1/jump_asd_epoch_count)+(1/stay_asd_epoch_count))
#     from_PFCB2AC_asd_jump = ((np.emath.arctanh(from_PFCB2AC_asd_jump) - (1 / jump_asd_epoch_count))  - (np.emath.arctanh(from_PFCB2AC_asd_stay)  - (1 / stay_asd_epoch_count))) / np.sqrt((1/jump_asd_epoch_count)+(1/stay_asd_epoch_count))

#     from_AC2PFCA_td_jump  = ((np.emath.arctanh(from_AC2PFCA_td_jump)  - (1 / jump_td_epoch_count))  - (np.emath.arctanh(from_AC2MT_td_stay)  - (1 / stay_td_epoch_count))) / np.sqrt((1/jump_td_epoch_count)+(1/stay_td_epoch_count))
#     from_AC2MT_td_jump  = ((np.emath.arctanh(from_AC2MT_td_jump)  - (1 / jump_td_epoch_count))  - (np.emath.arctanh(from_AC2MT_td_stay)  - (1 / stay_td_epoch_count))) / np.sqrt((1/jump_td_epoch_count)+(1/stay_td_epoch_count))
#     from_AC2PFCB_td_jump = ((np.emath.arctanh(from_AC2PFCB_td_jump) - (1 / jump_td_epoch_count))  - (np.emath.arctanh(from_AC2PFCB_td_stay)  - (1 / stay_td_epoch_count))) / np.sqrt((1/jump_td_epoch_count)+(1/stay_td_epoch_count))

#     from_AC2PFCA_asd_jump  = ((np.emath.arctanh(from_AC2PFCA_asd_jump)  - (1 / jump_asd_epoch_count))  - (np.emath.arctanh(from_AC2PFCA_asd_stay)  - (1 / stay_asd_epoch_count))) / np.sqrt((1/jump_asd_epoch_count)+(1/stay_asd_epoch_count))
#     from_AC2MT_asd_jump  = ((np.emath.arctanh(from_AC2MT_asd_jump)  - (1 / jump_asd_epoch_count))  - (np.emath.arctanh(from_AC2MT_asd_stay)  - (1 / stay_asd_epoch_count))) / np.sqrt((1/jump_asd_epoch_count)+(1/stay_asd_epoch_count))
#     from_AC2PFCB_asd_jump = ((np.emath.arctanh(from_AC2PFCB_asd_jump) - (1 / jump_asd_epoch_count))  - (np.emath.arctanh(from_AC2PFCB_asd_stay)  - (1 / stay_asd_epoch_count))) / np.sqrt((1/jump_asd_epoch_count)+(1/stay_asd_epoch_count))

# SMALL_SIZE = 32
# plt.rcParams["font.family"] = "Arial"
# plt.rc('font', size=SMALL_SIZE)
# plt.rc('axes', titlesize=SMALL_SIZE)   

# n_td  = np.array(from_PFCA2AC_td_jump).shape[0]
# n_asd = np.array(from_PFCA2AC_asd_jump).shape[0]
# lims  = [-.35,.35] if normalize_gc else [.04,.12] 
# figsize = (13.33,  6.13)
# l_freq = 8
# h_freq = 12
# pvals  = []

# rand_factor_td = np.array([random.randint(-10,10) for _ in range(n_td)])/100
# rand_factor_asd = np.array([random.randint(-10,10) for _ in range(n_asd)])/100

# # From supramarginal gyrus
# data4scatter_asd = np.median(np.array(from_PFCA2AC_asd_jump)[:,0,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# data4scatter_td  = np.median(np.array(from_PFCA2AC_td_jump)[:,0,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# tstat, pval = st.ttest_ind(data4scatter_td,data4scatter_asd)
# print('\nSG->AC : t = %s, p = %s' % (tstat, pval))
# pvals.append(pval)

# fig, ax = plt.subplots(1,2, figsize=figsize, layout='tight')
# ax[0].plot((1,2), (np.median(data4scatter_td), np.median(data4scatter_asd)), color='k', alpha=.2, linewidth=2)
# ax[0].scatter(np.ones(n_td)+rand_factor_td, data4scatter_td, color='lightgreen', s=150, edgecolors='w')
# ax[0].scatter(np.ones(n_asd)*2+rand_factor_asd, data4scatter_asd, color='orchid', s=150, edgecolors='w')
# ax[0].set_xlim([0, 3])
# # ax[0].set_ylim(lims)
# ax[0].set_xticks([1, 2], labels=['ASD', 'TD'])
# ax[0].set_ylabel('Median α-band GC (a.u.)')
# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)
# ax[0].errorbar([1,2],
#             [np.median(data4scatter_td),np.median(data4scatter_asd)],
#             yerr=[np.std(data4scatter_td),np.std(data4scatter_asd)], ecolor='black', fmt='o', 
#             capsize=5, markerfacecolor='black', markeredgecolor='black')
# ax[0].spines['left'].set_linewidth(2)
# ax[0].spines['bottom'].set_linewidth(2)
# ax[0].set_title('SG->AC')

# data4scatter_asd = np.median(np.array(from_AC2PFCA_asd_jump)[:,1,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# data4scatter_td  = np.median(np.array(from_AC2PFCA_td_jump)[:,1,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# tstat, pval = st.ttest_ind(data4scatter_td,data4scatter_asd)
# print('\nAC-SG : t = %s, p = %s' % (tstat, pval))
# pvals.append(pval)

# ax[1].plot((1,2), (np.median(data4scatter_td), np.median(data4scatter_asd)), color='k', alpha=.2, linewidth=2)
# ax[1].scatter(np.ones(n_td)+rand_factor_td, data4scatter_td, color='lightgreen', s=150, edgecolors='w')
# ax[1].scatter(np.ones(n_asd)*2+rand_factor_asd, data4scatter_asd, color='orchid', s=150, edgecolors='w')
# ax[1].set_xlim([0, 3])
# # ax[1].set_ylim(lims)
# ax[1].set_xticks([1, 2], labels=['ASD', 'TD'])
# # ax[1].set_ylabel('Median α-band GC (a.u.)')
# ax[1].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].errorbar([1,2],
#             [np.median(data4scatter_td),np.median(data4scatter_asd)],
#             yerr=[np.std(data4scatter_td),np.std(data4scatter_asd)], ecolor='black', fmt='o', 
#             capsize=5, markerfacecolor='black', markeredgecolor='black')
# ax[1].spines['left'].set_linewidth(2)
# ax[1].spines['bottom'].set_linewidth(2)
# ax[1].set_title('AC->SG')
# fig.show()

# # From superior parietal lobe
# data4scatter_asd = np.median(np.array(from_MT2AC_asd_jump)[:,0,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# data4scatter_td  = np.median(np.array(from_MT2AC_td_jump)[:,0,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# tstat, pval = st.ttest_ind(data4scatter_td,data4scatter_asd)
# print('\nSPL->AC : t = %s, p = %s' % (tstat, pval))
# pvals.append(pval)

# fig, ax = plt.subplots(1,2, figsize=figsize, layout='tight')
# ax[0].plot((1,2), (np.median(data4scatter_td), np.median(data4scatter_asd)), color='k', alpha=.2, linewidth=2)
# ax[0].scatter(np.ones(n_td)+rand_factor_td, data4scatter_td, color='lightgreen', s=150, edgecolors='w')
# ax[0].scatter(np.ones(n_asd)*2+rand_factor_asd, data4scatter_asd, color='orchid', s=150, edgecolors='w')
# ax[0].set_xlim([0, 3])
# # ax[0].set_ylim(lims)
# ax[0].set_xticks([1, 2], labels=['ASD', 'TD'])
# ax[0].set_ylabel('Median α-band GC (a.u.)')
# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)
# ax[0].errorbar([1,2],
#             [np.median(data4scatter_td),np.median(data4scatter_asd)],
#             yerr=[np.std(data4scatter_td),np.std(data4scatter_asd)], ecolor='black', fmt='o', 
#             capsize=5, markerfacecolor='black', markeredgecolor='black')
# ax[0].spines['left'].set_linewidth(2)
# ax[0].spines['bottom'].set_linewidth(2)
# ax[0].set_title('SPL->AC')

# data4scatter_asd = np.median(np.array(from_AC2MT_asd_jump)[:,1,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# data4scatter_td  = np.median(np.array(from_AC2MT_td_jump)[:,1,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# tstat, pval = st.ttest_ind(data4scatter_td,data4scatter_asd)
# print('\nAC->SPL : t = %s, p = %s' % (tstat, pval))
# pvals.append(pval)

# ax[1].plot((1,2), (np.median(data4scatter_td), np.median(data4scatter_asd)), color='k', alpha=.2, linewidth=2)
# ax[1].scatter(np.ones(n_td)+rand_factor_td, data4scatter_td, color='lightgreen', s=150, edgecolors='w')
# ax[1].scatter(np.ones(n_asd)*2+rand_factor_asd, data4scatter_asd, color='orchid', s=150, edgecolors='w')
# ax[1].set_xlim([0, 3])
# # ax[1].set_ylim(lims)
# ax[1].set_xticks([1, 2], labels=['ASD', 'TD'])
# ax[1].set_ylabel('Median α-band GC (a.u.)')
# ax[1].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].errorbar([1,2],
#             [np.median(data4scatter_td),np.median(data4scatter_asd)],
#             yerr=[np.std(data4scatter_td),np.std(data4scatter_asd)], ecolor='black', fmt='o', 
#             capsize=5, markerfacecolor='black', markeredgecolor='black')
# ax[1].spines['left'].set_linewidth(2)
# ax[1].spines['bottom'].set_linewidth(2)
# ax[1].set_title('AC->SPL')
# fig.show()

# # From Prefrontal  cortex
# data4scatter_asd = np.median(np.array(from_PFCB2AC_asd_jump)[:,0,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# data4scatter_td  = np.median(np.array(from_PFCB2AC_td_jump)[:,0,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# tstat, pval = st.ttest_ind(data4scatter_td,data4scatter_asd)
# print('\nPFC->AC : t = %s, p = %s' % (tstat, pval))
# pvals.append(pval)

# fig, ax = plt.subplots(1,2, figsize=figsize, layout='tight')
# ax[0].plot((1,2), (np.median(data4scatter_td), np.median(data4scatter_asd)), color='k', alpha=.2, linewidth=2)
# ax[0].scatter(np.ones(n_td)+rand_factor_td, data4scatter_td, color='lightgreen', s=150, edgecolors='w')
# ax[0].scatter(np.ones(n_asd)*2+rand_factor_asd, data4scatter_asd, color='orchid', s=150, edgecolors='w')
# ax[0].set_xlim([0, 3])
# # ax[0].set_ylim(lims)
# ax[0].set_xticks([1, 2], labels=['ASD', 'TD'])
# ax[0].set_ylabel('Median α-band GC (a.u.)')
# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)
# ax[0].errorbar([1,2],
#             [np.median(data4scatter_td),np.median(data4scatter_asd)],
#             yerr=[np.std(data4scatter_td),np.std(data4scatter_asd)], ecolor='black', fmt='o', 
#             capsize=5, markerfacecolor='black', markeredgecolor='black')
# ax[0].spines['left'].set_linewidth(2)
# ax[0].spines['bottom'].set_linewidth(2)
# ax[0].set_title('PFC->AC')

# data4scatter_asd = np.median(np.array(from_AC2PFCB_asd_jump)[:,1,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# data4scatter_td  = np.median(np.array(from_AC2PFCB_td_jump)[:,1,(np.array(freqs) >= l_freq) & (np.array(freqs) <= h_freq)], axis=1)
# tstat, pval = st.ttest_ind(data4scatter_td,data4scatter_asd)
# print('\nAC->PFC : t = %s, p = %s' % (tstat, pval))
# pvals.append(pval)

# ax[1].plot((1,2), (np.median(data4scatter_td), np.median(data4scatter_asd)), color='k', alpha=.2, linewidth=2)
# ax[1].scatter(np.ones(n_td)+rand_factor_td, data4scatter_td, color='lightgreen', s=150, edgecolors='w')
# ax[1].scatter(np.ones(n_asd)*2+rand_factor_asd, data4scatter_asd, color='orchid', s=150, edgecolors='w')
# ax[1].set_xlim([0, 3])
# # ax[1].set_ylim(lims)
# ax[1].set_xticks([1, 2], labels=['ASD', 'TD'])
# ax[1].set_ylabel('Median α-band GC (a.u.)')
# ax[1].spines['top'].set_visible(False)
# ax[1].spines['right'].set_visible(False)
# ax[1].errorbar([1,2],
#             [np.median(data4scatter_td),np.median(data4scatter_asd)],
#             yerr=[np.std(data4scatter_td),np.std(data4scatter_asd)], ecolor='black', fmt='o', 
#             capsize=5, markerfacecolor='black', markeredgecolor='black')
# ax[1].spines['left'].set_linewidth(2)
# ax[1].spines['bottom'].set_linewidth(2)
# ax[1].set_title('AC->PFC')
# fig.show()

# mne.stats.fdr_correction(pvals)[1]
# print('')

