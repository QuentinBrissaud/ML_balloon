#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import obspy
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import h5py
import seaborn as sns 

import fastmap
import sklearn
from sklearn.pipeline import Pipeline

from math import log, ceil, floor

def closest_power(x, power=8):
    possible_results = floor(log(x, power)), ceil(log(x, power))
    return min(possible_results, key= lambda z: abs(x-power**z))

def load_data(root_dir, h5_file, normalize=True, constrain_size=True, normalize_type = 'std'):
    filepath = f'{root_dir}{h5_file}'
    event_types = ''
    with h5py.File(filepath, "r") as f:
        X = f['X'][:]
        labels = f['label'][:]
        windows = f['window'][:]
        ids = f['id'][:]
        section_ids = f['section_id'][:]
        
    if constrain_size:
        power = closest_power(X.shape[1], power=2)
        max_size = 2**power
        if max_size > X.shape[1]:
            power -= 1
            max_size = 2**power
        X = X[:,:max_size,:]
        labels = labels[:,:max_size,:]
    
    
    if normalize:
        
        if normalize_type == 'std':
            mean = np.expand_dims(np.mean(X[:,:,0], axis=1), axis=1)
            std = np.expand_dims(np.std(X[:,:,0], axis=1), axis=1)
            X[:,:,0] = (X[:,:,0]-mean)/std
        elif normalize_type == 'minmax':
            min_per_entry = np.expand_dims(np.min(X[:,:,0], axis=1), axis=1)
            X[:,:,0] += abs(min_per_entry)
            normalization_factor = np.expand_dims(np.max(abs(X[:,:,0]), axis=1), axis=1)
            X[:,:,0] /= normalization_factor
        labels[:,:,0] = X[:,:,0]
        
    return X, labels, windows, event_types, ids, section_ids

def preprocess_tr(st, id, origin_time, freqmin=0.1, freqmax=5., tmin=330., tmax=483.):
    #distances = [tr.stats.starttime - origin_time for tr in st]
    #idx = np.argsort(distances)
    tr = st[id].copy()
    try:
        tr.remove_response()
    except:
        pass
    tr.detrend()
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True, corners=6)
    tr.resample(2*freqmax)
    tr.trim(starttime=tr.stats.starttime+tmin, endtime=tr.stats.starttime+tmax)
    tr.taper(0.05)
    return tr

import numpy as np
import matplotlib.pyplot as plt
from pyrocko import moment_tensor as pmt
from obspy.geodetics import kilometers2degrees
from scipy.interpolate import RectBivariateSpline
from pyrocko.gf import LocalEngine, Target, DCSource, ws
from pyrocko import trace
from pyrocko.gui.marker import PhaseMarker
from pyrocko import gf

def create_trace(tr_pyrocko, sampling, station, freqmin=0.1, freqmax=5.):
    tr = obspy.Trace()
    tr.stats.station = station
    tr.data = tr_pyrocko.get_ydata()
    tr.stats.delta = 1./sampling
    tr.detrend()
    tr.filter('bandpass', freqmin=freqmin, freqmax=freqmax, zerophase=True, corners=6)
    tr.resample(freqmax*2.)
    return tr

def build_synthetic_traces(dists, depths, periods, base_folder, store_id, freqmin=0.1, freqmax=5., offset_trim_begin=40., offset_trim_end=140.):

    engine = gf.LocalEngine(store_dirs=[f'{base_folder}{store_id}/'])
    store = engine.get_store(store_id)
    sampling = store.config.sample_rate
    
    factor = 1.
    synthetic_traces = obspy.Stream()
    cpt_waveform = -1
    for dist, depth, period in zip(dists, depths, periods):

        cpt_waveform += 1
        stf = gf.BoxcarSTF(period, anchor=0.)
        
        waveform_targets = [
            gf.Target(
               quantity='velocity',
               lat=0., lon=lon,
               store_id=store_id,
               interpolation='multilinear',
               codes=('NET', 'STA', 'LOC', 'Z'))
            for lon in [dist]
            ]
        
        #factor = 1e20
        stf_add = {}
        if stf is not None:
            stf_add['stf'] = stf
        mt_source = gf.MTSource(
           lat=0., lon=0., depth=depth,
           mnn=0.*factor, mee=-1.*factor, mdd=1.*factor, mne=0.*factor, mnd=0.*factor, med=0.*factor,**stf_add)

        # The computation is performed by calling process on the engine
        response = engine.process(mt_source, waveform_targets)

        # convert results in response to Pyrocko traces
        for tr_pyrocko in response.pyrocko_traces():
            station = f'{cpt_waveform:05d}'
            synthetic_traces += create_trace(tr_pyrocko, sampling, station, freqmin=freqmin, freqmax=freqmax)
    
    synthetic_traces.trim(starttime=synthetic_traces[0].stats.starttime+offset_trim_begin, endtime=synthetic_traces[0].stats.starttime+offset_trim_end)

    return synthetic_traces

def build_perturbed_dataset(X_total, section_ids_all, st_input, SNR, sampling_X_total, offset_end_max=20., seed=1):

    np.random.seed(seed)

    #offset_end_max = 20. # seconds from the end allowed for the input data incorporation
    tmax_X_total = X_total.shape[1]/sampling_X_total
    X_total_c = X_total.copy()
    unique_section_ids = np.unique(section_ids_all)

    perturbed_idx = []
    pert_corresponding_input = []
    for itr, tr in enumerate(st_input):

        offset_first_window = np.random.rand()*(tmax_X_total-offset_end_max)
        duration_input_left = tr.times()[-1]
        n_windows_needed = int(np.ceil((duration_input_left+offset_first_window)/tmax_X_total))

        id_section_selected = np.random.randint(0, unique_section_ids.size)
        available_waveform_ids = np.where(section_ids_all==unique_section_ids[id_section_selected])[0]
        id_first = available_waveform_ids[np.random.randint(0, available_waveform_ids.size-1-n_windows_needed)]

        max_X_total = X_total[id_first:id_first+n_windows_needed,:,0].max()
        #max_X_total = np.std(X_total[id_first:id_first+n_windows_needed,:,0])
        input_data_max = abs(tr.data).max()
        perturbed_idx += [i for i in range(id_first, id_first+n_windows_needed)]
        pert_corresponding_input += [itr for _ in range(id_first, id_first+n_windows_needed)]

        istart_input = 0
        for iwindow in range(n_windows_needed):

            istart_window = 0
            if iwindow == 0:
                istart_window = int(offset_first_window*sampling_X_total)

            iend_window = X_total.shape[1]
            if iwindow == n_windows_needed-1:
                iend_window = int(duration_input_left*sampling_X_total)+1

            data_cropped = tr.data[istart_input:istart_input+(iend_window-istart_window)]
            """
            if id_first+iwindow == 1032:
                #print(max_X_total, SNR, data_cropped.max(), input_data_max)
                #print((max_X_total*SNR*data_cropped/input_data_max).max()/X_total_c[id_first+iwindow,istart_window:iend_window,0].max())
                plt.figure()
                plt.plot(X_total_c[id_first+iwindow,istart_window:iend_window,0])
                plt.plot(max_X_total*SNR*data_cropped/input_data_max)
            """
            X_total_c[id_first+iwindow,istart_window:iend_window,0] += max_X_total*SNR*data_cropped/input_data_max

            istart_input += iend_window-istart_window
            duration_input_left -= (iend_window-istart_window)/sampling_X_total
    
    return X_total_c, perturbed_idx, pert_corresponding_input

def build_perturbed_dataset_SNRs(SNRs, X_total, section_ids_all, st_real, sampling_X_total, offset_end_max=20., seed=10):

    X_total_SNR = {}
    perturbed_idx_SNR = {}
    pert_corresponding_input_SNR = {}
    #SNRs = [2.5]
    for SNR in SNRs:
        X_total_SNR[SNR], perturbed_idx_SNR[SNR], pert_corresponding_input_SNR[SNR] = build_perturbed_dataset(X_total, section_ids_all, st_real[:], SNR, sampling_X_total, offset_end_max=20., seed=10)

    return X_total_SNR, perturbed_idx_SNR, pert_corresponding_input_SNR

from obspy.signal.trigger import classic_sta_lta

def compute_sta_lta(X_total, X_total_SNR, sampling_X_total, section_ids_all, sta=1., lta=30.):
    sta_lta_SNR = {}
    sta_lta = np.zeros_like(X_total)
    for SNR, X_total_loc in X_total_SNR.items():
        unique_section_ids = np.unique(section_ids_all)
        sta_lta_SNR[SNR] = np.zeros_like(X_total_loc)
        for isection in unique_section_ids[:]:
            available_waveform_ids = np.where(section_ids_all==isection)[0]
            reshaped_waveforms = X_total_loc[available_waveform_ids,:].reshape(-1, X_total_loc.shape[2])[:,0]
            sta_lta_signal_loc = classic_sta_lta(reshaped_waveforms, int(sta * sampling_X_total), int(lta * sampling_X_total))
            sta_lta_SNR[SNR][available_waveform_ids,:] = sta_lta_signal_loc[:,None].reshape(X_total_loc[available_waveform_ids,:].shape)
            
            reshaped_waveforms = X_total[available_waveform_ids,:].reshape(-1, X_total.shape[2])[:,0]
            sta_lta_signal_loc = classic_sta_lta(reshaped_waveforms, int(sta * sampling_X_total), int(lta * sampling_X_total))
            sta_lta[available_waveform_ids,:] = sta_lta_signal_loc[:,None].reshape(X_total[available_waveform_ids,:].shape)
            
            
    return sta_lta_SNR, sta_lta

from scipy.signal import spectrogram
from matplotlib import patheffects
path_effects = [patheffects.withStroke(linewidth=3, foreground="w")]

def generate_colors(n, cmap_name='viridis'):
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i / n) for i in range(n)]
    return colors

def plot_pert_dataset(X_total, X_total_loc, sta_lta_signal_loc, sta_lta_signal, perturbed_idx_loc_in, pert_corresponding_input_loc, sampling_X_total, factor_spectro=3):

    perturbed_idx_loc = perturbed_idx_loc_in[:10]
    
    dict_Sxx = dict(nperseg=30, noverlap=20)

    fig = plt.figure(figsize=(10,8))
    l_spectro = len(perturbed_idx_loc)//factor_spectro
    l_total = len(perturbed_idx_loc) + l_spectro
    grid = fig.add_gridspec(l_total,2)

    ax_orig_spectro = fig.add_subplot(grid[:l_spectro,0])
    ax_orig = fig.add_subplot(grid[l_spectro:,0], sharex=ax_orig_spectro)
    ax_pert_spectro = fig.add_subplot(grid[:l_spectro,1])
    ax_pert = fig.add_subplot(grid[l_spectro:,1], sharex=ax_pert_spectro, sharey=ax_orig)
    
    ax_orig.set_xlabel('Time (s)')
    ax_orig.tick_params(axis='both', which='both', labelleft=False, left=False)
    ax_pert.set_xlabel('Time (s)')
    ax_pert.tick_params(axis='both', which='both', labelleft=False, left=False)
    
    ax_orig_spectro.set_ylabel('Frequency (Hz)')
    ax_orig_spectro.tick_params(axis='both', which='both', labelbottom=False, )
    ax_pert_spectro.tick_params(axis='both', which='both', labelleft=False, labelbottom=False,)

    max_val = X_total_loc[np.array(perturbed_idx_loc),:].max()
    max_val_sta = sta_lta_signal_loc[np.array(perturbed_idx_loc),:].max()
    colors = generate_colors(np.unique(pert_corresponding_input_loc).size, cmap_name='tab10')
    times = np.arange(0, X_total_loc.shape[1]/sampling_X_total, 1./sampling_X_total)
    for ioffset, idx in enumerate(perturbed_idx_loc[::-1]):

        if ioffset == len(perturbed_idx_loc)-1:
            frequencies_orig, times_orig, Sxx_orig = spectrogram(X_total[idx,:,0], fs=sampling_X_total, **dict_Sxx)
            ax_orig_spectro.pcolormesh(times_orig, frequencies_orig, Sxx_orig)

            frequencies_pert, times_pert, Sxx_pert = spectrogram(X_total_loc[idx,:,0], fs=sampling_X_total, **dict_Sxx)
            ax_pert_spectro.pcolormesh(times_pert, frequencies_pert, Sxx_pert)

        i_pert_corresponding_input_loc = pert_corresponding_input_loc[ioffset]
        mean, std = X_total[idx,:,0].mean(), np.std(X_total[idx,:,0])
        waveform = (X_total[idx,:,0]-mean)/std
        ax_orig.plot(times, waveform/max_val+ioffset, color=colors[i_pert_corresponding_input_loc])
        ax_pert.plot(times, X_total_loc[idx]/max_val+ioffset, color=colors[i_pert_corresponding_input_loc])
        
        #print(sta_lta_signal_loc[idx])
        ax_orig.plot(times, sta_lta_signal[idx]/max_val_sta+ioffset, color='black', alpha=0.75)
        ax_pert.plot(times, sta_lta_signal_loc[idx]/max_val_sta+ioffset, color='black', alpha=0.75)

from tsfresh import extract_features
def array_to_dataframe(array, sampling_freq=10.):
    # Get shape of array
    A, B = array.shape
    
    # Flatten the array into a single column of data points
    data = array.flatten()
    
    # Create an ID array corresponding to each row's data
    ids = np.repeat(np.arange(A), B)
    
    # Create a time array corresponding to each column's index
    time_index = np.tile(np.arange(B), A)
    
    # Combine these into a DataFrame
    df = pd.DataFrame({
        'id': ids,
        'time': (time_index).astype(float)/sampling_freq,
        'pressure': data
    })
    
    return df

def extract_features_SNRs(X_total_SNR, sampling_X_total):

    extracted_features_SNR = {}
    for SNR in X_total_SNR:
        pd_data = array_to_dataframe(X_total_SNR[SNR][:,:,0], sampling_freq=sampling_X_total)
        extracted_features_SNR[SNR] = extract_features(pd_data, column_id="id", column_sort="time", n_jobs=10)

    return extracted_features_SNR

from tsfresh.feature_selection.relevance import calculate_relevance_table

def compute_best_features(perturbed_idx_SNR, idx_train, extracted_features_SNR, reference_SNR, n_inputs, n_features, plot=False):
    relevance_table_SNR = {}
    for iSNR, SNR in enumerate(extracted_features_SNR):


        labels = np.zeros(n_inputs, dtype=int)
        labels[np.array(perturbed_idx_SNR[SNR])] = 1
        labels = labels[idx_train]

        extracted_features = extracted_features_SNR[SNR].loc[extracted_features_SNR[SNR].index.isin(idx_train)]
        extracted_features_nona = extracted_features.dropna(axis=1)
        relevance_table = calculate_relevance_table(extracted_features_nona, pd.Series(labels))
        #relevance_table = relevance_table[relevance_table.relevant]
        relevance_table['p_value'] = relevance_table['p_value'].fillna(0)
        relevance_table.reset_index(inplace=True, drop=True)
        relevance_table_sorted = relevance_table.loc[relevance_table.p_value>0.].sort_values("p_value")
        if SNR == reference_SNR:
            #order_features = relevance_table_sorted.feature.values[-n_features:]
            order_features = relevance_table_sorted.index.values[:n_features]
            best_features = relevance_table_sorted.feature.values[:n_features]
        relevance_table_SNR[SNR] = relevance_table

        print(best_features)

    if plot:
        height = 0.25
        fig = plt.figure(figsize=(12,14))
        for iSNR, SNR in enumerate(relevance_table_SNR):    
            selected_features = relevance_table_SNR[SNR].iloc[order_features]#loc[relevance_table_SNR[SNR].feature.isin(order_features)]
            p_values = selected_features.p_value.values
            #print(selected_features.shape, relevance_table_sorted.shape, extracted_features.shape, extracted_features_nona.shape)

            #plt.barh(relevance_table.feature.values[-n_features:], relevance_table.p_value.values[-n_features:])
            plt.barh(np.arange(n_features)+height*iSNR, p_values, height=height, label=SNR)

        plt.yticks(np.arange(n_features), relevance_table.feature.values[-n_features:])
        plt.legend(title='SNR')
        #plt.xlim([0.9, 1.03])
        fig.subplots_adjust(left=0.5)
        plt.xscale('log')
        
    return relevance_table_SNR, best_features

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest

from pyod.models.auto_encoder import AutoEncoder                 
from pyod.models.ecod import ECOD               
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest       
from pyod.models.lof import LOF

def return_model_and_scores(method, X_in, X_test, sta_lta_signal_pert_train, sta_lta_signal_pert_test, add_STA_LTA, contamination, max_samples, n_estimators, seed=50):
    added_to_pipe = []
    if add_STA_LTA:
        sta_lta_max = sta_lta_signal_pert_train.max(axis=1)[:,0]
        X_in = np.c_[X_in, sta_lta_max]
        sta_lta_max = sta_lta_signal_pert_test.max(axis=1)[:,0]
        X_test = np.c_[X_test, sta_lta_max]

    fastmapsvm = None
    if method == 'isolation_forest':
        added_to_pipe.append( ('iso', IsolationForest(bootstrap=False, contamination=contamination, n_estimators=n_estimators)) )
    elif method == 'IForest':
        added_to_pipe.append( ('iforest', IForest(bootstrap=True, contamination=contamination, n_estimators=n_estimators)) )
    elif method == 'ecod':
        added_to_pipe.append( ('ecod', ECOD(contamination=contamination)) )
    elif method == 'copod':
        added_to_pipe.append( ('copod', COPOD(contamination=contamination)) )
    elif method == 'lof':
        added_to_pipe.append( ('lof', LOF(contamination=contamination)) )
    elif method == 'autoencoder':
        #pyod.models.auto_encoder.AutoEncoder(contamination=0.1, preprocessing=True, lr=0.001, epoch_num=10, batch_size=32, optimizer_name='adam', device=None, random_state=42, use_compile=False, compile_mode='default', verbose=1, optimizer_params: dict = {'weight_decay': 1e-05}, hidden_neuron_list=[64, 32], hidden_activation_name='relu', batch_norm=True, dropout_rate=0.2)[source]
        added_to_pipe.append( ('autoencoder', AutoEncoder(contamination=contamination, device='cpu', epoch_num=50, hidden_neuron_list=[64, 32, 16])) )
    else:
        added_to_pipe.append( ('scaler', StandardScaler()) )
        added_to_pipe.append( ('svc', RandomForestClassifier()) )
    fastmapsvm = Pipeline(added_to_pipe)
    fastmapsvm.fit(X_in, y=None)

    scores = fastmapsvm.decision_function(X_in)
    scores_test = fastmapsvm.decision_function(X_test)
    
    return fastmapsvm, scores, scores_test

from scipy.signal import spectrogram
from matplotlib import patheffects
from tqdm import tqdm
path_effects = [patheffects.withStroke(linewidth=3, foreground="w")]

def compute_confusion(idx_test, idx_pert_in_test, idx_pert_in_normal_test, idx_anomaly_test, idx_normal_test, idx_pert_in_anomaly_test,
                     idx_train, idx_pert_in_train, idx_pert_in_normal_train, idx_anomaly_train, idx_normal_train, idx_pert_in_anomaly_train):
    
    POS = idx_pert_in_test.size
    NEG = idx_test.size-POS
    TP = len(idx_pert_in_anomaly_test)
    FP = idx_anomaly_test.size - TP
    FN = len(idx_pert_in_normal_test)
    TN = idx_normal_test.size - FN

    POS_train = idx_pert_in_train.size
    NEG_train = idx_train.size-POS_train
    TP_train = len(idx_pert_in_anomaly_train)
    FP_train = idx_anomaly_train.size - TP_train
    FN_train = len(idx_pert_in_normal_train)
    TN_train = idx_normal_train.size - FN_train

    TPR_train = TP_train/POS_train
    TNR_train = TN_train/NEG_train
    FPR_train = FP_train/NEG_train
    FNR_train = FN_train/POS_train

    TPR = TP/POS
    TNR = TN/NEG
    FPR = FP/NEG
    FNR = FN/POS
    
    return TPR, TNR, FPR, FNR, TPR_train, TNR_train, FPR_train, FNR_train

def get_indexes(idx_train, idx_test, idx_pert_all, scores_train, scores_test, threshold, method):
    
    if not method == 'isolation_forest':
        bool_anomaly_in_train = scores_train>=threshold
        bool_normal_in_train = scores_train<threshold
        bool_anomaly_in_test = scores_test>=threshold
        bool_normal_in_test = scores_test<threshold
        
    else:
        bool_anomaly_in_train = scores_train<threshold
        bool_normal_in_train = scores_train>=threshold
        bool_anomaly_in_test = scores_test<threshold
        bool_normal_in_test = scores_test>=threshold

    idx_anomaly_train = idx_train[bool_anomaly_in_train]
    idx_normal_train = idx_train[bool_normal_in_train]
    idx_anomaly_test = idx_test[bool_anomaly_in_test]
    idx_normal_test = idx_test[bool_normal_in_test]

    idx_pert_in_normal_test = np.intersect1d(idx_pert_all, idx_normal_test)
    idx_pert_in_anomaly_test = np.intersect1d(idx_pert_all, idx_anomaly_test)
    idx_pert_in_normal_train = np.intersect1d(idx_pert_all, idx_normal_train)
    idx_pert_in_anomaly_train = np.intersect1d(idx_pert_all, idx_anomaly_train)
    
    return idx_pert_in_normal_test, idx_anomaly_test, idx_normal_test, idx_pert_in_anomaly_test, idx_pert_in_normal_train, idx_anomaly_train, idx_normal_train, idx_pert_in_anomaly_train

def compute_performances(idx_train, idx_test, idx_pert_all, scores_train, scores_test, thresholds, method):

    idx_pert_in_train = np.intersect1d(idx_pert_all, idx_train)
    idx_pert_in_test = np.intersect1d(idx_pert_all, idx_test)

    s_total_train = idx_train.size
    s_total_test = idx_test.size
    
    performances = pd.DataFrame()
    for threshold in tqdm(thresholds):
    
        idx_pert_in_normal_test, idx_anomaly_test, idx_normal_test, idx_pert_in_anomaly_test, idx_pert_in_normal_train, idx_anomaly_train, idx_normal_train, idx_pert_in_anomaly_train = get_indexes(idx_train, idx_test, idx_pert_all, scores_train, scores_test, threshold, method)

        s_anomaly_test = idx_anomaly_test.size
        s_anomaly_train = idx_anomaly_train.size
        
        TPR, TNR, FPR, FNR, TPR_train, TNR_train, FPR_train, FNR_train = compute_confusion( 
                          idx_test, idx_pert_in_test, idx_pert_in_normal_test, idx_anomaly_test, idx_normal_test, idx_pert_in_anomaly_test,
                          idx_train, idx_pert_in_train, idx_pert_in_normal_train, idx_anomaly_train, idx_normal_train,idx_pert_in_anomaly_train)
        performance = dict(TPR=TPR, TNR=TNR, FPR=FPR, FNR=FNR, perc_data_sent_test=s_anomaly_test/s_total_test, 
                           TPR_train=TPR_train, TNR_train=TNR_train, FPR_train=FPR_train, FNR_train=FNR_train, perc_data_sent_train=s_anomaly_train/s_total_train, 
                           threshold=threshold)
        performances = pd.concat([performances, pd.DataFrame([performance])])
        
    performances.reset_index(drop=True, inplace=True)
    return performances

from matplotlib import colors
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def add_cbar(ax, sc, caption, x=1.01):
        
    axins = inset_axes(ax, width="4.%", height="75%", loc='lower left', 
                        bbox_to_anchor=(x, 0.25, 1, 1.), bbox_transform=ax.transAxes, borderpad=0)
    axins.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False, labelrotation=0.)
    cbar = plt.colorbar(sc, cax=axins, extend='both', orientation="vertical")  
    cbar.ax.xaxis.set_ticks_position('top') 
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.tick_top()
    cbar.ax.set_ylabel(caption, labelpad=1) 

def plot_performances(file_figure, waveforms_all, perturbed_idx, idx_train, idx_test, performances, X_train, X_test, scores_train, scores_test, sampling_X_total, method, fastmapsvm, performances_sta_lta=None, max_waveforms=10, threshold=None, threshold_mission=0.05):
    
    if threshold == None:
        if method not in ['isolation_forest']:
            threshold = fastmapsvm[-1].threshold_
        else:
            threshold = 0.

    if method in ['isolation_forest',]:
        divnorm=colors.TwoSlopeNorm(vmin=scores_train.min(), vcenter=0., vmax=scores_train.max())
    elif method in ['ecod', 'copod', 'autoencoder', 'IForest', 'lof']:
        divnorm=colors.TwoSlopeNorm(vmin=scores_train.min(), vcenter=threshold, vmax=scores_train.max())
    else:
        print(f'Method {method} not recognized')
        return

    scores_all = np.zeros(waveforms_all.shape[0])
    scores_all[idx_train] = scores_train
    scores_all[idx_test] = scores_test
    
    times = np.arange(waveforms_all.shape[1])/sampling_X_total
    
    ## Finding indexes in each subset
    idx_pert_in_normal_test, idx_anomaly_test, idx_normal_test, idx_pert_in_anomaly_test, dx_pert_in_normal_train, idx_anomaly_train, idx_normal_train, idx_pert_in_anomaly_train = get_indexes(idx_train, idx_test, perturbed_idx, scores_train, scores_test, threshold, method)
    
    ## Selecting waveforms in test set

    ## Normal
    idx_waveform_normal = idx_pert_in_normal_test[:max_waveforms]
    added_waveforms = max_waveforms-idx_waveform_normal.size
    label_normal = np.zeros(max_waveforms)
    label_normal[:idx_waveform_normal.size] = 1.
    if added_waveforms > 0:
        idx_normal_in_normal_test = np.setdiff1d(idx_normal_test, idx_waveform_normal)
        idx_waveform_normal = np.r_[idx_waveform_normal, idx_normal_in_normal_test[:added_waveforms]]
    waveforms_normal = waveforms_all[idx_waveform_normal,:]
    scores_normal = scores_all[idx_waveform_normal]

    scores_normal, waveforms_normal, label_normal = scores_normal[::-1], waveforms_normal[::-1,:], label_normal[::-1]

    ## Anomaly
    idx_waveform_anomaly = idx_pert_in_anomaly_test[:max_waveforms]
    added_waveforms = max_waveforms-idx_waveform_anomaly.size
    label_anomaly = np.zeros(max_waveforms)
    label_anomaly[:idx_waveform_anomaly.size] = 1.
    if added_waveforms > 0:
        idx_normal_in_anomaly_test = np.setdiff1d(idx_anomaly_test, idx_waveform_anomaly)
        idx_waveform_anomaly = np.r_[idx_waveform_anomaly, idx_normal_in_anomaly_test[:added_waveforms]]
    added_waveforms = max_waveforms-idx_waveform_anomaly.size
    if added_waveforms > 0:
        idx_waveform_anomaly = np.r_[idx_waveform_anomaly, idx_pert_in_anomaly_train[:added_waveforms]]
        label_anomaly[added_waveforms:] = 1.
    waveforms_anomaly = np.zeros_like(waveforms_normal) + np.nan
    waveforms_anomaly[:idx_waveform_anomaly.size] = waveforms_all[idx_waveform_anomaly,:]
    scores_anomaly = np.zeros_like(scores_normal) + np.nan
    scores_anomaly[:idx_waveform_anomaly.size] = scores_all[idx_waveform_anomaly]

    scores_anomaly, waveforms_anomaly, label_anomaly = scores_anomaly[::-1], waveforms_anomaly[::-1,:], label_anomaly[::-1]

    #print(idx_waveform_anomaly)

    #print('idx_test', idx_test.size, idx_anomaly_test.size, idx_anomaly_train.size, idx_pert_in_anomaly_train.size, idx_pert_in_anomaly_test.size, added_waveforms)
    #print(max_waveforms, idx_pert_in_anomaly_test.size, idx_anomaly_test.size, idx_test.size, idx_waveform_anomaly.size, waveforms_anomaly.shape)

    loc_performances = performances.loc[abs(performances.threshold-threshold)==abs(performances.threshold-threshold).min()]
    
    ## TSNE clustering
    W_all = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=3).fit_transform(np.r_[X_train,X_test])
    W_train = W_all[:X_train.shape[0]]
    W_test = W_all[X_train.shape[0]:]

    intersection_values = np.intersect1d(perturbed_idx, idx_train)
    idx_pert_train = np.nonzero(np.in1d(idx_train, intersection_values))[0]

    intersection_values = np.intersect1d(perturbed_idx, idx_test)
    idx_pert_test = np.nonzero(np.in1d(idx_test, intersection_values))[0]

    FPR_mission_train = performances.loc[performances.perc_data_sent_train<=threshold_mission, 'FPR_train'].max()
    FPR_mission_test = performances.loc[performances.perc_data_sent_test<=threshold_mission, 'FPR'].max()
    
    fig = plt.figure(figsize=(10,8))
    h_total = 10
    w_total = 12
    perc_roc = 0.3
    perc_spectro = 0.2
    perc_waveform_width = 0.35
    h_waveforms = h_total-int(perc_spectro*h_total)
    w_waveforms = int(perc_waveform_width*w_total)
    h_roc = int(perc_roc*h_total)
    grid = fig.add_gridspec(h_total, w_total)
    
    ax_clustrer = fig.add_subplot(grid[:h_total-2*h_roc,2*w_waveforms:])
    sc_train = ax_clustrer.scatter(W_train[:, 0], W_train[:, 1], s=4, c=scores_train, cmap='coolwarm', norm=divnorm, zorder=10, marker='o')
    sc = ax_clustrer.scatter(W_test[:, 0], W_test[:, 1], s=4, c=scores_test, cmap='coolwarm', norm=divnorm, zorder=10, marker='^')
    #plt.colorbar(sc)
    sc = ax_clustrer.scatter(W_train[idx_pert_train,0], W_train[idx_pert_train,1], facecolor=None, edgecolor='black', s=35)
    sc.set_facecolor("none")
    sc = ax_clustrer.scatter(W_test[idx_pert_test,0], W_test[idx_pert_test,1], facecolor=None, edgecolor='black', s=35)
    sc.set_facecolor("none")
    #ax_clustrer.set_xlabel('$w_0$')
    #ax_clustrer.set_ylabel('$w_1$')
    ax_clustrer.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False, )
    add_cbar(ax_clustrer, sc_train, 'Score', x=1.2)
    
    ax_roc_train = fig.add_subplot(grid[-2*h_roc:-h_roc, 2*w_waveforms:])
    sc = ax_roc_train.scatter(performances.FPR_train, performances.TPR_train, c=performances.threshold, zorder=10)
    ax_roc_train.text(performances.FPR_train.max(), performances.TPR_train.min(), 'Train', ha='right', va='bottom')
    ax_roc_train.tick_params(axis='both', which='both', labelbottom=False, )
    ax_roc_train.yaxis.tick_right()
    add_cbar(ax_roc_train, sc, 'Threshold', x=1.2)
    sc = ax_roc_train.scatter(loc_performances.FPR_train, loc_performances.TPR_train, facecolor=None, edgecolor='tab:red', s=40, zorder=100)
    sc.set_facecolor("none")
    if performances_sta_lta is not None:
        _ = ax_roc_train.plot(performances_sta_lta.FPR_train, performances_sta_lta.TPR_train, linestyle=':', color='black', zorder=1)
    #ax_roc_train.plot([0., 1.], [0., 1.], color='tab:red', alpha=0.4, linestyle='--', zorder=1)
    ax_roc_train.plot(performances.FPR_train, performances.perc_data_sent_train, color='magenta', zorder=2, alpha=0.5, label='\% data sent')
    ax_roc_train.grid(alpha=0.3)
    ax_roc_train.axvspan(0., FPR_mission_train, color='tab:red', alpha=0.2, zorder=0)
    ax_roc_train.set_ylim([0., 1.])
    ax_roc_train.set_xlim([0., 1.])

    ax_roc_test = fig.add_subplot(grid[-h_roc:, 2*w_waveforms:], sharex=ax_roc_train, sharey=ax_roc_train)
    ax_roc_test.scatter(performances.FPR, performances.TPR, c=performances.threshold, zorder=10, label=method)
    ax_roc_test.set_xlabel('False Positive Rate')
    ax_roc_test.set_ylabel('True Positive Rate')
    #ax_roc_test.text(performances.FPR_train.max(), performances.TPR_train.min(), 'Test', ha='right', va='bottom')
    ax_roc_test.yaxis.tick_right()
    ax_roc_test.yaxis.set_label_position("right")
    sc = ax_roc_test.scatter(loc_performances.FPR, loc_performances.TPR, facecolor=None, edgecolor='tab:red', s=40, zorder=100)
    sc.set_facecolor("none")
    if performances_sta_lta is not None:
        _ = ax_roc_test.plot(performances_sta_lta.FPR, performances_sta_lta.TPR, linestyle=':', color='black', zorder=1, label='sta lta')
    ax_roc_test.plot(performances.FPR, performances.perc_data_sent_test, color='magenta', zorder=2, alpha=0.5, label='% data sent')
    ax_roc_test.legend(frameon=False, title='Test')
    ax_roc_test.grid(alpha=0.3)
    ax_roc_test.axvspan(0., FPR_mission_test, color='tab:red', alpha=0.2, zorder=0)
    #ax_roc_test.plot([0., 1.], [0., 1.], color='tab:red', alpha=0.4, linestyle='--', zorder=1)

    ax_waveform_anomaly = fig.add_subplot(grid[-h_waveforms:, :w_waveforms])
    ax_waveform_anomaly.tick_params(axis='both', which='both', labelleft=False, left=False)
    ax_waveform_anomaly.set_xlabel('Time (s)')
    
    ax_spectro_anomaly = fig.add_subplot(grid[:-h_waveforms,:w_waveforms])
    ax_spectro_anomaly.tick_params(axis='both', which='both', labelbottom=False, )
    frequencies, times_Sxx, Sxx = spectrogram(waveforms_anomaly[-1,:], fs=1./times[1], nperseg=30, noverlap=25)
    Sxx_anomaly_max = Sxx.max()
    ax_spectro_anomaly.pcolormesh(times_Sxx, frequencies, np.log10(Sxx/Sxx_anomaly_max), shading='gouraud')
    ax_spectro_anomaly.set_ylabel('Freq (Hz)')
    ax_spectro_anomaly.set_title(f'Anomalies')
    
    ax_waveform_normal = fig.add_subplot(grid[-h_waveforms:, w_waveforms:2*w_waveforms])
    ax_waveform_normal.tick_params(axis='both', which='both', labelleft=False, )
    
    ax_spectro_normal = fig.add_subplot(grid[:-h_waveforms,w_waveforms:2*w_waveforms])
    ax_spectro_normal.tick_params(axis='both', which='both', labelleft=False, labelbottom=False, )
    frequencies, times_Sxx, Sxx = spectrogram(waveforms_normal[-1,:], fs=1./times[1], nperseg=30, noverlap=25)
    ax_spectro_normal.pcolormesh(times_Sxx, frequencies, np.log10(Sxx/Sxx_anomaly_max), shading='gouraud')
    ax_spectro_normal.set_title('Normal')
    
    max_val = max(np.nanmax(waveforms_anomaly), np.nanmax(waveforms_normal))
    #for i in range(waveforms_anomaly.shape[0]-1, -1, -1):
    for i in range(waveforms_anomaly.shape[0]):
        color = 'black' if label_normal[i] == 0. else 'tab:red'
        ax_waveform_normal.plot(times, waveforms_normal[i,:]/max_val+i, color=color)
        ax_waveform_normal.text(times[-1], i, f'{scores_normal[i]:.2f}', ha='right', va='center', path_effects=path_effects)
        if np.isnan(scores_anomaly[i]):
            continue
        color = 'black' if label_anomaly[i] == 0. else 'tab:red'
        ax_waveform_anomaly.plot(times, waveforms_anomaly[i,:]/max_val+i, color=color)
        ax_waveform_anomaly.text(times[-1], i, f'{scores_anomaly[i]:.2f}', ha='right', va='center', path_effects=path_effects)
        
    ax_waveform_normal.set_ylim([-1, max_waveforms])
    ax_waveform_anomaly.set_ylim([-1, max_waveforms])

    fig.subplots_adjust(right=0.8)
    fig.savefig(file_figure)

def normalization(X_total_SNR):

    for SNR in X_total_SNR:
        mean = np.expand_dims(np.mean(X_total_SNR[SNR][:,:,0], axis=1), axis=1)
        std = np.expand_dims(np.std(X_total_SNR[SNR][:,:,0], axis=1), axis=1)
        X_total_SNR[SNR][:,:,0] = (X_total_SNR[SNR][:,:,0]-mean)/std

    return X_total_SNR

def find_splits(X_total_shape, perc_train=0.7):

    last_ind_train = int(X_total_shape*perc_train)
    idx_train = np.arange(last_ind_train)
    idx_test = np.arange(last_ind_train, X_total_shape)

    return idx_train, idx_test

def get_training_testing_data(idx_train, idx_test, extracted_features, X_total, best_features, method):

    if method == 'autoencoder':
        X_in = X_total[idx_train,:,0]
        X_test = X_total[idx_test,:,0]
    else:
        X_in = extracted_features.loc[extracted_features.index.isin(idx_train),extracted_features.columns.isin(best_features)].values
        X_test = extracted_features.loc[extracted_features.index.isin(idx_test),extracted_features.columns.isin(best_features)].values 

    return X_in, X_test

def load_real_data(file, nmax=10, freqmin=0.15, freqmax=2.5, tmin=340., tmax=503.):

    #file = '/projects/infrasound/data/infrasound/2023_ML_balloon/data/data_alaska/ev_2020-10-19T20_Mw7.6.mseed'
    st = obspy.read(file)
    file = file.replace('.mseed', '_metadata.csv')
    metadata = pd.read_csv(file, header=[0])
    selected = np.arange(len(st))
    origin_time = UTCDateTime(metadata.ev_time.iloc[0])
    
    st_real = obspy.Stream()
    opt_processing = dict(freqmin=freqmin, freqmax=freqmax, tmin=tmin, tmax=tmax)
    for itr in selected[:nmax]:
        st_real += preprocess_tr(st, itr, origin_time, **opt_processing)

    return st_real

def correlate(a, b, axis=-1):
    A = np.fft.rfft(a, axis=axis)
    B = np.fft.rfft(b, axis=axis)
    X = A * np.conj(B)
    x = np.fft.irfft(X)
    x = np.fft.fftshift(x, axes=axis)
    norm = np.sqrt(
        a.shape[-1] * np.var(a, axis=axis)
        *
        b.shape[-1] * np.var(b, axis=axis)
    )
    norm = norm[..., np.newaxis]

    return np.nan_to_num(x / norm, neginf=0, posinf=0)

def correlation_distance(a, b, axis=-1):
    '''
    Compute the pair-wise correlation distance matrix.
    '''
    xcorr = correlate(a, b, axis=axis)
    xcorr = np.abs(xcorr)
    xcorr = np.nanmean(xcorr, axis=-2)
    xcorr = np.max(xcorr, axis=-1)
    xcorr = np.clip(xcorr, 0, 1)

    return 1 - xcorr

class FastMap(fastmap.FastMapABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distance_func = correlation_distance

##########################
if __name__ == '__main__':

    root_dir = '/projects/infrasound/data/infrasound/2023_ML_balloon/data/'
    dataset_name = 'minibooster'
    dataset_validation_name = 'minibooster'
    h5_file  = f'all_waveform_dataset_{dataset_name}_0.015Hz.h5'
    sampling_X_total = 5.
    X_total, labels_all, windows_all, event_types_all, ids_all, section_ids_all = load_data(root_dir, h5_file, normalize_type='none')

    #####################################################
    ## Select subset of data to reduce computational time
    max_id = 1100
    X_total = X_total[:max_id]
    labels_all = labels_all[:max_id]
    windows_all = windows_all[:max_id]
    event_types_all = event_types_all[:max_id]
    ids_all = ids_all[:max_id]
    section_ids_all = section_ids_all[:max_id]

    ##############
    ## Load Alaska
    file = '/projects/infrasound/data/infrasound/2023_ML_balloon/data/data_alaska/ev_2020-10-19T20_Mw7.6.mseed'
    st_real = load_real_data(file, nmax=10, freqmin=0.15, freqmax=2.5, tmin=340., tmax=503.)

    ##################
    ## Load synthetics
    ## Number of synthetic waveforms
    perc_pert = 1.5
    n_pert = int((perc_pert/100.)*X_total.shape[0])

    ## Discretization
    dists = np.arange(1., 5., 0.01)[:] # in degrees
    depths = np.linspace(10e3, 30e3, 500)
    period_stf = np.linspace(1., 20., 100)
    DISTS, DEPTHS, PERIODS = np.meshgrid(dists, depths, period_stf)
    DISTS, DEPTHS, PERIODS = DISTS.ravel(), DEPTHS.ravel(), PERIODS.ravel()
    idx = np.random.randint(0, DISTS.size, n_pert)
    DISTS, DEPTHS, PERIODS = DISTS[idx], DEPTHS[idx], PERIODS[idx]

    ## Greens functions STORES
    base_folder = '/projects/infrasound/data/infrasound/2023_Venus_inversion/'
    store_id = 'GF_venus_small'

    st_real = build_synthetic_traces(DISTS, DEPTHS, PERIODS, base_folder, store_id, freqmin=0.15, freqmax=2.5, offset_trim_begin=40., offset_trim_end=140.)
    
    ##########################
    ## Build perturbed dataset
    SNRs = [2.5]
    X_total_SNR, perturbed_idx_SNR, pert_corresponding_input_SNR = build_perturbed_dataset_SNRs(SNRs, X_total, section_ids_all, st_real, sampling_X_total, offset_end_max=20., seed=10)

    ##########
    ## STA/LTA
    sta_lta_signal_pert_SNR, sta_lta_signal = compute_sta_lta(X_total, X_total_SNR, sampling_X_total, section_ids_all, sta=1., lta=50.)

    ################
    ## Normalization
    X_total_SNR = normalization(X_total_SNR)

    #####################
    ## Feature extraction
    extracted_features_SNR = extract_features_SNRs(X_total_SNR, sampling_X_total)

    ###################
    ## Train/test split
    perc_train = 0.7
    SNR = [key for key in X_total_SNR.keys()][0]
    idx_train, idx_test = find_splits(X_total_SNR[SNR].shape[0], perc_train=perc_train)

    #####################
    ## Find best features
    n_features = 25
    reference_SNR = [SNR for SNR in X_total_SNR.keys()][0]
    n_inputs = X_total_SNR[reference_SNR].shape[0]
    relevance_table_SNR, best_features = compute_best_features(perturbed_idx_SNR, idx_train, extracted_features_SNR, reference_SNR, n_inputs, n_features, plot=False)

    ##################################
    ## Feature selection in input data
    SNR = [SNR for SNR in X_total_SNR.keys()][0]
    X_in, X_test = get_training_testing_data(idx_train, extracted_features_SNR[SNR], best_features)

    ####################
    ## Score computation
    methods = ['ecod', 'copod', 'IForest', 'autoencoder']
    add_STA_LTA = False
    contamination = 0.05
    max_samples = 'auto'
    n_estimators = 100

    for method in methods:

        fastmapsvm, scores, scores_test = return_model_and_scores(method, X_in, X_test, sta_lta_signal_pert_SNR[SNR][idx_train], sta_lta_signal_pert_SNR[SNR][idx_test], add_STA_LTA, contamination, max_samples, n_estimators, seed=50)

        #######################
        ## Compute performances
        ## METHOD
        thresholds = np.linspace(np.quantile(scores, q=0.01), np.quantile(scores, q=0.99), 200)
        performances = compute_performances(idx_train, idx_test, perturbed_idx_SNR[SNR], scores, scores_test, thresholds, method)

        ## STA/LTA
        sta_lta_max_train = sta_lta_signal_pert_SNR[SNR][idx_train,:,0].max(axis=1)
        sta_lta_max_test = sta_lta_signal_pert_SNR[SNR][idx_test,:,0].max(axis=1)
        thresholds = np.linspace(np.quantile(sta_lta_max_train, q=0.01), np.quantile(sta_lta_max_train, q=0.99), 2000)
        performances_sta_lta = compute_performances(idx_train, idx_test, perturbed_idx_SNR[SNR], sta_lta_max_train, sta_lta_max_test, thresholds, 'sta_lta')

        ####################
        ## Plot performances
        waveforms_all = X_total_SNR[SNR][:,:,0]
        threshold = None
        file_figure = f'./figures/performances_{method}_{dataset_name}_{dataset_validation_name}_SNR{SNR}.pdf'
        plot_performances(file_figure, waveforms_all, perturbed_idx_SNR[SNR], idx_train, idx_test, performances, X_in, X_test, scores, scores_test, sampling_X_total, method, fastmapsvm, performances_sta_lta=performances_sta_lta, max_waveforms=10, threshold=threshold)

    bp()