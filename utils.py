import numpy as np
import matplotlib.pyplot as plt
import mne # needed for the topoplot
import nibabel
from scipy import signal
from scipy.stats import zscore
from scipy.ndimage.filters import gaussian_filter

### FIXME: is there a smarter way to do this? Maybe a util 
import platform
location = platform.system()
dirct = dict(collab =  '/content/gdrive/My Drive/',
              Darwin = '/Users/lrg1213/',
              Linux = '/share/volume0/lwehbe/MEG18/')[location]



def delay_one(mat, d):
	# delays a matrix by a delay d. Positive d ==> row t has row t-d
    new_mat = np.zeros_like(mat)
    if d>0:
        new_mat[d:] = mat[:-d]
    elif d<0:
        new_mat[:d] = mat[-d:]
    else:
        new_mat = mat
    return new_mat

def delay_mat(mat, delays):
	# delays a matrix by a set of delays d.
	# a row t in the returned matrix has the concatenated: 
	# row(t-delays[0],t-delays[1]...t-delays[last] )
    new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
    return new_mat

 
import csv

# with open(dirct + 'data/HP/meg/locations.txt', 'r') as f:
#     locs = csv.reader(f,delimiter=',')
#     loc306 = np.array([[float(w1[0].split(' ')[1]),float(w1[0].split(' ')[2])] for w1 in locs ])

# loc102 = loc306[::3]

# from sklearn.metrics.pairwise import euclidean_distances

# dists = euclidean_distances(loc102, loc306)

# neighbors = np.argsort(dists,axis = 1)

# neighbors = neighbors[:,:27]

# sensor_groups = np.zeros((102,306))

# for i in range(102):
#     sensor_groups[i,neighbors[i]] = 1



def topoplot(mat, nrow = 4, ncol = 5, time_step = 25, time_start = 0, cmap = 'RdBu_r',
             vmin = -0.1, vmax = 0.1, figsize = (15,15), fontsize = 16):
    
    # check we have the right number of channels
    # assert mat.shape[0] in [306,102]
    loc = {306:loc306,102:loc102}[mat.shape[0]] # pick the correct channel locations

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol,figsize = figsize)
    i = 0
    for row in ax:
        for col in row:
            if i<mat.shape[1]:
                h = mne.viz.plot_topomap(mat[:,i],loc, vmin = vmin , vmax = vmax ,axes = col, cmap = cmap,
                                     show = False)
            i+=1
    i = 0
    for row in ax:
        for col in row:
            col.set_title('{} to {} ms'.format(i*time_step+time_start, 
                                               (i+1)*time_step+time_start), fontsize = fontsize)
            i+=1
            
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([1.05, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(h[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.tight_layout()
    return fig


def make_sub_figures(prefix_results, titles, prefix,
                     result_suffix = '',
                     save_directory= '/Users/lwehbe/results/figures/meg_res/', 
                     start_time = -40,stop_time = 60,
                     make_time = True,make_spatial=False, ylims = [-0.05,0.25],
                     colors =['blue','red','purple','green','darkorange','cyan']):
    
    acc_time = np.load(prefix_results+'_time{}.npy'.format(result_suffix))
    acc_time_std = np.load(prefix_results+'_std_time{}.npy'.format(result_suffix))
    res = [acc_time, acc_time_std] 

    try:
        acc_time_sensor = np.load(prefix_results+'_time_sensor{}.npy'.format(result_suffix))
        acc_time_sensor_std = np.load(prefix_results+'_std_time_sensor{}.npy'.format(result_suffix))
        res.append(acc_time_sensor)
        res.append(acc_time_sensor_std)
    except: 
        print('no spatial results')

    
    nF = acc_time.shape[0]-1
    
    if make_time:
        n_rows = max([int(np.ceil(nF/2)),2])

        fig, ax = plt.subplots(nrows=n_rows, ncols=2,figsize = (16,4*n_rows))

        min_sig_all = acc_time[-1]-acc_time_std[-1]/np.sqrt(1000)
        max_sig_all = acc_time[-1]+ acc_time_std[-1]/np.sqrt(1000)


        time = np.arange(start_time,stop_time)*25

        idx = 0
        
        plt.rcParams.update({'font.size': 18})

        for row in ax:
            for col in row:
                if idx < nF:
                    s = acc_time[idx]
                    std = acc_time_std[idx]
                    min_sig = s-std/np.sqrt(1000)
                    max_sig = s+std/np.sqrt(1000)
                    col.plot(time,acc_time[-1] - s,'-',color=colors[idx],linewidth = 2)
                    col.plot(time,(max_sig<min_sig_all)*0.2,'.k')#color=colors[idx])
                    col.title.set_text('\n'+titles[idx]+'\n')
                    col.title.set(fontsize = 22)
                    col.plot(time, np.zeros_like(time),'k',linewidth = 0.5)
                    col.plot([0,0],ylims,'k',linewidth = 0.5)
                    col.plot([500,500],ylims,'k',linewidth = 0.5)
                    col.set_xlim(25*start_time,25*stop_time-25)
                    col.set_ylim(ylims)
                    col.set_xlabel('time w.r.t current word onset (ms)')
                    col.set_ylabel('$\Delta$ accuracy')
                idx+=1

        plt.subplots_adjust(bottom=-0.1, right=1.1, top=1.2,hspace=0.5,wspace = 0.5)
        plt.savefig(save_directory+prefix+'_time.pdf',bbox_inches='tight')

    
    if make_spatial:
        min_sig_all = acc_time_sensor[-1]-acc_time_sensor_std[-1]/np.sqrt(1000)
        max_sig_all = acc_time_sensor[-1]+acc_time_sensor_std[-1]/np.sqrt(1000)
        
        for II in range(nF):
            min_sig_II = acc_time_sensor[II]-acc_time_sensor_std[II]/np.sqrt(1000)
            max_sig_II = acc_time_sensor[II]+acc_time_sensor_std[II]/np.sqrt(1000)
            
            sig1 = acc_time_sensor[-1] - acc_time_sensor[II]
            
            sig2  = max_sig_II<min_sig_all

            sig3 = sig1*sig2

            topoplot(sig3[:,:] ,  nrow = 4, ncol = 5, 
                     time_step = 25,vmin = -0.25, vmax = 0.25,
                    figsize = (15,20),time_start = 0);
            plt.savefig(save_directory+prefix+'_sensor_time_{}.pdf'.format(II),
                        bbox_inches='tight')

    return res

def smooth_run_not_masked(data,smooth_factor):
    smoothed_data = np.zeros_like(data)
    for i,d in enumerate(data):
        smoothed_data[i] = gaussian_filter(data[i], sigma=smooth_factor, order=0, output=None, 
                 mode='reflect', cval=0.0, truncate=4.0)
    return smoothed_data



def load_and_process(file,start_trim = 20, end_trim = 15, do_detrend=True, smoothing_factor = 1,
                     do_zscore = True): 
    
    dat = nibabel.load(file).get_data()
    # very important to transpose otherwise data and brain surface don't match
    dat = dat.T 
    
    #trimming
    if end_trim>0:
        dat = dat[start_trim:-end_trim]
    else: # to avoid empty error when end_trim = 0
        dat = dat[start_trim:]
    
    # detrending
    if do_detrend:
        dat = signal.detrend(np.nan_to_num(dat),axis =0)
    
    # smoothing
    if smoothing_factor>0:
        # need to zscore before smoothing
        dat = np.nan_to_num(zscore(dat))
        dat = smooth_run_not_masked(dat, smoothing_factor)
        
    # zscore
    if do_zscore:
        dat = np.nan_to_num(zscore(dat))
        

    return dat