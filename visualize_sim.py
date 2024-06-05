import seaborn as sns

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 18}
import matplotlib
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
import numpy as np


def box_plot(ax, data, edge_color, fill_color, positions=None):
    bp = ax.boxplot(data, patch_artist=True)
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color) 
        
    return bp


def sim_plots(Results,name,settings,filename='',var_dict={}, ylim0 = [-0.1, 1],
             ylim1=[0,0.5],ylim2=[0,0.5]): # shift=0.2,:
    
    import math

    var = sorted(list(set(Results[name].tolist())))
    print(var)
    
    stack_avg = []
    concat_avg = []
    max_avg = []
    varpar_X0_concat = []
    varpar_X0_stacked = []
    varpar_X0_real = []

    
    for px in var:
        stack_avg.append(Results[Results[name]==px]['stacked'])
        concat_avg.append(Results[Results[name]==px]['concat'])
        max_avg.append(Results[Results[name]==px]['max'])
        varpar_X0_real.append(Results[Results[name]==px]['varpar_X0_real'])
        varpar_X0_concat.append(Results[Results[name]==px]['varpar_X0_concat'])
        varpar_X0_stacked.append(Results[Results[name]==px]['varpar_X0_stacked'])
    
    c_fill = 'white'
    pos = None 

    fig, axs = plt.subplots(1, 2,figsize=(20,5))
    
    
    varpar_X0_concat = np.array(varpar_X0_concat)
    varpar_X0_stacked = np.array(varpar_X0_stacked)
    varpar_X0_real = np.array(varpar_X0_real)
    
    import pandas as pd
    vec = [[np.abs(varpar_X0_stacked[k,i] - varpar_X0_real[k,i]),k,'stack'] for i in range(varpar_X0_stacked.shape[1])  
           for k in range(varpar_X0_stacked.shape[0])]
    vec_concat = [[np.abs(varpar_X0_concat[k,i] - varpar_X0_real[k,i]),k,'concat'] for i in range(varpar_X0_concat.shape[1]) 
                  for k in range(varpar_X0_concat.shape[0])]
    r2_results = pd.concat([pd.DataFrame(vec,columns = ['Err','setting','method']),
                            pd.DataFrame(vec_concat,columns = ['Err','setting','method'])])
   
    sns.boxplot(ax=axs[1], x="setting", y="Err", hue="method",data=r2_results, palette="Set2")
    sns.move_legend(axs[1], "upper left", bbox_to_anchor=(1, 1))


    axs[1].set_ylim(ylim1)
    if name=='noise':
        axs[1].set_xlabel(r'$\sigma$')
    if name=='Feat_Dim_ratio':
        axs[1].set_xlabel(r'$d_1$')
    if name=='Cond':
        axs[1].set_xlabel(r'$\alpha_1$')
    if name=='Sample_Dim_ratio':
        axs[1].set_xlabel(r'$n$')
    
    
    axs[1].set_ylabel(r'Var. par. error for ${\bf x}_1$')
    axs[1].set_ylim(ylim2)
    axs[1].set_xticks(range(len(settings)))
    axs[1].set_xticklabels(settings,rotation=45)

    from scipy.stats import ttest_rel
    
    for idx,s in enumerate(settings):
        setting_res = r2_results[r2_results['setting']==idx]
        stack_res = [e for e in setting_res[setting_res['method']=='stack']['Err']]
        concat_res = [e for e in setting_res[setting_res['method']=='concat']['Err']]
        for i in np.arange(len(stack_res)):
            axs[1].plot([idx-0.22,idx+0.22], [stack_res[i],concat_res[i]],'-',linewidth=0.25,color='black')
        tstat,pval = ttest_rel( np.array(concat_res), np.array(stack_res),alternative='two-sided')
        if pval<0.05/20:
            if tstat>0:
                axs[1].plot(idx,0.45,'*',color = 'black',markersize=10)
            else:
                axs[1].plot(idx,0.45,'o',color = 'black',markersize=10)
    # blabla
    
    stack_avg = np.array(stack_avg)
    max_avg = np.array(max_avg)
    concat_avg = np.array(concat_avg)
    
    import pandas as pd
    vec = [[stack_avg[k,i] ,k,'stack'] for i in range(stack_avg.shape[1])  
           for k in range(stack_avg.shape[0])]
    vec_concat = [[concat_avg[k,i] ,k,'concat'] for i in range(concat_avg.shape[1]) 
                  for k in range(concat_avg.shape[0])]
    vec_max = [[max_avg[k,i] ,k,'max'] for i in range(max_avg.shape[1]) 
               for k in range(max_avg.shape[0])]
    r2_results = pd.concat([pd.DataFrame(vec,columns = ['R2','setting','method']),
                                         pd.DataFrame(vec_concat,columns = ['R2','setting','method']),
                                         pd.DataFrame(vec_max,columns = ['R2','setting','method'])])
   
    sns.boxplot(ax=axs[0], x="setting", y="R2", hue="method",data=r2_results, palette="hot_r")
    sns.move_legend(axs[0], "upper right", bbox_to_anchor=(-0.18, 1))
    
    mean_stack = np.mean(stack_avg)
    if name=='noise':
        axs[0].set_xlabel(r'$\sigma$')
    if name=='Feat_Dim_ratio':
        axs[0].set_xlabel(r'$d_1$')
    if name=='Cond':
        axs[0].set_xlabel(r'$\alpha_1$')
    if name=='Sample_Dim_ratio':
        axs[0].set_xlabel(r'$n$')
    if name=='correl':
        axs[0].set_xlabel(r'$\rho$')

    for idx,s in enumerate(settings):
        for i in np.arange(stack_avg.shape[1]):
            axs[0].plot([idx-0.25,idx,idx+0.25], [stack_avg[idx,i],concat_avg[idx,i],max_avg[idx,i]],
                        '-',linewidth=0.15,color='black')
        # bla
    
    axs[0].set_ylabel(r'$R^2$')
    axs[0].set_ylim(ylim0)
    axs[0].set_xticks(range(len(settings)))
    axs[0].set_xticklabels(settings,rotation=45)
    
    


    if name=='noise':
        plt.suptitle(r'Vary $\sigma:~\alpha = {},~d={},~n={}$'.format(var_dict['alphas'],var_dict['ds'],var_dict['n']))
    if name=='Feat_Dim_ratio':
        plt.suptitle(r'Vary $d_1:~\sigma = {},~\alpha = {},~d_2+d_3+d_4={},~n={}$'.format(var_dict['sigma'], var_dict['alphas'], var_dict['d_sum'], var_dict['n']))
    if name=='Cond':
        plt.suptitle(r'Vary $\alpha_1:~\sigma = {},~d={},~n={}$'.format(var_dict['sigma'], var_dict['ds'],var_dict['n']))
    if name=='Sample_Dim_ratio':
        plt.suptitle(r'Vary $n:~\sigma = {},~\alpha = {},~d={}$'.format(var_dict['sigma'],var_dict['alpha'],var_dict['ds']))
    if name=='correl':
        plt.suptitle(r'Vary $\rho:~\sigma = {},~\alpha = {},~d={}, ~n={}$'.format(var_dict['sigma'],var_dict['alpha'],var_dict['ds'],
                                                                                 var_dict['n']))

    
    
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)
    plt.tight_layout()
    plt.savefig(filename+timestr+'.jpg')
    plt.show()    

