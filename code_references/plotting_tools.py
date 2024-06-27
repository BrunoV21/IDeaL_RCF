from ipynb.fs.full. processing_tools import *

import matplotlib.pyplot as plt
import matplotlib.cm
import seaborn as sns
from scipy.stats import kde
from matplotlib.colors import Normalize
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import numpy as np

### Functions for plotting
def plot_graphs1(var1, var2, string, metrics,ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    metrics[[var1, var2]].plot(ax=ax)
    
    ax.set_title(' ', fontsize=40)
    ax.set_xlabel('Number of epochs', fontsize=40)
    ax.set_ylabel(string, fontsize=40)
    ax.legend(labels=[var1, var2], fontsize=30)
    ax.set_xlim(0, len(metrics))
    if string == 'Loss and LR':
        ax.set_yscale('log')
    else :     
        ax.set_ylim(0, 2*np.mean(metrics[var2]))
    return ax



tol_dict = {'PHLL': 7e-2,
            'BUMP': 1.8e-3,
            'CNDV': 3e-2,
            'CBFS': 9.5e-2
            }   


def spatial_average(Cx, tol): 
    for i in range(1, len(Cx)-1):
        if Cx[i+1] - Cx[i] <= tol:
            Cx[i+1] = Cx[i]
            #print(1)
         
    return Cx


def extract_wall_surf(Cx, Cy, test):

    x_sort_index = np.argsort(Cx)

    Cx_sorted = np.take_along_axis(Cx, x_sort_index, axis = 0)

    Cy_sorted = np.take_along_axis(Cy, x_sort_index, axis = 0)

    Cx_around = spatial_average(Cx_sorted, tol_dict[test[:4]])


    Cx_unique =  np.unique(Cx_around, return_index=False)


    avg_spacing = (Cx_unique[0] + Cx_unique[1])/len(Cx_unique)



    y = np.zeros(len(Cx_unique))
    i = 0

    for x in Cx_unique:
        y[i] = np.min(Cy_sorted[np.where(Cx_around == x)[0]])
        #print(y[i])
        i += 1

    #plt.scatter(Cx_unique, y)

    return Cx_unique, y

        
def create_levels(val1, val2):
    return np.linspace(val1, val2, num=500).tolist()  


### this functions can be moved to file tbh
def get_parity_plots(pred, DNS, id, current_file, metrics_file, label = ['labels_a_11', 'labels_a_12', 'labels_a_22', 'labels_a_33'] ):
      
    fig, axs = plt.subplots(1,4, figsize=(40, 10))
        
    #plt.subplots_adjust(wspace=0.25)
    #fig.suptitle(id, fontsize=40, y=1.)
    n = 1
    j = 0
    x = np.arange(-1000, 1000)
    y = x
       
    # create a common colorbar for each pair of subplots
    
    for labe in label:
        #ax = plt.subplot(pred.shape[1], 4, n)
        axs[j].scatter(DNS[:, j], pred[:,j], edgecolors = (0, 0, 0))   
        axs[j].plot(x, y, 'r')
        axs[j].set_xlim([min(DNS[:, j]), max(DNS[:, j])])
        axs[j].set_ylim([min(pred[:, j]), max(pred[:, j])])
        #if len(label) == 1:
        #ax.set_aspect('equal', 'box')
        axs[j].set_title(f'{labe[-4:]}', fontsize=40)
        axs[j].set_xlabel('Labels', fontsize=40)
        
        axs[j].tick_params(axis = 'both', labelsize = 20)
        axs[j].text(0.12, 0.95, f'R2 = {r2_score(DNS[:, j], pred[:,j]):.2f}',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = axs[j].transAxes,
                   fontsize = 25)
    

        print(f'\n====== {labe} ======')
        print(f'> mean absolute error = {mean_absolute_error(DNS[:, j], pred[:,j])}')
        print(f'> mean squared error = {mean_squared_error(DNS[:, j], pred[:,j])}')
        print(f'> mean r2 score = {r2_score(DNS[:, j], pred[:,j]):.3f}')
        
        metrics_file += f'\n====== {labe} ======\n> MSE = {mean_absolute_error(DNS[:, j], pred[:,j])}\n> MAE = {mean_squared_error(DNS[:, j], pred[:,j])}\n> R2 = {r2_score(DNS[:, j], pred[:,j])}\n'
        
        #wandb.log({f'{test} mean absolute error': mean_absolute_error(pred[:,j], DNS[:, j]), 
         #          f'{test} mean squared error': mean_squared_error(pred[:,j], DNS[:, j]),
          #         f'{test} mean r2 score': r2_score(pred[:,j], DNS[:, j])})
                              
        j  += 1
        n  += 1      
        
    axs[0].set_ylabel(current_file, fontsize=40)
    plt.tight_layout()

    plt.savefig(f'imgs\\{current_file}_Parity_Plot_{id}')
    
    return metrics_file    
    

    
def get_plots(pred, DNS, cX, cY, test, current_file, id, label = ['a_11', 'a_12', 'a_22', 'a_33']):
    fig = plt.figure(figsize=(60, 20))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(id, fontsize=60, y=1.)
    #plt.subplots_adjust(hspace=3)
         
    x, y = extract_wall_surf(cX, cY, test)    
        
    # create a common colorbar for each pair of subplots
    cmap = matplotlib.cm.get_cmap('coolwarm')
    #cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.subplot(111))
    #cb.set_label('Colorbar', fontsize=14)º
    #levels =[-0.7,-0.35, 0, 0.35, 0.7]
    for n in [0, 4]:
        for j in range(len(label)):
            vmax =  1.2*DNS[:, j].max()#max(max(pred[:, j]),)
            vmin =  1.2*DNS[:, j].min()#min(min(pred[:, j]), ) 
            #vmax = DNS[:,j].max()
            #vmin = DNS[:,j].min()
            norm = Normalize(vmin=vmin, vmax=vmax)
            levels = create_levels(vmin, vmax)
            #ticks = [0.8*vmin, 0.8*(-vmin + vmax)*0.5, 0.8vmax]
            #if labe != 'labels_b_13' and labe != 'labels_b_23':
            # add a new subplot iteratively
            ax = plt.subplot(2, 4, j+1+n)
            
            # filter df and plot ticker on the new subplot axis
            if n == 0:
                _plt =  pred[:,j]
                ax.set_title(label[j], fontsize=60)
                if j == 0:
                    ax.set_ylabel(current_file, fontsize=60, rotation=90, labelpad=8)
            else:
                _plt = DNS[:,j]
                if j == 0:
                    ax.set_ylabel('Labels', fontsize=60, rotation=90, labelpad=8)
               
                
                        
            cont = ax.tricontourf(cX, cY, _plt, norm = norm, cmap = cmap, levels = levels,  extend = 'both')
            ax.fill_between(x, y, facecolor = 'lightsteelblue', edgecolor = 'black', interpolate = True)
            
            ax.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            
            ax.set_aspect(1.3)
            if n != 0:
             # Add a colorbar to the plot
                if id[:4] == 'PHLL':
                    cbar = plt.colorbar(cont, ax=ax, format='%0.e', orientation='horizontal', shrink=.8, pad=0.1, ticks = [vmin, vmax])
                else:
                    cbar = plt.colorbar(cont, ax=ax, format='%0.3f', orientation='horizontal', shrink=.8, pad=0.1, ticks = [vmin, vmax])
                cbar.ax.tick_params(labelsize=30)
            
    fig.tight_layout()   
    plt.savefig(f'imgs\\{current_file}_anisotropy_plots_{id}')
                         

def get_plots_error(metric, cX, cY, test, current_file, id, label = ['a']):
    fig = plt.figure(figsize=(10, 3))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(id, fontsize=15, y=1.)
    #plt.subplots_adjust(hspace=3)
         
    x, y = extract_wall_surf(cX, cY, test)    
        
    # create a common colorbar for each pair of subplots
    #if id[-2] == 'R': 
    #    cmap = matplotlib.cm.get_cmap('cool_r')
    #else :
    # create a common colorbar for each pair of subplots
    cmap = matplotlib.cm.get_cmap('cool')
    #cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.subplot(111))
    #cb.set_label('Colorbar', fontsize=14)º
    #levels =[-0.7,-0.35, 0, 0.35, 0.7]
    n = 0
    for j in range(len(label)):
        vmax =  1*metric.max()#max(max(pred[:, j]),)
        vmin =  1*metric.min()#min(min(pred[:, j]), ) 
        #vmax = DNS[:,j].max()
        #vmin = DNS[:,j].min()
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)#Normalize(vmin=vmin, vmax=vmax)
        levels = create_levels(vmin, vmax)
        #ticks = [0.8*vmin, 0.8*(-vmin + vmax)*0.5, 0.8vmax]
        #if labe != 'labels_b_13' and labe != 'labels_b_23':
        # add a new subplot iteratively
        ax = plt.subplot(1, 1, 1)

        # filter df and plot ticker on the new subplot axis

        #ax.set_title(label[j], fontsize=30)
        # if j == 0:
        #     ax.set_ylabel('TBNN_3b_ff', fontsize=60, rotation=90, labelpad=8)


        cont = ax.tricontourf(cX, cY, metric, norm = norm, cmap = cmap, levels = levels, extend = 'both')
        ax.fill_between(x, y, facecolor = 'lightsteelblue', edgecolor = 'black', interpolate = True)

        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        ax.set_aspect(1.3)
        
    # if id[-2] == 'R2':
    #     cbar = plt.colorbar(cont, ax=ax, format='%0.f', orientation='vertical', shrink=.5, pad=0.1, ticks = [vmin, vmax])
    # else:
    cbar = plt.colorbar(cont, ax=ax, format='%0.e', orientation='vertical', shrink=.5, pad=0.1, ticks = [vmin, vmax])

    cbar.ax.tick_params(labelsize=15)
    
    print(f'> sucessfuly plotted {id}\n')
            
    fig.tight_layout()   
    plt.savefig(f'imgs\\{current_file}_relative_error_distribution_{id}')


def pedict_parity(pred_dict, case_dict, model, cases, feat_list, remove_outliers_id, shuffling, scaler, labels_scaler_NL, labels_scaler, labels_scaler_eV, metrics_file, current_file, id):
        
    for case in cases:
        

        if type(model) == list:
            _labels_1 = data_clean_case(case_dict, [case], feat_list, remove_outliers_id, shuffling, scaler[1], labels_scaler[1])[0] 
            _labels_2 = data_clean_case(case_dict, [case], feat_list, remove_outliers_id, shuffling, scaler[2], labels_scaler[2])[0] 
            print(f'> averaging out results from the best two models')
            pred_dict[case] = 0.5*labels_scaler[1].inverse_transform(model[1]([_labels_1[0], _labels_1[1], _labels_1[2]])) + 0.5*labels_scaler[2].inverse_transform(model[2]([_labels_2[0], _labels_2[1], _labels_2[2]]))
            
            metrics_file += f'\n\n====== {case} - {id}  ======'

            #force realizability
            print('\n==================== Ensuring Realizability ====================')
            pred_dict[case] = force_realizability(pred_dict[case])
            _labels_1[4] = force_realizability(labels_scaler[1].inverse_transform(_labels_1[4]))

            metrics_file += get_parity_plots(pred_dict[case],_labels_1[4],  case, current_file, metrics_file) 
            
            relative_error_percent = {case: np.array([np.linalg.norm(pred_dict[case][i]-_labels_1[4][i]) / np.linalg.norm(_labels_1[4][i]) for i in range(_labels_1[4].shape[0]) ])                       
                          for case in [case]}
            
            get_plots_error(relative_error_percent[case], _labels_1[-2], _labels_1[-1], case, current_file, f'{case}')

            get_plots(pred_dict[case], _labels_1[4], _labels_1[-2], _labels_1[-1], case, current_file, case)
            
            print(f'\n====== {case} Overall Metrics ======')
            print(f'> mean absolute error = {mean_absolute_error(_labels_1[4], pred_dict[case])}')
            print(f'> mean squared error = {mean_squared_error(_labels_1[4], pred_dict[case])}')
            print(f'> r2 score = {r2_score(_labels_1[4], pred_dict[case]):.3f}')
            print('\n\n\n')
            metrics_file += f'\n====== {id} Overall Metrics ======\n> MAE = {mean_absolute_error(_labels_1[4], pred_dict[case])}\n> MSE = {mean_squared_error(_labels_1[4], pred_dict[case])}\n> R2 = {r2_score(_labels_1[4], pred_dict[case])}'
            
            
            
        else:
            _labels = data_clean_case(case_dict, [case], feat_list, remove_outliers_id, shuffling, scaler, labels_scaler_NL, labels_scaler,labels_scaler_eV)[0] 
            pred_dict[case] = labels_scaler.inverse_transform(model([_labels[0], _labels[1], _labels[2]]))
           

            metrics_file += f'\n\n====== {case} - {id}  ======'

            #force realizability
            print('\n==================== Ensuring Realizability ====================')
            pred_dict[case] = force_realizability(pred_dict[case])
            _labels[4] = force_realizability(labels_scaler.inverse_transform(_labels[4]))

            metrics_file += get_parity_plots(pred_dict[case],_labels[4],  case, current_file, metrics_file) 
            
            relative_error_percent = {case: np.array([np.linalg.norm(pred_dict[case][i]-_labels[4][i]) / np.linalg.norm(_labels[4][i]) for i in range(_labels[4].shape[0]) ])                       
                          for case in [case]}
            
            get_plots_error(relative_error_percent[case], _labels[-2], _labels[-1], case, current_file, f'{case}')

            get_plots(pred_dict[case], _labels[4], _labels[-2], _labels[-1], case, current_file, case)

            print(f'\n====== {case} a_NL Overall Metrics ======')
            print(f'> mean absolute error = {mean_absolute_error(_labels[4], pred_dict[case])}')
            print(f'> mean squared error = {mean_squared_error(_labels[4], pred_dict[case])}')
            print(f'> r2 score = {r2_score(_labels[4], pred_dict[case]):.3f}')
            print('\n\n\n')
      
            metrics_file += f'\n====== {id} Overall Metrics ======\n> MAE = {mean_absolute_error(_labels[4], pred_dict[case])}\n> MSE = {mean_squared_error(_labels[4], pred_dict[case])}\n> R2 = {r2_score(_labels[4], pred_dict[case])}'
    
    return metrics_file


def get_plots_eV(pred, eV, cX, cY, test, current_file, label = ['a_11', 'a_12', 'a_22', 'eV']):
    fig = plt.figure(figsize=(60, 13))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(test, fontsize=60, y=1.)
    #plt.subplots_adjust(hspace=3)
         
    x, y = extract_wall_surf(cX, cY, test)    
        
    # create a common colorbar for each pair of subplots
    cmap = matplotlib.cm.get_cmap('coolwarm')
    #cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.subplot(111))
    #cb.set_label('Colorbar', fontsize=14)º
    #levels =[-0.7,-0.35, 0, 0.35, 0.7]
    for n in [0]:
        for j in range(len(label)):
            ax = plt.subplot(1, 4, j+1+n)
            
            # filter df and plot ticker on the new subplot axis
            if n == 0:
                if j == 3:
                    _plt = eV
                else:
                    _plt =  pred[:,j]
                ax.set_title(label[j], fontsize=60)
                if j == 0:
                    ax.set_ylabel('eVTBNN_3b_ff', fontsize=60, rotation=90, labelpad=8)
           
            vmax =  1.2*max(_plt)
            vmin =  1.2*min(_plt) if min(_plt) < 0 else 0.8*min(_plt) 
            
            norm = Normalize(vmin=vmin, vmax=vmax)
            levels = create_levels(vmin, vmax)
            
            cont = ax.tricontourf(cX, cY, _plt, norm = norm, cmap = cmap, levels = levels,  extend = 'both')
            ax.fill_between(x, y, facecolor = 'lightsteelblue', edgecolor = 'black', interpolate = True)
            
            ax.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            
            ax.set_aspect(1.3)

         # Add a colorbar to the plot
           
            cbar = plt.colorbar(cont, ax=ax, format='%0.3f', orientation='horizontal', shrink=.8, pad=0.1, ticks = [vmin, vmax])
            cbar.ax.tick_params(labelsize=30)
            
    fig.tight_layout()   
    plt.savefig(f'imgs\\{current_file}_eV_plots_{test}')
    
    
def get_plot_eV_hist(eV, current_file, set_):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    sns.histplot(eV, ax = axes, bins = 300, stat = 'count', element = 'bars', kde = False, log_scale = [False, False])
    axes.set_title(f'eVTBNN_3b_ff', fontsize=20)
    axes.tick_params(axis = 'both', labelsize = 16)
    axes.set_ylabel('Count', fontsize=16)
    fig.tight_layout()   
    plt.savefig(f'imgs\\{current_file}_eV_hist_{set_}')
    
    
    
def plot_eV(pred_dict, case_dict, model, cases, feat_list, remove_outliers_id, shuffling, scaler, labels_scaler_NL, labels_scaler, current_file):
        
    for case in cases:
        if type(model) == list:
            _labels_1 = data_clean_case(case_dict, [case], feat_list, remove_outliers_id, shuffling, scaler[1], labels_scaler[1])[0] 
            _labels_2 = data_clean_case(case_dict, [case], feat_list, remove_outliers_id, shuffling, scaler[2], labels_scaler[2])[0] 
            print(f'> averaging out results from the best two models')
            pred_dict[case] = 0.5*model[1](_labels_1[0]) + 0.5*model[2](_labels_2[0])
            _labels_1[4] = labels_scaler[1].inverse_transform(_labels_1[4])            
            _labels = _labels_1 #for plotting coordinates haha
            
        else:
            _labels = data_clean_case(case_dict, [case], feat_list, remove_outliers_id, shuffling, scaler, labels_scaler_NL, labels_scaler)[0] 
            pred_dict[case] = model([_labels[0], _labels[5]])
            
        pred_dict[case] = pred_dict[case].numpy().reshape(1,-1)[0]

        
        print(f'> {case}:\n \max: {pred_dict[case].max()}\n \mean: {pred_dict[case].mean()}\n \min: {pred_dict[case].min()}\n')
        
        
        get_plot_eV_hist(pred_dict[case], current_file, case)
        
        get_plots_eV(np.transpose([-pred_dict[case]*case_dict[case][2][:,i] for i in range(3)]), pred_dict[case], _labels[-2], _labels[-1], case, current_file)
        
def eV_plot(ev, eV_labels, cX, cY, case, current_file):
    fig, axs = plt.subplots(2,1, figsize=(10, 12))
    
    plt.suptitle(f'{current_file}_eV', fontsize=20, y=0.95)
    #plt.suptitle('OeVTBNN', fontsize=20, y=1.)
    #plt.subplots_adjust(hspace=3)
         
    x, y = extract_wall_surf(cX, cY, case)    
        
    # create a common colorbar for each pair of subplots
    cmap = matplotlib.colormaps['coolwarm']
    
    
            
    # filter df and plot ticker on the new subplot axis
    
    #ax.set_title('eV', fontsize=20)
        
    #ax.set_ylabel('eVTBNN_3b_ff', fontsize=20, rotation=90, labelpad=8)
    if max(ev) > .1:   
        vmax =  float(max(ev))
        vmin =  float(min(ev)) 
    else:
        vmax =  float(0.005)
        vmin =  float(0) 

    norm = Normalize(vmin=vmin, vmax=vmax)
    levels = create_levels(vmin, vmax)

    for ax, eV, lab in zip(axs, [ev, eV_labels[:,0]], [current_file, 'Labels']):
        cont = ax.tricontourf(cX, cY, eV, norm = norm, cmap = cmap, levels = levels,  extend = 'both')
        ax.fill_between(x, y, facecolor = 'lightsteelblue', edgecolor = 'black', interpolate = True)
        ax.set_ylabel(lab,fontsize=20,rotation=90, labelpad=8)
        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_aspect(1.3)

    cbar = plt.colorbar(cont, ax=ax, format='%0.5f', orientation='horizontal', shrink=.8, pad=0.1, ticks = [vmin, vmax])
    cbar.ax.tick_params(labelsize=10)
    
    fig.tight_layout()   
    
    plt.savefig(f'imgs\\{current_file}_eff_visc_{case}')
    
        
### plots for gm clustering    
def get_plots_metrics_gm(label, gm_dict, cX, cY):
    if len(label) == 1:
        fig = plt.figure(figsize=(25,4))
    else:    
        fig = plt.figure(figsize=(20, 80))
    plt.subplots_adjust(hspace=1)
    #plt.suptitle('Bayesian Gaussian Mixture Clustering', fontsize=18, y=0.95)
    n = 0
    #j = 0
       
    
    for labe in label:
        #cX, cY = load_coordinates([labe], 'komegaSST')
        x, y = extract_wall_surf(cX[labe], cY[labe], labe)  
        
        #b = loadLabels([labe], 'b')
        #b = np.delete(b.reshape((len(b),9)),[2, 3 , 5, 6, 7],axis=1)
        
        ax = plt.subplot(len(gm_dict), 2, n+1)
        # filter df and plot ticker on the new subplot axis
        cont = ax.tricontourf(cX[labe], cY[labe], gm_dict[labe])
        ax.fill_between(x, y, facecolor = 'lightsteelblue', edgecolor = 'black', interpolate = True)
        ax.set_title(labe)
        #ax.set_aspect(1)
        
        #plt.colorbar(cont, ax=ax)
        #fig.tight_layout()
            
        n  += 1   
 
    
    
def get_plots_metrics_spat(metric, label, cX, cY, test, id):
    if len(label) == 1:
        plt.figure(figsize=(25,4))
    else:    
        plt.figure(figsize=(20, 12))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(id, fontsize=18, y=0.95)
    n = 0
    #j = 0
       
    x, y = extract_wall_surf(cX, cY, test)    
    if id[-2] == 'R': 
        cmap = matplotlib.cm.get_cmap('cool_r')
    else :
    # create a common colorbar for each pair of subplots
        cmap = matplotlib.cm.get_cmap('cool')
    #cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.subplot(111))
    #cb.set_label('Colorbar', fontsize=14)º
    #levels =[-0.7,-0.35, 0, 0.35, 0.7]
    for labe in label:
        vmax = metric.max()
        vmin = metric.min()
        norm = Normalize(vmin=vmin, vmax=vmax)
        levels = create_levels(vmin, vmax)
        #if labe != 'labels_b_13' and labe != 'labels_b_23':
        # add a new subplot iteratively
        ax = plt.subplot(metric.shape[1], 2, n+1)
        # filter df and plot ticker on the new subplot axis
        cont = ax.tricontourf(cX, cY, metric[:,n], norm = norm, cmap = cmap, levels = levels, extend = 'both')
        ax.fill_between(x, y, facecolor = 'lightsteelblue', edgecolor = 'black', interpolate = True)
        ax.set_title(labe)
        ax.set_aspect(1)
        
        plt.colorbar(cont, ax=ax)
      
        n  += 1   

    
    #### Calculate Error metrics and distribuitions
def generate_spatial_distr_metrics(R_labels_dict, R_pred_dict, test_case, val):
    R2 = {}
    RMSE = {}
    MAE = {}    

        
    for test in test_case:
        
        R2_means = np.transpose(np.array([np.resize(np.mean(R_labels_dict[test][:,i]), len(R_labels_dict[test])) for i in range(4)]))
        R2[test] = 1 - (R_labels_dict[test] - R_pred_dict[test])/(R_labels_dict[test] - R2_means)
        
        R2[test][R2[test]>=1] = 1
        R2[test][R2[test]<=0] = 0
        
        RMSE[test] = np.abs(R_labels_dict[test] - R_pred_dict[test])
        
        MAE[test] = R_labels_dict[test] - R_pred_dict[test]
        
            
    for val_a in val:
        
        R2_means = np.transpose(np.array([np.resize(np.mean(R_labels_dict[val_a][:,i]), len(R_labels_dict[val_a])) for i in range(4)]))
        R2[val_a] = 1 - (R_labels_dict[val_a] - R_pred_dict[val_a])/(R_labels_dict[val_a] - R2_means)
        
        R2[val_a][R2[val_a]>=1] = 1
        R2[val_a][R2[val_a]<=0] = 0
                
        RMSE[val_a] = np.abs(R_labels_dict[val_a] - R_pred_dict[val_a])
        
        MAE[val_a] = R_labels_dict[val_a] - R_pred_dict[val_a]
                
    return R2, RMSE, MAE


#### Metric Histographs_plot

def generate_metrics_hist(RMSE_dict, label, set_):
    fig, axes = plt.subplots(2, 2, figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle(f'{set_}', fontsize=18, y=0.95)
    x = 0
    y = 0
    for i in range(RMSE_dict.shape[1]):
        sns.histplot(RMSE_dict[:,i], ax = axes[x, y], stat = 'count', element = 'bars', kde = False, log_scale = [False, True])
        axes[x, y].set_title(f'{label[i]}')
        y += 1
        if y == 2:
            y = 0
            x += 1
            
            
            
            