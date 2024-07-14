
from ideal_rcf.dataloader.caseset import CaseSet

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from typing import Optional, List
from matplotlib.colors import Normalize
from matplotlib import colormaps
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


class PlottingTools(object):
    def __init__(self,
                 sklearn_metrics_list :Optional[List],
                 exp_id :Optional[Path]=None,
                 img_folder :Optional[Path]=None) -> None:
        
        self.metrics = sklearn_metrics_list
        self.img_folder = img_folder
        self.exp_id = exp_id


    def format_float(self,
                     number, 
                     threshold=1e-3):
        """Formats a float to have up to 3 decimal places or scientific notation.

        Args:
            number: The float value to format.
            threshold: The absolute value threshold above which to use scientific notation (default 1000).

        Returns:
            A string representation of the formatted number.
        """
        ### Use f-string with format specifier for 3 decimal places
        if abs(number) >= threshold:
            return f"{number:.3f}"
        else:
            ### Use 'e' for scientific notation with 3 decimal places
            return f"{number:.3e}"


    def parity_plots(self,
                     caseset_obj :CaseSet,
                     attr_list :Optional[List[str]]=['predictions','labels'],
                     subplots_components :Optional[List[str]]=['a_11','a_12','a_22','a_33']):
        
        n_rows = int(len(attr_list)/2)
        n_cols = max([getattr(caseset_obj, attr).shape[1] for attr in attr_list])

        assert n_cols == len(subplots_components), f'subplots_title arg shoudl have n_cols entry but got {len(subplots_components)}'
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

        x = np.arange(-1000, 1000)
        y = x
        
        ### create a common colorbar for each pair of subplots
        for j, component in enumerate(subplots_components):
            axs[j].scatter(caseset_obj.labels[:, j], caseset_obj.predictions[:,j], edgecolors = (0, 0, 0))   
            axs[j].plot(x, y, 'r')
            axs[j].set_xlim([min(caseset_obj.labels[:, j]), max(caseset_obj.labels[:, j])])
            axs[j].set_ylim([min(caseset_obj.predictions[:, j]), max(caseset_obj.predictions[:, j])])

            axs[j].set_title(f'{component}', fontsize=20)
            axs[j].set_xlabel('Labels', fontsize=20)
            
            axs[j].tick_params(axis = 'both', labelsize = 10)
            axs[j].text(
                0.12, 
                0.95, 
                f'R2 = {r2_score(caseset_obj.labels[:, j], caseset_obj.predictions[:,j]):.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform = axs[j].transAxes,
                fontsize = 10
            )

            print(f' [{component}]')
            for metric in self.metrics:
                print(f'  > {metric.__name__}: {self.format_float(metric(caseset_obj.labels[:,j], caseset_obj.predictions[:,j]))}')
            
        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_parity_plots")

        if self.exp_id:    
            axs[0].set_ylabel(self.exp_id, fontsize=20)

        plt.tight_layout()
        plt.show(block=False)


    def extract_wall_surf(self,
                          caseset_obj :CaseSet):
        
        threshold = 0.8*(caseset_obj.Cx.max()-caseset_obj.Cx.min())
        for i in range(1, len(caseset_obj.Cx)):
            if np.abs(caseset_obj.Cx[-i-1]-caseset_obj.Cx[-i])>threshold:
                cutoff = -i
                break
        return caseset_obj.Cx[cutoff:], caseset_obj.Cy[cutoff:]


    def create_levels(self, 
                      val1 :np.array, 
                      val2 :np.array):
        
        return np.linspace(val1, val2, num=500).tolist() 


    def plot_oev(self,
                 caseset_obj :CaseSet,
                 cmap_id:Optional[str]='coolwarm'):
        
        fig = plt.figure(figsize=(7, 3), num=f"{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_oev")
        plt.subplots_adjust(hspace=0.5)
            
        x, y = self.extract_wall_surf(caseset_obj)
            
        ### create a common colorbar for each pair of subplots
        cmap = colormaps[cmap_id]
        
        ax = plt.subplot(1, 1, 1)
                
        if max(caseset_obj.predictions_oev) > .1:   
            vmax =  caseset_obj.predictions_oev.max()
            vmin =  caseset_obj.predictions_oev.min()
        else:
            vmax =  float(0.005)
            vmin =  float(0) 

        norm = Normalize(vmin=vmin, vmax=vmax)
        levels = self.create_levels(vmin, vmax)

        cont = ax.tricontourf(
            caseset_obj.Cx[:,0], 
            caseset_obj.Cy[:,0], 
            caseset_obj.predictions_oev, 
            norm=norm, 
            cmap=cmap, 
            levels=levels,  
            extend='both'
        )
        ax.fill_between(
            x[:,0], 
            y[:,0],
            facecolor='lightsteelblue',
            edgecolor='black',
            interpolate=True)

        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title(caseset_obj.set_id or caseset_obj.case[0], fontsize=20, y=1.)
        ax.set_aspect(1.3)

        ### Add a colorbar to the plot
        cbar = plt.colorbar(
            cont, 
            ax=ax, 
            format='%0.5f', 
            orientation='vertical',
            shrink=.8, 
            pad=0.1, 
            ticks = [vmin, vmax]
        )
        cbar.ax.tick_params(labelsize=10)

        fig.tight_layout()
        plt.show(block=False)

        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_oev")


    def get_plots_error(self,
                        caseset_obj :CaseSet,
                        error_function,
                        cmap_id:Optional[str]='cool'):
        
        fig = plt.figure(figsize=(7, 3), num=f"{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_{error_function.__name__}")
        plt.subplots_adjust(hspace=0.5)
            
        x, y = self.extract_wall_surf(caseset_obj)

        error_metrics = error_function(caseset_obj)
            
        ### create a common colorbar for each pair of subplots
        cmap = colormaps[cmap_id]
        
        ax = plt.subplot(1, 1, 1)
        
        vmax =  error_metrics.max()
        vmin =  error_metrics.min()

        norm = Normalize(vmin=vmin, vmax=vmax)
        levels = self.create_levels(vmin, vmax)

        cont = ax.tricontourf(
            caseset_obj.Cx[:,0], 
            caseset_obj.Cy[:,0], 
            error_metrics, 
            norm=norm, 
            cmap=cmap, 
            levels=levels,  
            extend='both'
        )
        ax.fill_between(
            x[:,0], 
            y[:,0],
            facecolor='lightsteelblue',
            edgecolor='black',
            interpolate=True)

        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_title(caseset_obj.set_id or caseset_obj.case[0], fontsize=20, y=1.)
        ax.set_aspect(1.3)

        ### Add a colorbar to the plot
        cbar = plt.colorbar(
            cont, 
            ax=ax, 
            format='%0.e', 
            orientation='vertical',
            shrink=.5, 
            pad=0.1, 
            ticks = [vmin, vmax]
        )
        cbar.ax.tick_params(labelsize=10)
        fig.tight_layout()   
        plt.show(block=False)
        
        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_{error_function.__name__}")


    def plot_anisotropy(self,
                        caseset_obj :CaseSet,
                        scalar_name :str='anisotropy',
                        attr_list :Optional[List[str]]=['predictions','labels'],
                        subplots_components :Optional[List[str]]=['a_11','a_12','a_22','a_33'],
                        cmap_id :Optional[str]='coolwarm'):
    
        try:
            bool(caseset_obj.labels)
            attr_list = [attr for attr in attr_list if attr != 'labels']
            try:
                bool(caseset_obj.predictions)
                attr_list = [attr for attr in attr_list if attr != 'predictions']
            except ValueError:
                ...
        except ValueError:
            ...
            
        n_rows = int(len(attr_list))
        n_cols = max([getattr(caseset_obj, attr).shape[1] for attr in attr_list])

        assert n_cols == len(subplots_components), f'subplots_title arg shoudl have n_cols entry but got {len(subplots_components)}'
        
        fig = plt.figure(figsize=(7*n_cols, 5*n_rows), num=f"{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_{scalar_name}")

        fig.suptitle(f'{caseset_obj.set_id or caseset_obj.case[0]}', fontsize=20)
        
        x, y = self.extract_wall_surf(caseset_obj)

        cmap = colormaps[cmap_id]

        n_cols_list = [i*n_cols for i in range(n_rows)]

        for i in n_cols_list:
            for j, component in enumerate(subplots_components):
                try:
                    vmax = 1.2*caseset_obj.labels[:,j].max()
                    vmin = 1.2*caseset_obj.labels[:,j].min()
                except TypeError:
                    vmax = 1.2*caseset_obj.predictions[:,j].max()
                    vmin = 1.2*caseset_obj.predictions[:,j].min()
                norm = Normalize(vmin=vmin, vmax=vmax)
                levels = self.create_levels(vmin, vmax)
                ax = plt.subplot(n_rows, n_cols, j+i+1)
                if i == 0:
                    scalar = caseset_obj.predictions[:,j]
                    ax.set_title(component, fontsize=20)
                    if j == 0 and self.exp_id:
                        ax.set_ylabel(self.exp_id, fontsize=20, rotation=90, labelpad=8)
                else:
                    scalar = caseset_obj.labels[:,j]
                    if j == 0:
                        ax.set_ylabel('Labels', fontsize=20, rotation=90, labelpad=8)

                cont = ax.tricontourf(
                    caseset_obj.Cx[:,0], 
                    caseset_obj.Cy[:,0], 
                    scalar, 
                    norm=norm, 
                    cmap=cmap, 
                    levels=levels,  
                    extend='both'
                )
                ax.fill_between(
                    x[:,0], 
                    y[:,0],
                    facecolor='lightsteelblue',
                    edgecolor='black',
                    interpolate=True)

                ax.xaxis.set_ticklabels([])
                ax.axes.yaxis.set_ticklabels([])
                
                ax.set_aspect(1.3)

                if i == n_cols_list[-1]:
                    ### Add a colorbar to the plot
                    if caseset_obj.case[0][:4] == 'PHLL':
                        cbar = plt.colorbar(
                            cont, 
                            ax=ax, 
                            format='%0.e', 
                            orientation='horizontal', 
                            shrink=.8, pad=0.1, 
                            ticks = [vmin, vmax])
                    else:
                        cbar = plt.colorbar(
                            cont, 
                            ax=ax, 
                            format='%0.3f',
                            orientation='horizontal', 
                            shrink=.8, pad=0.1, 
                            ticks = [vmin, vmax])
                        
                    cbar.ax.tick_params(labelsize=10)
                
        fig.tight_layout()   
        plt.show(block=False)

        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_{scalar_name}")