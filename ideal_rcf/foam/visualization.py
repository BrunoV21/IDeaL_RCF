from ideal_rcf.infrastructure.visualization import PlottingTools
from ideal_rcf.foam.postprocess import extract_U_profiles, ODE_operator
from ideal_rcf.dataloader.caseset import CaseSet

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from typing import Optional, List, Dict
from matplotlib.colors import Normalize
from matplotlib import colormaps
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np




class FoamPlottingTools(PlottingTools):
    def __init__(self,
                 sklearn_metrics_list :Optional[List],
                 exp_id :Optional[Path]=None,
                 img_folder :Optional[Path]=None) -> None:
        
        super().__init__( sklearn_metrics_list, exp_id, img_folder)


    def parity_plots(self,
                     caseset_obj :CaseSet,
                     subplots_components :Optional[List[str]]=['||U||']):
        
        n_rows = 1
        n_cols = 1

        assert n_cols == len(subplots_components), f'subplots_title arg shoudl have n_cols entry but got {len(subplots_components)}'
        
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

        x = np.arange(-1000, 1000)
        y = x
        
        ### create a common colorbar for each pair of subplots
        for j, component in enumerate(subplots_components):
            U_mag_labels = [np.linalg.norm([u, v]) for u,v in zip(caseset_obj.u[:,0], caseset_obj.v[:,0])]

            axs.scatter(U_mag_labels, caseset_obj.predictions_U, edgecolors = (0, 0, 0))   
            axs.plot(x, y, 'r')
            axs.set_xlim([min(U_mag_labels), max(U_mag_labels)])
            axs.set_ylim([min(caseset_obj.predictions_U), max(caseset_obj.predictions_U)])

            axs.set_title(f'{component}', fontsize=20)
            axs.set_xlabel('DNS', fontsize=20)
            
            axs.tick_params(axis = 'both', labelsize = 10)
            axs.text(
                0.12, 
                0.95, 
                f'R2 = {r2_score(U_mag_labels, caseset_obj.predictions_U):.2f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform = axs.transAxes,
                fontsize = 10
            )

            print(f' [{component}]')
            for metric in self.metrics:
                print(f'  > {metric.__name__}: {self.format_float(metric(U_mag_labels, caseset_obj.predictions_U))}')
            
        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_u_parity_plots")

        if self.exp_id:    
            axs.set_ylabel(self.exp_id, fontsize=20)

        plt.tight_layout()
        plt.show(block=False)


    def velocity_plots(self,
                       caseset_obj :CaseSet,
                       velocities :List[str]=['rans', 'predictions', 'dns'],
                       cmap_id :Optional[str]='Spectral_r'):
        
        fig, axs = plt.subplots(len(velocities),1, figsize=(7, 5*len(velocities)),layout='constrained')

        x, y = self.extract_wall_surf(caseset_obj)

        cmap = colormaps[cmap_id]

        for i, attr in enumerate(velocities[::-1]):
            i+=1
            ax = axs[-i]
            if attr == 'dns':
                ax.set_title(attr.upper(), fontsize=20)
                U = [np.linalg.norm([u, v]) for u,v in zip(getattr(caseset_obj,f'u')[:,0], getattr(caseset_obj,f'v')[:,0])]
                vmax = 1.2*max(U)
                vmin = 1.2*min(U)
                norm = Normalize(vmin=vmin, vmax=vmax)
                levels = self.create_levels(vmin, vmax)
            elif attr != 'predictions':
                U = [np.linalg.norm([u, v]) for u,v in zip(getattr(caseset_obj,f'{attr}_u'), getattr(caseset_obj,f'{attr}_v'))]
                ax.set_title(attr.upper(), fontsize=20)
            else:
                U = caseset_obj.predictions_U
                ax.set_title(self.exp_id or 'Predictions', fontsize=20)
                
            cont = ax.tricontourf(
                    caseset_obj.Cx[:,0], 
                    caseset_obj.Cy[:,0], 
                    U, 
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

        if caseset_obj.case[0][:4] == 'PHLL':
            cbar = plt.colorbar(
                cont, 
                ax=axs[-1], 
                format='%0.e', 
                orientation='horizontal', 
                shrink=.8, pad=0.1, 
                ticks = [vmin, vmax])
        else:
            cbar = plt.colorbar(
                cont, 
                ax=axs[-1], 
                format='%0.3f',
                orientation='horizontal', 
                shrink=.8, pad=0.1, 
                ticks = [vmin, vmax])
            
        cbar.ax.tick_params(labelsize=10)

        plt.show(block=False)

        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_U")


    def plot_velocity_profiles(self,
                               caseset_obj :CaseSet,
                               nX :int=99,
                               nY :int=149,
                               velocities :List[str]=['rans', 'predictions', 'dns'],
                               pass_plot_color_dict :Optional[Dict]=None,
                               pass_plot_linestyle_dict :Optional[Dict]=None,
                               pass_plot_location_dict :Optional[Dict]=None):

        color_dict = {
            'rans': 'darkorange',
            'predictions': 'crimson',
            'dns': 'mediumblue'
        }
        if pass_plot_color_dict:
            color_dict.update(pass_plot_color_dict)
    
        linestyle_dict = {
            'rans': '-',
            'predictions': '-',
            'dns': '--'
        }
        if pass_plot_linestyle_dict:
            linestyle_dict.update(pass_plot_linestyle_dict)

        location_dict = {
            'u': 'upper left',
            'v': 'upper right'
        }
        if pass_plot_location_dict:
            location_dict.update(pass_plot_location_dict)

        x, y = self.extract_wall_surf(caseset_obj)

        u_profiles = extract_U_profiles(caseset_obj, nX, nY, velocities).get_profiles()

        for comp in ['u', 'v']:
            fig, ax = plt.subplots(1,1, figsize=(20, 10), num=f"{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_{comp}_profiles")
            plt.subplots_adjust(hspace=0.5)

            if self.exp_id: 
                plt.suptitle(self.exp_id, fontsize=20, y=1.)

            ax.set_ylabel(r'$\dfrac{y}{H}$', fontsize=20, labelpad=8)
            ax.set_xlabel(fr'$\dfrac{{x}}{{H}}+\dfrac{{2}}{{3}}\dfrac{{{comp}}}{{\bar{{U}}}}$', fontsize=20, labelpad=8)
            ax.set_xlim(caseset_obj.Cx.min(), caseset_obj.Cx.max())
            ax.set_ylim(caseset_obj.Cy.min(), caseset_obj.Cy.max())

            for velocity in velocities:
                velocity_profiles = u_profiles[velocity]
                for x_loc, profile in velocity_profiles.items():
                    ax.plot(
                        profile[comp][:,0],
                        profile[comp][:,1],
                        color=color_dict[velocity], 
                        linestyle=linestyle_dict[velocity], 
                        linewidth=4)
                    
            ax.fill_between(
                    x[:,0], 
                    y[:,0],
                    facecolor='lightsteelblue',
                    edgecolor='black',
                    interpolate=True)

            ax.set_aspect(1.5)
    
            handles = [
                plt.Rectangle((0, 0), 0, 0, color=color_dict[source], label=source.upper()) for source in velocities]
            
            ax.legend(handles=handles, fontsize=15, loc = location_dict[comp])
                    
            fig.tight_layout()   
            plt.show(block=False)

            if self.img_folder:
                plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_{comp}_profiles")


    def velocity_abs_error(self,
                           caseset_obj :CaseSet):
        U_DNS_mag = [np.linalg.norm([u, v]) for u,v in zip(getattr(caseset_obj,f'u')[:,0], getattr(caseset_obj,f'v')[:,0])]
        return np.abs(caseset_obj.predictions_U-U_DNS_mag)


    def plot_wall_sheer_stress(self,
                               caseset_obj :CaseSet,
                               velocities :List[str]=['rans', 'predictions', 'dns'],
                               wall :Optional[str]='bottom',
                               pass_plot_color_dict :Optional[Dict]=None,
                               pass_plot_linestyle_dict :Optional[Dict]=None,
                               pass_plot_location_dict :Optional[Dict]=None):

        color_dict = {
            'rans': 'darkorange',
            'predictions': 'crimson',
            'dns': 'mediumblue'
        }
        if pass_plot_color_dict:
            color_dict.update(pass_plot_color_dict)
    
        linestyle_dict = {
            'rans': '-',
            'predictions': '-',
            'dns': '--'
        }
        if pass_plot_linestyle_dict:
            linestyle_dict.update(pass_plot_linestyle_dict)

        location_dict = {
            'u': 'upper left',
            'v': 'upper right'
        }
        if pass_plot_location_dict:
            location_dict.update(pass_plot_location_dict)

        wss_dict = {}
        for profile in velocities:
            scalar = f'{profile}_u' if profile != 'dns' else 'u'
            wss_dict[profile] = ODE_operator(caseset_obj, scalar).extract_WSS()[wall]
        
        fig, axs = plt.subplots(2,1, figsize=(20, 10), num=f"{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_wss_profiles_{wall}_wall")
        plt.subplots_adjust(hspace=0.5)

        if self.exp_id: 
            plt.suptitle(f'{self.exp_id}_WSS_along_{wall}_wall', fontsize=20, y=1.)
        else:
            plt.suptitle(f'WSS_{wall}_wall', fontsize=20, y=1.)

        for ax, clip_start, clip_end, i in zip(axs, [-99, -70], [-1, -25], [0,1]):
            ax.set_ylabel(r'$\nu \dfrac{\partial u}{\partial n}$', fontsize=20, labelpad=8)
            ax.set_xlabel(fr'$\dfrac{{x}}{{H}}$', fontsize=20, labelpad=8)
            ax.set_xlim(caseset_obj.Cx[clip_start:clip_end].min(), caseset_obj.Cx[clip_start:clip_end].max())

            ax.axhline(y=0, xmin=-5, xmax=15, linestyle='--', color='black')

            ax .tick_params(axis = 'both', labelsize = 15) 
            for source, profile in wss_dict.items():
                ax.plot(
                    caseset_obj.Cx[clip_start:clip_end],
                    profile[clip_start:clip_end],
                    color=color_dict[source], 
                    linestyle=linestyle_dict[source], 
                    linewidth=4
                )
                if i == 1 and source != 'RANS':
                    for index, wss in enumerate(profile):
                        if wss < 0:
                            break
                    x_reattachment = caseset_obj.Cx[index]#*0.5+0.5*cX[clip_start:clip_end][index-1]
                    ax.axvline(
                        x=x_reattachment,
                        ymin=-1, ymax=1,
                        linestyle=linestyle_dict[source], 
                        color=color_dict[source],
                        linewidth=2
                    )

        handles = [plt.Rectangle((0, 0), 0, 0, color=color_dict[source], label=source.upper()) for source in wss_dict.keys()]
        axs[0].legend(handles=handles, fontsize=15, loc = 'upper left')
                
        fig.tight_layout()   
        plt.show(block=False)

        if self.img_folder:
            plt.savefig(f"{self.img_foler}/{self.exp_id or ''}_{caseset_obj.set_id or caseset_obj.case[0]}_wss_profiles")



