### calculating metrics 
### receives from framework, inference results and labels both stored in casesets
### receives list of sklearn metrics to calculate
from ideal_rcf.dataloader.caseset import CaseSet
from ideal_rcf.infrastructure.visualization import PlottingTools

from typing import Optional, List
from pathlib import Path
import numpy as np

class Evaluator(PlottingTools):
    """
    A class for evaluating predictions using specified sklearn metrics and generating diagnostic plots.

    Inherits from PlottingTools.

    Attributes:
    - `sklearn_metrics_list`: Optional list of sklearn metrics to use for evaluation.
    - `exp_id`: Optional path representing the experiment identifier.
    - `img_folder`: Optional path to the folder for saving generated images.

    Methods:
    - `__init__(sklearn_metrics_list=None, exp_id=None, img_folder=None) -> None`: 
      Initializes the Evaluator instance by inheriting metrics list, experiment identifier, and image folder from PlottingTools.

    - `calculate_metrics(caseset_obj, show_plots=True, dump_metrics=False)`: 
      Calculates evaluation metrics for the provided CaseSet object.
      Displays metrics values and optionally saves them if `dump_metrics` is True.
      Generates diagnostic plots if `show_plots` is True.

    - `relative_error(caseset_obj)`: 
      Computes the relative error between labels and predictions in the provided CaseSet object.
      Raises ValueError if labels or predictions are missing.

    - Other plotting methods inherited from PlottingTools: 
      `parity_plots`, `plot_oev`, `plot_anisotropy`, `get_plots_error`.

    """
    def __init__(self, 
                 sklearn_metrics_list :Optional[List]=None,
                 exp_id :Optional[Path]=None,
                 img_folder :Optional[Path]=None) -> None: 
        
        super().__init__(sklearn_metrics_list, exp_id, img_folder)


    def calculate_metrics(self,
                          caseset_obj :CaseSet,
                          show_plots :Optional[bool]=True,
                          dump_metrics :Optional[bool]=False):
        
        if self.metrics:
            print(f'[{caseset_obj.set_id or caseset_obj.case[0]}] metrics')
            try:
                bool(caseset_obj.predictions)
                raise ValueError('Make sure to run inference before passing the caseset_obj')
            except ValueError:
                ...

            results = []
            for metric in self.metrics:
                result = metric(caseset_obj.labels, caseset_obj.predictions)
                print(f' > {metric.__name__}: {self.format_float(result)}')
                results.append(result) if dump_metrics else ...
        else:
            print('[warning] you must pass a list of sklearn metrics when initiating to use this method.')
        
        if show_plots:
            self.parity_plots(caseset_obj)
            self.plot_oev(caseset_obj)
            self.plot_anisotropy(caseset_obj)
            self.get_plots_error(caseset_obj, error_function=self.relative_error)

        return results if dump_metrics else None

    def relative_error(self,
                      caseset_obj :CaseSet):
        try:
            bool(caseset_obj.labels)
            bool(caseset_obj.predictions)
            raise ValueError(f'[{caseset_obj.set_id or caseset_obj.case[0]}] labels or predictions missing')
        
        except ValueError:
            relative_error = np.array(
                [
                    np.linalg.norm(
                        caseset_obj.labels[i]-caseset_obj.predictions[i]
                    )/ np.linalg.norm(caseset_obj.labels[i])
                    for i in range(caseset_obj.labels.shape[0])
                ]            
            )
            return relative_error