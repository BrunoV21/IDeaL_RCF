### calculating metrics 
### receives from framework, inference results and labels both stored in casesets
### receives list of sklearn metrics to calculate
from ideal_rcf.dataloader.caseset import CaseSet
from ideal_rcf.infrastructure.visualization import PlottingTools

from typing import Optional, List
from pathlib import Path
import numpy as np

class Evaluator(PlottingTools):
    def __init__(self, 
                 sklearn_metrics_list :Optional[List],
                 exp_id :Optional[Path]=None,
                 img_folder :Optional[Path]=None) -> None: 
        
        super().__init__(sklearn_metrics_list, exp_id, img_folder)


    def calculate_metrics(self,
                          caseset_obj :CaseSet,
                          show_plots :Optional[bool]=True):
        
        print(f'[{caseset_obj.set_id or caseset_obj.case[0]}] metrics')
        for metric in self.metrics:
            print(f' > {metric.__name__}: {self.format_float(metric(caseset_obj.labels, caseset_obj.predictions))}')

        if show_plots:
            self.parity_plots(caseset_obj)
            self.plot_oev(caseset_obj)
            self.plot_scalar(caseset_obj)
            self.get_plots_error(caseset_obj, error_function=self.relative_error)


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