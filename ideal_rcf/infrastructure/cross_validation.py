from ideal_rcf.infrastructure.evaluator import Evaluator
from ideal_rcf.dataloader.config import SetConfig
from ideal_rcf.dataloader.caseset import CaseSet
from ideal_rcf.dataloader.dataset import DataSet
from ideal_rcf.models.config import ModelConfig
from ideal_rcf.models.framework import FrameWork

from typing import Optional, Union, List, Dict
from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
import numpy as np
import os


class CrossValConfig(object):
    """
    Implements a cross-validation process based on given configurations and manages multiple folds for evaluation.

    Attributes:
    - `n_folds`: Number of folds for cross-validation.
    - `folds_config`: List of dictionaries specifying training and validation sets for each fold.
    - `use_best_n_folds`: Optional integer specifying the number of best folds to use for evaluation.
    - `cost_metrics`: Optional list of metrics to use as cost functions for selecting best folds.
    - `debug`: Optional boolean flag for debugging mode.

    Methods:
    - `update_base_fold_config(fold_config)`: Updates base set and model configurations for a specific fold.
    - `start()`: Initializes datasets and models for each fold based on `folds_config`.
    - `execute(show_plots=False)`: Executes cross-validation, trains models, and calculates metrics.
    - `indexes_of_n_lowest(arr)`: Returns the indices of the `n` lowest values in the array `arr`.
    - `get_best_n()`: Updates `best_folds` with indices of best performing folds based on evaluation metrics.
    - `inference(caseset)`: Performs inference using the best performing folds and updates predictions in `caseset`.
    - `dump_all(dir_path)`: Dumps scalers and model configurations from each fold to `dir_path`.
    - `load_all(dir_path)`: Loads scalers and model configurations from `dir_path` for each fold.
    """
    def __init__(self,
                 n_folds :int,
                 folds_config :List[Dict[str,Union[str,int]]],
                 use_best_n_folds :Optional[int]=None,
                 cost_metrics :Optional[List]=None,
                 debug :Optional[bool]=False) -> None:
        """
        Initializes a CrossVal object with given configurations.

                Args:
                - `n_folds`: Number of folds for cross-validation.
                - `folds_config`: List of dictionaries specifying training and validation sets for each fold.
                - `use_best_n_folds`: Optional integer specifying the number of best folds to use for evaluation.
                - `cost_metrics`: Optional list of metrics to use as cost functions for selecting best folds.
                - `debug`: Optional boolean flag for debugging mode.

                Example `folds_config`:
                ```json
                example_folds_config = [
                    {
                        'set': {
                            'trainset': ['PHLL_case_0p5', 'PHLL_case_0p8', 'PHLL_case_1p5'],
                            'valset': ['PHLL_case_1p0'],
                        },
                        'model': {
                            'random_seed': 42
                        }
                    },
                    {
                        'set': {
                            'trainset': ['PHLL_case_0p5', 'PHLL_case_1p0', 'PHLL_case_1p5'],
                            'valset': ['PHLL_case_0p8'],
                        },
                        'model': {
                            'random_seed': 84
                        }
                    },
                ]
                ```
        """
        self.n_folds = n_folds
        self.folds_config = folds_config
        assert self.n_folds == len(folds_config), f'folds_config should have lenght n_folds but got {len(folds_config)} and {self.folds_config}'
        self.use_best_n_folds = use_best_n_folds        
        self.cost_metrics = cost_metrics

        if self.use_best_n_folds:
            if not self.cost_metrics:
                raise ValueError('to use_best_n_folds you must pass a list of sklearn metrics to use as cost function')
        
        self.ensure_mandatory_entries({'set':{'trainset': None, 'valset':None}}) if self.use_best_n_folds else self.ensure_mandatory_entries({'set':{'trainset': None}})

        self.debug = debug


    def ensure_mandatory_entries(self, mandatory_dict :Dict):
        for key, value in mandatory_dict.items():
            for fold in self.folds_config:
                is_key = fold.get(key)
                if not is_key:
                    raise KeyError(f'folds_config must have key {key}')

                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        is_sub_key = fold[key].get(sub_key)
                        if not is_sub_key:
                            raise KeyError(f'folds_config {value} must have key {sub_key}')



class CrossVal(Evaluator):
    def __init__(self,
                 cross_val_config :CrossValConfig,
                 base_set_config :SetConfig,
                 base_model_config :ModelConfig,
                 exp_id :Optional[Path]=None,
                 img_folder :Optional[Path]=None) -> None:
         
        if not isinstance(cross_val_config, CrossValConfig):
            raise AssertionError(f'[config_error] cross_val_config must be of instance {CrossValConfig()}')

        if not isinstance(base_set_config, SetConfig):
            raise AssertionError(f'[config_error] base_set_config must be of instance {SetConfig()}')

        if not isinstance(base_model_config, ModelConfig):
            raise AssertionError(f'[config_error] base_model_config must be of instance {ModelConfig()}')


        super().__init__(cross_val_config.cost_metrics, exp_id, img_folder)

        self.cross_val_config = cross_val_config
        self.folds = []
        self.best_folds = [i for i in range(self.cross_val_config.n_folds)]
        self.base_set_config = base_set_config
        self.base_model_config = base_model_config
        self.start()


    def update_base_fold_config(self, fold_config :Dict[str,Union[str, int]]):
        set_config = deepcopy(self.base_set_config)
        model_config = deepcopy(self.base_model_config)

        base_set_config = fold_config.get('set')
        model_set_config = fold_config.get('model')

        if base_set_config:
            for attr, fold_value in base_set_config.items():
                setattr(set_config, attr, fold_value)
            
        if model_set_config:
            for attr, fold_value in model_set_config.items():
                setattr(model_config, attr, fold_value)

        return set_config, model_config


    def start(self):
        for fold, fold_config in enumerate(self.cross_val_config.folds_config):
            set_fold_config, model_set_config = self.update_base_fold_config(fold_config)
            
            current_fold = SimpleNamespace()
            current_fold.dataset = DataSet(set_config=set_fold_config)
            current_fold.model = FrameWork(model_set_config, _id=f'fold_{fold}')
            current_fold.model.compile_models()

            self.folds.append(current_fold)


    def execute(self, 
                show_plots :Optional[bool]=False):
        ### create, train, val and calculate metrics
        ### use top_n
        for fold in range(len(self.folds)):
            ### init scalers and split
            train, val = self.folds[fold].dataset.split_train_val_test()
            self.folds[fold].model.train(self.folds[fold].dataset, train, val)
            self.folds[fold].eval = self.calulate_metrics(val,show_plots=show_plots, dump_metrics=True)
        self.get_best_n if self.cross_val_config.use_best_n_folds else ...


    def indexes_of_n_lowest(self, 
                            arr :np.array):
        
        if self.cross_val_config.use_best_n_folds <= 0:
            raise ValueError("n must be a positive integer.")
        if self.cross_val_config.use_best_n_folds > len(arr):
            raise ValueError("n must not be greater than the length of the array.")
        
        ### Get the indexes that would sort the array
        sorted_indices = np.argsort(arr)
        ### Return the first n indexes
        return sorted_indices[:self.use_best_n_folds]


    def get_best_n(self):
        if len(self.best_folds) == self.cross_val_config.use_best_n_folds:
            return

        for fold in range(len(self.folds)):
            try:
                self.folds[fold].eval;
            except AttributeError:
                train, val = self.folds[fold].dataset.split_train_val_test()
                self.folds[fold].eval = self.calulate_metrics(val,show_plots=False, dump_metrics=True)
        
        metrics_matrix = np.array([fold_obj.eval for fold_obj in self.folds])  

        self.best_folds = self.indexes_of_n_lowest(metrics_matrix.mean(axis=1))


    def inference(self,
                  caseset :CaseSet):
        
        self.get_best_n if self.cross_val_config.use_best_n_folds else ...

        acc_preds = None
        acc_preds_oev = None
        for best_fold in self.best_folds:
            print(f'[fold {best_fold}]')
            predictions = self.folds[best_fold].model.inference(
                self.folds[best_fold].dataset,
                caseset,
                dump_predictions=True
            )

            if isinstance(predictions, tuple):
                preds_oev = predictions[0]
                preds = predictions[1]
            else:
                preds_oev = None
                preds = predictions

            try:
                bool(acc_preds)
                acc_preds = preds 
            except ValueError:
                acc_preds += preds

            try:
                bool(acc_preds_oev)
                acc_preds_oev = preds_oev
            except ValueError:
                acc_preds_oev += preds_oev

        acc_preds /= len(self.best_folds)
        caseset.predictions = acc_preds
        try:
            bool(acc_preds_oev)
        except ValueError:
            acc_preds_oev /= len(self.best_folds)
            caseset.predictions_oev = acc_preds_oev

        caseset.set_id = f'f_{"_".join([str(best_fold) for best_fold in self.best_folds])}_avg'


    def dump_all(self,
                 dir_path :Path):

        for fold, fold_obj in enumerate(self.folds):
            fold_dir = f'{dir_path}/fold_{fold}'
            if not os.path.exists(fold_dir):
                os.mkdir(fold_dir)
            print(f'[fold_{fold}]')
            fold_obj.dataset.dump_scalers(fold_dir)
            fold_obj.model.dump_to_dir(fold_dir)


    def load_all(self,
                 dir_path :Path):

        for fold in range(len(self.folds)):
            fold_dir = f'{dir_path}/fold_{fold}'
            print(f'[fold_{fold}]')
            self.folds[fold].dataset.load_scalers(fold_dir)
            self.folds[fold].model.load_from_dir(fold_dir)