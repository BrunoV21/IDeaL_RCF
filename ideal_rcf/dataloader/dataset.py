from ideal_rcf.dataloader.config import SetConfig
from ideal_rcf.dataloader.caseset import CaseSet

from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
import joblib
import os

class DataSet(object):
    """
    A class for managing a collection of cases and their configurations.

    Parameters
    ----------
    cases : List[str], optional
        List of case identifiers.
    set_config : Optional[SetConfig], optional
        Configuration object containing setup details for the dataset.

    Methods
    -------
    __init__(cases: List[str]=None, set_config: Optional[SetConfig]=None) -> None
        Initializes the DataSet with cases and configuration, creating CaseSet instances.

    check_set()
        Validates the integrity of each CaseSet in the dataset.

    _filter()
        Placeholder method for filtering case sets (not implemented).

    shuffle()
        Shuffles the dataset (implemented in CaseSet).

    dump_scalers(dir_path: Path)
        Saves scaler objects to the specified directory.

    load_scalers(dir_path: Path)
        Loads scaler objects from the specified directory.

    stack_case_sets(case_sets: List[CaseSet], set_id: Optional[str]=None)
        Stacks multiple CaseSet instances into a single CaseSet.

    split_train_val_test()
        Splits the dataset into training, validation, and test sets based on configuration.
    """
    def __init__(self,
                 cases :List[str]=None,
                 set_config :Optional[SetConfig]=None) -> None:
        
        if set_config and not isinstance(set_config, SetConfig):
            raise AssertionError(f'set_config must of instance {SetConfig}')
        
        if set_config:        
            self.config = set_config
            
            self.cases = self.config.cases if  len(self.config.cases) > 1 else self.config.ensure_list_instance(cases)

            self.features_scaler = self.config.features_scaler
            self.labels_scaler = self.config.labels_scaler
            self.features_oev_scaler = self.config.features_oev_scaler
            self.labels_oev_scaler = self.config.labels_oev_scaler
            self.mixer_invariant_features_scaler = self.config.mixer_invariant_features_scaler
            self.mixer_invariant_oev_features_scaler = self.config.mixer_invariant_oev_features_scaler
            
            self.scaler_objs = [key for key, value in self.__dict__.items() if ('scaler' in key and value)]

            self.contents = [
                CaseSet(case=case, set_config=self.config)
                for case in tqdm(self.cases)
            ]
        
        else:
            print('DataSet initialized empty')


    def check_set(self):
        for case_obj in self.contents:
            case_obj.check_set()


    def _filter(self):
        ...
        return 'Not implemented yet'

    
    def shuffle(self):
        ...
        return 'Implemented at CaseSet level'


    def dump_scalers(self, 
                     dir_path :Path):
        scalers_dir = f'{dir_path}/scalers'
        if not os.path.exists(scalers_dir):
            os.mkdir(scalers_dir)

        for scaler_type in self.scaler_objs:
            scaler = getattr(self,scaler_type)
            joblib.dump(scaler,f'{scalers_dir}/{scaler_type}.save')
            print(f'[{scaler_type}] dumped sucessfully')


    def load_scalers(self, 
                     dir_path :Path):
        
        scalers_dir = f'{dir_path}/scalers'
        if not os.path.exists(scalers_dir):
            raise FileNotFoundError(f'Ensure {scalers_dir} exists and contains the scaler files')
        
        for scaler_file in os.listdir(Path(scalers_dir)):
            scaler_type = scaler_file.split('.')[0]
            scaler = joblib.load(f'{scalers_dir}/{scaler_file}')
            setattr(self, scaler_type, scaler)
            print(f'[{scaler_type}] loaded sucessfully')        


    def stack_case_sets(self,
                        case_sets :List[CaseSet],
                        set_id :Optional[str]=None):
        
        if not case_sets:
            return None
        
        stacked_case = case_sets[0]
        stacked_case.set_id = set_id

        for case_set in case_sets[1:]:
            stacked_case._stack(*case_set._export_for_stack())

        return stacked_case


    def split_train_val_test(self):
        local_contents = deepcopy(self.contents)

        train_set = []
        val_set = []
        test_set = []

        for case_set in local_contents:
            if case_set.case in self.config.trainset:
                train_set.append(case_set)
            elif case_set.case in self.config.valset:
                val_set.append(case_set)
            elif case_set.case in self.config.testset:
                test_set.append(case_set)
        
        train_set = self.stack_case_sets(train_set, set_id='train')
        val_set = self.stack_case_sets(val_set, set_id='val')
        test_set = self.stack_case_sets(test_set, set_id='test')        

        tain_val_test = tuple(_set for _set in [train_set, val_set, test_set] if _set)

        if self.config.debug:
            for _set in tain_val_test:
                _set.check_set()

        return tain_val_test