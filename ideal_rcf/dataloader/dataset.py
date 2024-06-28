try:
    from dataloader.config import config
    from dataloader.caseset import CaseSet

except ModuleNotFoundError:
    from config import config
    from caseset import CaseSet

from typing import List
from tqdm import tqdm
from copy import deepcopy

class DataSet(object):
    def __init__(self,
                 cases :List[str]=None,
                 set_config :config=None) -> None:
        
        if not isinstance(set_config, config):
            raise AssertionError(f'set_config must of instance {config()}')
        
        self.config = set_config
        
        self.cases = self.config.cases if  len(self.config.cases) > 1 else self.config.ensure_list_instance(cases)
        
        self.contents = [
            CaseSet(case=case, set_config=self.config)
            for case in tqdm(self.cases)
        ]


    def check_set(self):
        for case_obj in self.contents:
            case_obj.check_set()


    def _filter(self):
        ...
        return 'Not implemented yet'


    def fit_scaler(self,
                   scaler :config.scalers_obj,
                   train_set :List[CaseSet]):
        
        if not scaler:
            return
        ...
        scaler.fit(train_set)


    def apply_scaler(self,
                     scaler :config.scalers_obj,
                    _set :List[CaseSet]):
        ### can remove it as is already implemeted at caseset kvk

        if scaler:
            _set = scaler.transform(_set)

        return _set
    



    def split_train_val_test(self):
        ...
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

        ### need to apply scaler on a per component basis via caseset
        
        # self.fit_scaler(train_set)
        # train_set = self.apply_scaler(train_set)
        # val_set = self.apply_scaler(val_set)´

        ### each of train, val, test can be a case_set resulting from statcing
        ### set_id = train, val, test

        ### operations to perform in train, val, test
        ### outliers removal
        ### features transform
        ### features scaling


        return 'Not implemented yet'


if __name__ == '__main__':

    ### test module
    dataset_path = 'D:/OneDrive - Universidade de Lisboa/Turbulence Modelling Database'
    turb_datasete = 'komegasst'
    custom_turb_dataset = 'a_3_1_2_NL_S_DNS_eV'

    case = [
        'PHLL_case_0p5',
        'PHLL_case_0p8',
        'PHLL_case_1p0',
        'PHLL_case_1p2',
        'PHLL_case_1p5'
    ]
    
    features_filter = ['I1_1', 'I1_2', 'I1_3', 'I1_4', 'I1_5', 'I1_6', 'I1_8', 'I1_9', 'I1_15', 'I1_17', 'I1_19', 'I2_3', 'I2_4', 'q_1', 'q_2']

    features = ['I1', 'I2', 'q']
    tensor_features = ['Tensors']
    tensor_features_linear = ['Shat']
    labels = ['a_NL']

    tensor_features_eV = ['S_DNS']
    labels_eV = ['a']


    standard_case_test_configuration = config(
        cases=case,
        turb_dataset=turb_datasete,
        dataset_path=dataset_path,
        features=features,
        tensor_features=tensor_features,
        tensor_features_linear=tensor_features_linear,
        labels='b'
    )

    print('Standard case:')
    DataSet(set_config=standard_case_test_configuration).check_set()

    optional_case_test_configuration = config(
        cases=case,
        turb_dataset=turb_datasete,
        dataset_path=dataset_path,
        features=features,
        tensor_features=tensor_features,
        tensor_features_linear=tensor_features_linear,
        labels=labels,
        custom_turb_dataset=custom_turb_dataset,
        tensor_features_eV=tensor_features_eV,
        labels_eV=labels_eV,
        features_filter=features_filter
    )

    print('\nCustom turb dataset with features filter:')
    DataSet(set_config=optional_case_test_configuration).check_set()


    
