try:
    from dataloader.config import config
    from dataloader.caseset import CaseSet

except ModuleNotFoundError:
    from config import config
    from caseset import CaseSet

from typing import List, Optional
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

        self.features_scaler = self.config.features_scaler
        self.labels_scaler = self.config.labels_scaler
        self.labels_eV_scaler = self.config.labels_eV_scaler
        self.mixer_invariant_features_scaler = self.config.mixer_invariant_features_scaler
        
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

    
    def shuffle(self):
        ...
        return 'Should be implemted at CaseSet level'


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
        
        if train_set:
            ### build scalers
            self.features_scaler, self.labels_scaler, self.labels_eV_scaler = train_set._fit_scaler(self.features_scaler , self.labels_scaler, self.labels_eV_scaler)
            
            ### build mixer features if enablred
            if self.config.enable_mixer:
                self.mixer_invariant_features_scaler = train_set._fit_mixer_scaler(self.mixer_invariant_features_scaler)
                train_set._scale_mixer(self.mixer_invariant_features_scaler)
                val_set._scale_mixer(self.mixer_invariant_features_scaler)
                test_set._scale_mixer(self.mixer_invariant_features_scaler)

                train_set._build_mixer_features(self.mixer_invariant_features_scaler)
                val_set._build_mixer_features(self.mixer_invariant_features_scaler) if self.config.valset else ...
                test_set._build_mixer_features(self.mixer_invariant_features_scaler) if self.config.testset else ...
            
            ### scale set
            train_set._transform_scale(self.features_scaler, self.labels_scaler, self.labels_eV_scaler)
            val_set._transform_scale(self.features_scaler, self.labels_scaler, self.labels_eV_scaler) if self.config.valset else ...
            test_set._transform_scale(self.features_scaler, self.labels_scaler, self.labels_eV_scaler) if self.config.testset else ...

        tain_val_test = tuple(_set for _set in [train_set, val_set, test_set] if _set)

        return tain_val_test


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

    trainset = [
        'PHLL_case_0p5',
        'PHLL_case_0p8',
        'PHLL_case_1p5'
    ]

    valset = [
        'PHLL_case_1p0',
    ]
    
    testset = [
        'PHLL_case_1p2',
    ]

    features_filter = ['I1_1', 'I1_2', 'I1_3', 'I1_4', 'I1_5', 'I1_6', 'I1_8', 'I1_9', 'I1_15', 'I1_17', 'I1_19', 'I2_3', 'I2_4', 'q_1', 'q_2']

    features = ['I1', 'I2', 'q']
    features_cardinality = [20, 20, 4]

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
    a = DataSet(set_config=standard_case_test_configuration)
    a.check_set()
    a.split_train_val_test()

    optional_case_test_configuration = config(
        cases=case,
        turb_dataset=turb_datasete,
        dataset_path=dataset_path,
        trainset=trainset,
        valset=valset,
        testset=testset,
        features=features,
        tensor_features=tensor_features,
        tensor_features_linear=tensor_features_linear,
        labels=labels,
        custom_turb_dataset=custom_turb_dataset,
        tensor_features_eV=tensor_features_eV,
        labels_eV=labels_eV,
        features_filter=features_filter,
        features_cardinality=features_cardinality,
        debug=True,
    )

    print('\nCustom turb dataset with features filter:')
    b = DataSet(set_config=optional_case_test_configuration)
    b.check_set()
    b.split_train_val_test()

    mixer_case_test_configuration = config(
        cases=case,
        turb_dataset=turb_datasete,
        dataset_path=dataset_path,
        trainset=trainset,
        valset=valset,
        testset=testset,
        features=features,
        tensor_features=tensor_features,
        tensor_features_linear=tensor_features_linear,
        labels=labels,
        custom_turb_dataset=custom_turb_dataset,
        tensor_features_eV=tensor_features_eV,
        labels_eV=labels_eV,
        features_filter=features_filter,
        features_cardinality=features_cardinality,
        enable_mixer=True,
        debug=True,
    )

    print('\nCustom turb dataset with features filter and mixer enabled:')
    c = DataSet(set_config=mixer_case_test_configuration)
    c.check_set()
    c.split_train_val_test()

    ### add simple tests here
    ### improve check_set for dataset
    ### i.e don't pass teest_set and assert shape
    ### shuffle method should be implemted at CaseSet level

    ### work on keras models module
    ### work on infrascture
    ### i.e receive CaseSet from DataSet, receive KerasModel -> Cross Val, Training Stuff, Realizability (store on case_set), Metrics, Visualization, Caching/Loading DataSet, CaseSets