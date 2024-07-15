from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict, Optional, Any
from copy import deepcopy
import numpy as np
import json
import os

class SetConfig(object):
    """
    Configuration class for setting up dataset parameters for turbulence modeling.

    Attributes:
        cases (List[str]): List of case names.
        turb_dataset (str): Name of the turbulence dataset.
        dataset_path (str): Path to the dataset.
        features (List[str]): List of feature names.
        tensor_features (str): Tensor features.
        labels (Optional[List[str]]): List of label names. Defaults to None.
        tensor_features_linear (Optional[str]): Linear tensor features. Defaults to None.
        trainset (Optional[List[str]]): List of training set case names. Defaults to None.
        valset (Optional[List[str]]): List of validation set case names. Defaults to None.
        testset (Optional[List[str]]): List of test set case names. Defaults to None.
        features_scaler (Optional[str]): Scaler for features. Defaults to 'minmax'.
        labels_scaler (Optional[str]): Scaler for labels. Defaults to 'standard'.
        features_oev_scaler (Optional[str]): Scaler for OEV features. Defaults to 'standard'.
        labels_oev_scaler (Optional[str]): Scaler for OEV labels. Defaults to 'standard'.
        mixer_invariant_features_scaler (Optional[str]): Scaler for mixer invariant features. Defaults to 'minmax'.
        mixer_invariant_oev_features_scaler (Optional[str]): Scaler for mixer invariant OEV features. Defaults to 'minmax'.
        custom_turb_dataset (Optional[str]): Custom turbulence dataset. Defaults to None.
        tensor_features_oev (Optional[str]): Tensor features for OEV. Defaults to None.
        features_filter (Optional[List[str]]): Filter for features. Defaults to None.
        features_cardinality (Optional[List[int]]): Cardinality of features. Defaults to None.
        features_z_score_outliers_threshold (Optional[int]): Threshold for z-score outlier removal. Defaults to None.
        features_transforms (Optional[List[str]]): List of feature transforms. Defaults to None.
        skip_features_transforms_for (Optional[List[str]]): Features to skip transforms for. Defaults to None.
        Cx (Optional[str]): X coordinate label. Defaults to 'Cx'.
        Cy (Optional[str]): Y coordinate label. Defaults to 'Cy'.
        u_velocity_label (Optional[str]): Label for u-velocity. Defaults to 'um'.
        v_velocity_label (Optional[str]): Label for v-velocity. Defaults to 'vm'.
        random_seed (Optional[int]): Random seed for shuffling. Defaults to 42.
        enable_mixer (Optional[bool]): Flag to enable mixer. Defaults to False.
        debug (Optional[bool]): Flag to enable debug mode. Defaults to False.
        dataset_labels_dir (Optional[str]): Directory for dataset labels. Defaults to 'labels'.
        pass_scalers_obj (Optional[Dict[str, Any]]): Dictionary of custom scalers. Defaults to None.
        pass_transforms_obj (Optional[Dict[str, Any]]): Dictionary of custom transforms. Defaults to None.
        pass_mixer_propertires_obj (Optional[Dict[str, Any]]): Dictionary of custom mixer properties. Defaults to None.

    Methods:
        __init__(self, cases, turb_dataset, dataset_path, features, tensor_features, labels=None, 
                 tensor_features_linear=None, trainset=None, valset=None, testset=None, 
                 features_scaler='minmax', labels_scaler='standard', features_oev_scaler='standard', 
                 labels_oev_scaler='standard', mixer_invariant_features_scaler='minmax', 
                 mixer_invariant_oev_features_scaler='minmax', custom_turb_dataset=None, 
                 tensor_features_oev=None, features_filter=None, features_cardinality=None, 
                 features_z_score_outliers_threshold=None, features_transforms=None, 
                 skip_features_transforms_for=None, Cx='Cx', Cy='Cy', u_velocity_label='um', 
                 v_velocity_label='vm', random_seed=42, enable_mixer=False, debug=False, 
                 dataset_labels_dir='labels', pass_scalers_obj=None, pass_transforms_obj=None, 
                 pass_mixer_propertires_obj=None):
            Initializes the configuration with the provided parameters.
        
        ensure_list_instance(self, attribute):
            Ensures that the provided attribute is a list.
        
        ensure_str_instance(self, attribute):
            Ensures that the provided attribute is a string.
        
        build_features_from_cardinality(self):
            Builds features based on the provided cardinality.
        
        apply_cbrt_signal_changes(features, debug=False):
            Applies cubic root transformation to features with both positive and negative values.
        
        apply_log_no_signal_changes(features, debug=False):
            Applies logarithmic transformation to features with only positive or only negative values.
        
        PHLL_coords_norm(Cx, Cy, case, PHLL_propeties={'H': 5.142, 'A': 3.858}):
            Normalizes coordinates for PHLL cases.
        
        BUMP_coords_norm(Cx, Cy, case, BUMP_propeties={'C': 0.5 * 0.305}):
            Normalizes coordinates for BUMP cases.
        
        IDENTITY_coords(Cx, Cy, case):
            Returns the original coordinates without any normalization.
    """
    def __init__(self,
                 cases :List[str],
                 turb_dataset :str,
                 dataset_path :str,
                 features :List[str],
                 tensor_features :str,
                 labels :Optional[List[str]]=None,                 
                 tensor_features_linear :Optional[str]=None,
                 trainset :Optional[List[str]]=None,
                 valset :Optional[List[str]]=None,
                 testset :Optional[List[str]]=None,
                 features_scaler :Optional[str]='minmax',
                 labels_scaler :Optional[str]='standard',
                 features_oev_scaler :Optional[str]='standard',
                 labels_oev_scaler :Optional[str]='standard',
                 mixer_invariant_features_scaler :Optional[str]='minmax',
                 mixer_invariant_oev_features_scaler :Optional[str]='minmax',
                 custom_turb_dataset :Optional[str]=None,
                 tensor_features_oev :Optional[str]=None,
                 features_filter :Optional[List[str]]=None,
                 features_cardinality :Optional[List[int]]=None,
                 features_z_score_outliers_threshold :Optional[int]=None,
                 features_transforms :Optional[List[str]]=None,
                 skip_features_transforms_for :Optional[List[str]]=None,
                 Cx :Optional[str]='Cx',
                 Cy :Optional[str]='Cy',
                 u_velocity_label :Optional[str]='um',
                 v_velocity_label :Optional[str]='vm',
                 random_seed :Optional[int]=42,
                 enable_mixer :Optional[bool]=False,
                 debug :Optional[bool]=False,
                 dataset_labels_dir :Optional[str]='labels',
                 pass_scalers_obj :Optional[Dict[str, Any]]=None,
                 pass_transforms_obj :Optional[Dict[str, Any]]=None,
                 pass_mixer_propertires_obj :Optional[Dict[str, Any]]=None) -> None:
        
        self.cases = self.ensure_list_instance(cases)
        self.turb_dataset = turb_dataset
        self.custom_turb_dataset = custom_turb_dataset
        self.dataset_path = dataset_path

        self.features = self.ensure_list_instance(features)
        self.tensor_features = self.ensure_str_instance(tensor_features)
        self.tensor_features_linear = self.ensure_str_instance(tensor_features_linear)
        self.labels = self.ensure_str_instance(labels)# if not labels_NL else labels_NL)        

        if pass_scalers_obj:
            self.scalers_obj.update(pass_scalers_obj)

        if pass_transforms_obj:
            self.transforms_obj.update(pass_transforms_obj)

        if pass_mixer_propertires_obj:
            self.mixer_propertires_obj = pass_mixer_propertires_obj

        ### applied to features
        self.features_scaler = deepcopy(self.scalers_obj.get(features_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {features_scaler}: no feature_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (features_scaler and not self.features_scaler) else ...
        
        ### applied to labels and tensor_features_linear
        self.labels_scaler = deepcopy(self.scalers_obj.get(labels_scaler)) 
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {labels_scaler}: no labels_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (labels_scaler and not self.labels_scaler) else ...
        
        ### aplied to oev feautres
        self.features_oev_scaler = deepcopy(self.scalers_obj.get(features_oev_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {features_oev_scaler}: no features_oev_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (features_oev_scaler and not self.features_oev_scaler) else ...

        ### applied to eV labels and
        self.labels_oev_scaler = deepcopy(self.scalers_obj.get(labels_oev_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {labels_oev_scaler}: no labels_oev_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (labels_oev_scaler and not self.labels_oev_scaler) else ...
        
        self.mixer_invariant_features_scaler = deepcopy(self.scalers_obj.get(mixer_invariant_features_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {mixer_invariant_features_scaler}: no mixer_invariant_features_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (mixer_invariant_features_scaler and not self.mixer_invariant_features_scaler) else ...

        self.mixer_invariant_oev_features_scaler = deepcopy(self.scalers_obj.get(mixer_invariant_oev_features_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {mixer_invariant_oev_features_scaler}: no mixer_invariant_features_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (mixer_invariant_oev_features_scaler and not self.mixer_invariant_oev_features_scaler) else ...

        self.trainset = [[_case] for _case in self.ensure_list_instance(trainset)] if trainset else []
        self.valset = [[_case] for _case in self.ensure_list_instance(valset)] if valset else []
        self.testset = [[_case] for _case in self.ensure_list_instance(testset)] if testset else []

        self.tensor_features_oev = self.ensure_str_instance(tensor_features_oev)
        
        ### if labels_oev is passed labels is used as the labels of NL term
        # self.labels_oev = self.ensure_str_instance(labels_oev)

        self.features_filter = self.ensure_list_instance(features_filter)
        
        self.features_cardinality = self.ensure_list_instance(features_cardinality)
        
        self.all_features = self.build_features_from_cardinality() if features_cardinality else None

        if not features_filter and features_cardinality:
            self.features_filter = deepcopy(self.all_features)            
        
        if self.features_filter or self.features_cardinality:
            try:
                assert len(self.features) == len(self.features_cardinality),\
                    f'''features_filter and features_cardinality should be passed at the same time. Features and features_cardinalities must be lists with same lenghts but got {len(self.features)} and {len(self.features_cardinality)}'''
            
            except TypeError:
                raise TypeError(f'''features_filter and features_cardinality should be passed at the same time. Features and features_cardinalities must be lists with same lenghts but got {self.features} and {self.features_cardinality}''')

        validated_features_transforms = []
        if features_transforms:
            for transform in features_transforms:
                if transform not in self.transforms_obj.keys():
                    print(f'[WARNING] available transforms are keys from {self.transforms_obj} but got {transform} which will be ignored. You can pass a custom transform_obj containing a user_defined_function with pass_transforms_obj arg.')
                else:
                    validated_features_transforms.append(transform)

        self.features_transforms = [self.transforms_obj[selected_transform] for selected_transform in  validated_features_transforms]
        self.skip_features_transforms_for = skip_features_transforms_for
        
        self.remove_outliers_threshold = features_z_score_outliers_threshold

        self.Cx = Cx
        self.Cy = Cy

        self.u = u_velocity_label
        self.v = v_velocity_label

        self.dataset_labels_dir = dataset_labels_dir

        self.random_seed = random_seed ### used for shuffling

        self.enable_mixer = enable_mixer

        self.debug = debug


    def ensure_list_instance(self, attribute):
        if isinstance(attribute, list) or not attribute:
            return attribute
        else:
            return [attribute]


    def ensure_str_instance(self, attribute):
        if isinstance(attribute, list):
            return attribute[0]
        else:
            return attribute


    def build_features_from_cardinality(self):
        features_list = [
            [
                f'{feature}_{i}' 
                for i in range(1, cardinality+1)
            ] for feature, cardinality in zip(self.features, self.features_cardinality)
        ]

        features_list = sum(features_list, [])
        return features_list


    @staticmethod
    def apply_cbrt_signal_changes(features :np.array,
                                  debug :Optional[bool]=False):
        ### applied through features dim 1, i.e cols of matrix
        if features.min() < 0 < features.max():
            features = np.cbrt(features)
            if debug:
                print('[transforms] applied cbrt')

        return features

    @staticmethod
    def apply_log_no_signal_changes(features :np.array,
                                    debug :Optional[bool]=False):
        if not features.min() < 0 < features.max():
            features = np.log(np.abs(features)+1)
            if debug:
                print('[transforms] applied log')
            
        return features

    @staticmethod
    def PHLL_coords_norm(Cx :np.array, 
                         Cy :np.array,
                         case :str,
                         PHLL_propeties :Dict={
                            'H': 5.142, 
                            'A':3.858
                         }):
        norm_Cx = Cx\
            /(
                PHLL_propeties['A']\
                *float(f'{case[-3]}.{case[-1]}')\
                +PHLL_propeties['H']
            )
        
        return norm_Cx, Cy

    @staticmethod
    def BUMP_coords_norm(Cx :np.array, 
                         Cy :np.array,
                         case :str,
                         BUMP_propeties :Dict={
                            'C': 0.5*0.305
                         }):
        norm_Cx = Cx/BUMP_propeties['C']
        norm_Cy = Cy/float(f'0.0{case[-2:]}')       
        
        return norm_Cx, norm_Cy

    @staticmethod
    def IDENTITY_coords(Cx :np.array, 
                        Cy :np.array,
                        case :str):        
        return Cx, Cy

    scalers_obj = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler()
    }

    transforms_obj = {
        'multi_sign_cbrt': apply_cbrt_signal_changes.__get__(object),
        'same_sign_log': apply_log_no_signal_changes.__get__(object) 
    }

    ### function that transforms/normalizes x and y coordinates
    mixer_propertires_obj = {
        'PHLL': PHLL_coords_norm.__get__(object),
        'BUMP': BUMP_coords_norm.__get__(object),
        'CNDV': IDENTITY_coords.__get__(object)
    }


def set_dataset_path(custom_path :Optional[str]=None):
    """
    Set the dataset path in the environment variable 'DATASET_PATH'.

    This function determines the dataset path based on the following order of precedence:
    1. If a file named 'local_config.json' exists in the current working directory, 
       it reads the 'dataset_path' from this file.
    2. If `custom_path` is provided, it uses this path.
    3. Otherwise, it defaults to './ml-turbulence-dataset'.

    The determined path is then set in the environment variable 'DATASET_PATH'.

    Args:
        custom_path (Optional[str]): An optional custom path to the dataset.

    Raises:
        KeyError: If 'dataset_path' key is not found in 'local_config.json'.
        json.JSONDecodeError: If 'local_config.json' contains invalid JSON.

    Example:
        >>> set_dataset_path('/custom/path/to/dataset')
    >>> print(os.environ['DATASET_PATH'])
        /custom/path/to/dataset
    """    
    if 'local_config.json' in os.listdir(os.getcwd()):
        with open('./local_config.json',  'r') as _file:
            content = json.load(_file)
        dataset_path = content['dataset_path']
    elif custom_path:
        dataset_path = custom_path
    else:
        dataset_path = './ml-turbulence-dataset'    
    os.environ['DATASET_PATH'] = dataset_path