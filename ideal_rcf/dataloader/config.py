from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Dict, Optional, Any
from copy import deepcopy
import numpy as np

class config(object):
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
                 labels_eV_scaler :Optional[str]='minmax',
                 custom_turb_dataset :Optional[str]=None,
                 tensor_features_eV :Optional[str]=None,
                 labels_eV :Optional[List[str]]=None,
                 labels_NL :Optional[List[str]]=None,
                 features_filter :Optional[List[str]]=None,
                 features_cardinality :Optional[List[int]]=None,
                 features_z_score_outliers_threshold :Optional[int]=None,
                 features_transforms :Optional[List[str]]=None,
                 skip_features_transforms_for :Optional[List[str]]=None,
                 Cx :Optional[str]='Cx',
                 Cy :Optional[str]='Cy',
                 u_velocity_label :Optional[str]='um',
                 v_velocity_label :Optional[str]='vm',
                 debug :Optional[bool]=False,
                 dataset_labels_dir :Optional[str]='labels',
                 pass_scalers_obj :Optional[Dict[str, Any]]=None,
                 pass_transforms_obj :Optional[Dict[str, Any]]=None) -> None:
        
        self.cases = self.ensure_list_instance(cases)
        self.turb_dataset = turb_dataset
        self.custom_turb_dataset = custom_turb_dataset
        self.dataset_path = dataset_path

        self.features = self.ensure_list_instance(features)
        self.tensor_features = self.ensure_str_instance(tensor_features)
        self.tensor_features_linear = self.ensure_str_instance(tensor_features_linear)
        self.labels = self.ensure_str_instance(labels if not labels_NL else labels_NL)        

        if pass_scalers_obj:
            self.scalers_obj.update(pass_scalers_obj)

        if pass_transforms_obj:
            self.transforms_obj.update(pass_transforms_obj)        

        ### applied to features
        self.features_scaler = deepcopy(self.scalers_obj.get(features_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {features_scaler}: no feature_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (features_scaler and not self.features_scaler) else ...
        
        ### applied to labels and tensor_features_linear
        self.labels_scaler = deepcopy(self.scalers_obj.get(labels_scaler)) 
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {labels_scaler}: no labels_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (labels_scaler and not self.labels_scaler) else ...

        ### applied to eV labels and
        self.labels_eV_scaler = deepcopy(self.scalers_obj.get(labels_eV_scaler))
        print(f'[WARNING] available scalers are keys from {self.scalers_obj} but got {labels_eV_scaler}: no labels_scaler will be applied. You can pass a custom scalers_obj containing sklearn scalers with pass_scalers_obj arg.') if (labels_eV_scaler and not self.labels_eV_scaler) else ...
        

        self.trainset = [[_case] for _case in self.ensure_list_instance(trainset)] if trainset else []
        self.valset = [[_case] for _case in self.ensure_list_instance(valset)] if valset else []
        self.testset = [[_case] for _case in self.ensure_list_instance(testset)] if testset else []

        self.tensor_features_eV = self.ensure_str_instance(tensor_features_eV)
        
        ### if labels_eV is passed labels is used as the labels of NL term
        self.labels_eV = self.ensure_str_instance(labels_eV)

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
            features = np.log(abs(features)+1)
            if debug:
                print('[transforms] applied log')
            
        return features

    scalers_obj = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler()
    }

    transforms_obj = {
        'multi_sign_cbrt': apply_cbrt_signal_changes.__get__(object),
        'same_sign_log': apply_log_no_signal_changes.__get__(object) 
    }