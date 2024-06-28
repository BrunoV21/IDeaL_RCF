from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
                 custom_turb_dataset :Optional[str]=None,
                 tensor_features_eV :Optional[str]=None,
                 labels_eV :Optional[List[str]]=None,
                 labels_NL :Optional[List[str]]=None,
                 features_filter :Optional[List[str]]=None,
                 features_cardinality :Optional[List[int]]=None,
                 features_z_score_outliers_threshold :Optional[int]=None,
                 Cx :Optional[str]='Cx',
                 Cy :Optional[str]='Cy',
                 u_velocity_label :Optional[str]='um',
                 v_velocity_label :Optional[str]='vm',
                 debug :Optional[bool]=False,
                 dataset_labels_dir :Optional[str]='labels') -> None:
        
        self.cases = self.ensure_list_instance(cases)
        self.turb_dataset = turb_dataset
        self.custom_turb_dataset = custom_turb_dataset
        self.dataset_path = dataset_path

        self.features = self.ensure_list_instance(features)
        self.tensor_features = self.ensure_str_instance(tensor_features)
        self.tensor_features_linear = self.ensure_str_instance(tensor_features_linear)
        self.labels = self.ensure_str_instance(labels if not labels_NL else labels_NL)

        ### applied to features
        self.features_scaler = self.scalers_obj.get(features_scaler)
        
        ### applied to labels and tensor_features_linear
        self.labels_scaler = self.scalers_obj.get(labels_scaler) 

        self.trainset = self.ensure_list_instance(trainset)
        self.valset = self.ensure_list_instance(valset)
        self.testset = self.ensure_list_instance(testset)

        self.tensor_features_eV = self.ensure_str_instance(tensor_features_eV)
        
        ### if labels_eV is passed labels is used as the labels of NL term
        self.labels_eV = self.ensure_str_instance(labels_eV)

        self.features_filter = self.ensure_list_instance(features_filter)
        self.features_cardinality = self.ensure_list_instance(features_cardinality)

        if self.features_filter or self.features_cardinality:
            try:
                assert len(self.features) == len(self.features_cardinality),\
                    f'''features_filter and features_cardinality should be passed at the same time. Features and features_cardinalities must be lists with same lenghts but got {len(self.features)} and {len(self.features_cardinality)}'''
            
            except TypeError:
                raise TypeError(f'''features_filter and features_cardinality should be passed at the same time. Features and features_cardinalities must be lists with same lenghts but got {self.features} and {self.features_cardinality}''')
            
        self.remove_outliers_threshold = features_z_score_outliers_threshold

        self.Cx = Cx
        self.Cy = Cy

        self.u = u_velocity_label
        self.v = v_velocity_label

        self.dataset_labels_dir = dataset_labels_dir

        self.debug = debug
    

    scalers_obj = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler()
    }

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