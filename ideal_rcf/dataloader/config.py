from typing import List, Optional


class config(object):
    def __init__(self,
                 cases :List[str],
                 turb_dataset :str,
                 dataset_path :str,
                 features :List[str],
                 tensor_features :str,
                 tensor_features_linear : str,
                 labels :List[str],
                 custom_turb_dataset :Optional[str]=None,
                 tensor_features_eV :Optional[str]=None,
                 labels_eV :Optional[List[str]]=None,
                 features_filter :Optional[List[str]]=None,
                 Cx :Optional[str]='Cx',
                 Cy :Optional[str]='Cy',
                 u_velocity_label :Optional[str]='um',
                 v_velocity_label :Optional[str]='vm',
                 dataset_labels_dir :Optional[str]='labels') -> None:
        
        self.cases = self.ensure_list_instance(cases)
        self.turb_dataset = turb_dataset
        self.custom_turb_dataset = custom_turb_dataset
        self.dataset_path = dataset_path

        self.features = self.ensure_list_instance(features)
        self.tensor_features = self.ensure_str_instance(tensor_features)
        self.tensor_features_linear = self.ensure_str_instance(tensor_features_linear)
        self.labels = self.ensure_str_instance(labels)

        self.tensor_features_eV = self.ensure_str_instance(tensor_features_eV)
        self.labels_eV = self.ensure_str_instance(labels_eV)

        self.features_filter = self.ensure_list_instance(features_filter)
        
        self.Cx = Cx
        self.Cy = Cy

        self.u_velocity_label = u_velocity_label
        self.v_velocity_label = v_velocity_label

        self.dataset_labels_dir = dataset_labels_dir


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