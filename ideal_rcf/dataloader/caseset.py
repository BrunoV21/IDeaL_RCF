try:
    from dataloader.config import config

except ModuleNotFoundError:
    from config import config

from typing import List
import numpy as np

class CaseSet(object):
    def __init__(self,
                 case :str,
                 set_config :config) -> None:
        
        if not isinstance(set_config, config):
            raise AssertionError(f'set_config must of instance {config()}')
        
        self.config = set_config

        self.case = self.config.ensure_list_instance(case)
        
        self.features = self.filter_features(
            self.loadCollumnStackFeatures(self.config.features)
        )
        self.tensor_features = self.loadCombinedArray(self.config.tensor_features)
        self.tensor_features_linear = self.loadCombinedArray(self.config.tensor_features_linear)
        self.labels = self.loadLabels(self.config.labels)

        self.tensor_features_eV = self.loadCombinedArray(self.config.tensor_features_eV)
        self.labels_eV = self.loadLabels(self.config.labels_eV)

        self.Cx = self.loadCombinedArray(self.config.Cx)
        self.Cx = self.loadCombinedArray(self.config.Cy)

        self.u_velocity_label = self.loadLabels(self.config.u_velocity_label)
        self.v_velocity_label = self.loadLabels(self.config.v_velocity_label)


    def loadLabels(self, 
                   field :List[str]):
        
        if not field:
            return None
        try:
            data = np.concatenate([
                np.load(f'{self.config.dataset_path}/{self.config.dataset_labels_dir}/{case}_{field}.npy') 
                for case in self.case
            ])

        except FileNotFoundError:
            data = self.loadCombinedArray(field)

        return data
    
    def loadCombinedArray(self,
                          field :List[str]):
        
        if not field:
            return None
        
        if self.config.custom_turb_dataset:
            data = np.concatenate([
                np.load(f'{self.config.dataset_path}/{self.config.turb_dataset}/{self.config.custom_turb_dataset}/{self.config.turb_dataset}_{case}_{field}.npy')
                for case in self.case
            ])    

        else:
            data = np.concatenate([
                np.load(f'{self.config.dataset_path}/{self.config.turb_dataset}/{self.config.turb_dataset}_{case}_{field}.npy')
                for case in self.case
            ])  

        return data
        
    
    def loadCollumnStackFeatures(self,
                                 fields :List[str]):
        if not fields:
            return None

        for i, field in enumerate(fields):
            data = self.loadCombinedArray(field)
            if i == 0 :
                features = data
            else : 
                features = np.column_stack((features, data))

        return features
    
    
    def filter_features(self, 
                        features :np.array):
    
        if self.config.features_filter:
            indexes_union = self.filtered_features_indexes()
            features = features[:, indexes_union]
        
        return features
       
    
    def filtered_features_indexes(self):

        indexes = []

        for feature in self.config.features_filter:
            if feature[0] == 'I':
                if feature[1] == '1':
                    indexes.append(int(feature[3:])-1)
                if feature[1] == '2':
                    indexes.append(int(feature[3:])+19) 
            else:
                indexes.append(int(feature[-1])+39)
                
        return indexes


    def remove_outliers(self):
        ...
        '''
        def remove_outliers(Features):
            stdev = np.std(Features,axis=0)
            means = np.mean(Features,axis=0)
            ind_drop = np.empty(0)
            for i in range(len(Features[0,:])):
                ind_drop = np.concatenate((ind_drop,np.where((Features[:,i]>means[i]+5*stdev[i]) | (Features[:,i]<means[i]-5*stdev[i]) )[0]))
            return np.unique(ind_drop.astype(int))

        def remove_outliers_test(x, basis, Shat, y, Cx, Cy):
            outlier_index = remove_outliers(x) #Features
            print('Found '+str(len(outlier_index))+' outliers in the input feature set')
            x = np.delete(x, outlier_index,axis=0)
            if bool(basis.any()):
                basis = np.delete(basis,outlier_index,axis=0)
            Shat = np.delete(Shat, outlier_index,axis=0)     
            y = np.delete(y,outlier_index,axis=0)
            Cx = np.delete(Cx,outlier_index,axis=0)
            Cy = np.delete(Cy,outlier_index,axis=0)
            return [x, basis, Shat, y, Cx, Cy, outlier_index]
        '''
        return 'not impplemented yet'
     

    
    
    def check_set(self):
        # List of attributes to check
        attributes = [
            'features',
            'tensor_features',
            'tensor_features_linear',
            'labels',
            'tensor_features_eV',
            'labels_eV',
            'u_velocity_label',
            'v_velocity_label'
        ]

        # Initialize a variable to store the first dimension of the first non-None attribute
        first_dim = None
        print(f'{self.case[0]}:')
        for attr in attributes:
            value = getattr(self, attr, None)
            if value is not None:
                shape = value.shape
                print(f' > {attr} ({getattr(self.config, attr, None)}): {shape}')
                
                if first_dim is None:
                    first_dim = shape[0]
                elif first_dim != shape[0]:
                    raise ValueError(f'The first dimension of {attr} does not match the first dimension of the previous attributes')
        
        if first_dim is None:
            raise ValueError('No attributes are set (all are None)')
    

if __name__ == '__main__':

    ### test module
    dataset_path = 'D:/OneDrive - Universidade de Lisboa/Turbulence Modelling Database'
    turb_datasete = 'komegasst'
    custom_turb_dataset = 'a_3_1_2_NL_S_DNS_eV'

    case = 'PHLL_case_1p2'
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
    CaseSet(case, set_config=standard_case_test_configuration).check_set()

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
    CaseSet(case, set_config=optional_case_test_configuration).check_set()


    
