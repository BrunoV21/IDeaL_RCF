try:
    from dataloader.config import config

except ModuleNotFoundError:
    from config import config


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Union, Optional
import numpy as np

class CaseSet(object):
    def __init__(self,
                 case :str,
                 set_config :config,
                 set_id :Optional[str]=None) -> None:
        
        if not isinstance(set_config, config):
            raise AssertionError(f'set_config must be of instance {config()}')
        
        self.config = set_config

        self.set_id = set_id

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
        self.Cy = self.loadCombinedArray(self.config.Cy)

        self.u = self.loadLabels(self.config.u)
        self.v = self.loadLabels(self.config.v)

        if self.config.remove_outliers_threshold:
            self.remove_outliers()

        if self.config.debug:
            self.check_set()


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
            if self.config.debug:
                assert len(indexes_union) == len(self.config.features_filter)
                print(f'[{self.set_id or self.case[0]}] sucessfuly filtered features {self.config.features} to {self.config.features_filter}')

            features = features[:, indexes_union]
        
        return features


    def filtered_features_indexes(self):
        ### should add a way to configure indexes...
        ### most likely infering from features.shape
        ### 
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


    def get_outliers_index(self):
        stdev = np.std(self.features,axis=0)
        means = np.mean(self.features,axis=0)
        ind_drop = np.empty(0)
        for i in range(len(self.features[0,:])):
            ind_drop = np.concatenate(
                (
                    ind_drop,np.where(
                        (self.features[:,i]>means[i]+self.config.remove_outliers_threshold*stdev[i]) | (self.features[:,i]<means[i]-self.config.remove_outliers_threshold*stdev[i])
                    )[0]
                )
            )

        return np.unique(ind_drop.astype(int))


    def remove_outliers(self):
        outliers_index = self.get_outliers_index()

        if self.config.debug:
            print(f'[{self.set_id or self.case[0]}] Found {len(outliers_index)} outliers in {self.config.features} feature set')

        self.features = np.delete(self.features, outliers_index, axis=0)
        self.tensor_features = np.delete(self.tensor_features, outliers_index, axis=0)
        self.tensor_features_linear = np.delete(self.tensor_features_linear, outliers_index, axis=0) if self.config.tensor_features_linear else None
        self.labels = np.delete(self.labels, outliers_index, axis=0)
        
        self.tensor_features_eV = np.delete(self.tensor_features_eV, outliers_index, axis=0) if self.config.tensor_features_eV else None
        self.labels_eV = np.delete(self.labels_eV, outliers_index, axis=0) if self.config.labels_eV else None

        self.Cx = np.delete(self.Cx, outliers_index, axis=0)
        self.Cy = np.delete(self.Cy, outliers_index, axis=0)

        self.u = np.delete(self.u, outliers_index, axis=0)
        self.v = np.delete(self.v, outliers_index, axis=0)

    
    def _transform(self,):
        '''
        def transform_data(x_train, feat_list, feats_not_to_transform = ['I1_2', 'I1_5', 'I1_8','I1_15','I1_17', 'I1_19']):
            for i in range(x_train.shape[1]) :
                if feat_list[i][0] != 'q':
                    if feat_list[i] in feats_not_to_transform :
                        print(f'>{feat_list[i]} not transformed as it is not skewed enough')
                        continue            
                    elif x_train[:, i].min() < 0 < x_train[:, i].max():
                        #x_train[:, i] = np.cbrt(x_train[:, i])
                        print(f'>{feat_list[i]} data positive and negative: skippnig') 
                        continue
                    else :     
                        x_train[:, i] = np.log(abs(x_train[:, i])+1)
                        print(f'>{feat_list[i]} data strictly positive or negative: applying log transformation')
            return x_train        
        '''
        self.config.features_cardinality
        ### need to set up required transforms as well into config
        # for self


    def _scale(self,
               features_scaler :Union[StandardScaler, MinMaxScaler, None],
               labels_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        self.features = features_scaler.transform(self.features) if features_scaler else self.features
        self.labels = labels_scaler.transform(self.labels) if labels_scaler else self.labels
        ### need to add logic here to only take first n_entries if shapes don't match
        # self.tensor_features_linear = labels_scaler.transform(self.tensor_features_linear) if (labels_scaler and self.tensor_features_linear) else self.tensor_features_linear

    def _ensure_eV_shapes(self):
        if bool(self.config.labels_eV) == bool(self.config.tensor_features_eV):
            if self.config.tensor_features_eV:
                tensor_shape = self.tensor_features_eV.shape[1]
                if self.labels_eV.shape[1] > tensor_shape:
                    if self.config.debug:
                        print('')
                    self.labels_eV = self.labels_eV[:,:tensor_shape]

                elif self.labels_eV.shape[1] < tensor_shape:
                    raise ValueError(f'[{self.set_id or self.case[0]}] Config_Error: labels_eV ({self.config.labels_eV}) and tensor_features_eV ({self.config.tensor_features_eV} must have same dim 1 but have ({self.labels_eV.shape[1] }) and ({tensor_shape})')

        else:
            raise AssertionError(f'[{self.set_id or self.case[0]}] Config_Error: labels_eV ({self.config.labels_eV}) and tensor_features_eV ({self.config.tensor_features_eV} must be passed simultaneously)')


    def check_set(self):
        # List of attributes to check
        attributes = [
            'features',
            'tensor_features',
            'tensor_features_linear',
            'labels',
            'tensor_features_eV',
            'labels_eV',
            'Cx',
            'Cy',
            'u_velocity_label',
            'v_velocity_label'
        ]

        # Initialize a variable to store the first dimension of the first non-None attribute
        first_dim = None
        print(f'{self.set_id or self.case[0]}:')
        for attr in attributes:
            value = getattr(self, attr, None)
            if value is not None:
                shape = value.shape
                print(f' > {attr} ({getattr(self.config, attr, None)}): {shape}')
                
                if first_dim is None:
                    first_dim = shape[0]
                elif first_dim != shape[0]:
                    raise ValueError(f'{self.set_id or self.case[0]}: the first dimension of {attr} does not match the first dimension of the previous attributes')
        
        if first_dim is None:
            raise ValueError(f'{self.set_id or self.case[0]}: no attributes are set (all are None)')


if __name__ == '__main__':

    ### test module
    dataset_path = 'D:/OneDrive - Universidade de Lisboa/Turbulence Modelling Database'
    turb_datasete = 'komegasst'
    custom_turb_dataset = 'a_3_1_2_NL_S_DNS_eV'

    case = 'PHLL_case_1p2'
    features_filter = ['I1_1', 'I1_2', 'I1_3', 'I1_4', 'I1_5', 'I1_6', 'I1_8', 'I1_9', 'I1_15', 'I1_17', 'I1_19', 'I2_3', 'I2_4', 'q_1', 'q_2']
    features_cardinality = [20, 20, 4]

    features = ['I1', 'I2', 'q']
    tensor_features = ['Tensors']
    tensor_features_linear = ['Shat']
    labels = ['a_NL']

    tensor_features_eV = ['S_DNS']
    labels_eV = ['a']

    features_z_score_outliers_threshold = 10


    standard_case_test_configuration = config(
        cases=case,
        turb_dataset=turb_datasete,
        dataset_path=dataset_path,
        features=features,
        tensor_features=tensor_features,
        tensor_features_linear=tensor_features_linear,
        labels='b',
        debug=True
    )

    print('Standard case:')
    CaseSet(case, set_config=standard_case_test_configuration)

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
        features_filter=features_filter,
        features_cardinality=features_cardinality
    )

    print('\nCustom turb dataset with features filter:')
    CaseSet(case, set_config=optional_case_test_configuration).check_set()

    extra_optional_case_test_configuration = config(
        cases=case,
        turb_dataset=turb_datasete,
        dataset_path=dataset_path,
        features=features,
        tensor_features=tensor_features,
        features_z_score_outliers_threshold=features_z_score_outliers_threshold,
        # tensor_features_linear=tensor_features_linear,
        labels=labels,
        custom_turb_dataset=custom_turb_dataset,
        tensor_features_eV=tensor_features_eV,
        labels_eV=labels_eV,
        features_filter=features_filter,
        features_cardinality=features_cardinality,
        debug=True
    )

    print('\nCustom turb dataset with features filter, no SHAT term and remove outliers:')
    CaseSet(case, set_config=extra_optional_case_test_configuration)

    
