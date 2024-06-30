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
            raise AssertionError(f'[config_error] set_config must be of instance {config()}')
        
        self.config = set_config

        self.set_id = set_id

        self.case = self.config.ensure_list_instance(case)
        
        self.features = self.loadCollumnStackFeatures(self.config.features)

        self.tensor_features = self.loadCombinedArray(self.config.tensor_features)
        self.tensor_features_linear = self.loadCombinedArray(self.config.tensor_features_linear)
        self.labels = self.loadLabels(self.config.labels)

        self.tensor_features_eV = self.loadCombinedArray(self.config.tensor_features_eV)
        self.labels_eV = self.loadLabels(self.config.labels_eV)
        self._ensure_eV_shapes()

        self.Cx = self.loadCombinedArray(self.config.Cx)
        self.Cy = self.loadCombinedArray(self.config.Cy)

        self.u = self.loadLabels(self.config.u)
        self.v = self.loadLabels(self.config.v)
            
        if self.config.features_filter and self.config.features_filter != self.config.all_features:
            self._filter_features()

        if self.config.remove_outliers_threshold:
            self._remove_outliers()

        if self.config.enable_mixer:
            try:
                self.augmented_spatial_mixing_coords = np.hstack(self.config.mixer_propertires_obj[self.case[0][:4]](self.Cx, self.Cy, self.case[0]))

            except KeyError:
                raise KeyError(f'[config_error] available mixer_properties_obj are {self.config.mixer_propertires_obj}. You can pass a new obj with arg pass_mixer_propertires_obj')
        else:
            self.augmented_spatial_mixing_coords = None
        
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

        if len(data.shape) == 1:
            data = data.reshape((-1, 1))

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

        if len(data.shape) == 1:
            data = data.reshape((-1, 1))

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


    def _filter_features(self,):
        indexes_union = [self.config.all_features.index(feature) for feature in self.config.features_filter]
        if self.config.debug:
            assert len(indexes_union) == len(self.config.features_filter)
            print(f'[{self.set_id or self.case[0]}] sucessfuly filtered features {self.config.features} to {self.config.features_filter}')

        self.features = self.features[:, indexes_union]


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


    def _remove_outliers(self):
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

    
    def _transform_features(self):
        ### must be applied after features_filter
        print(f'[{self.set_id or self.case[0]}]')
        for i, feature in enumerate(self.config.features_filter):
            if feature in self.config.skip_features_transforms_for:
                continue
            print(f'[transforms] {feature}:')
            for transform in self.config.features_transforms:
                self.features[:,i] = transform(self.features[:,i], self.config.debug)


    def _fit_scaler(self,
                    features_scaler :Union[StandardScaler, MinMaxScaler, None],
                    labels_scaler :Union[StandardScaler, MinMaxScaler, None],
                    labels_eV_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        features_scaler.fit(self.features) if features_scaler else ...
        labels_scaler.fit(self.labels) if labels_scaler else ...
        labels_eV_scaler.fit(self.labels_eV) if (labels_eV_scaler and self.config.labels_eV) else ...
        
        if self.config.debug:
            applied_scalers = [
                scaler for scaler in [features_scaler, labels_scaler, labels_eV_scaler] 
                if scaler
            ]
            print(f'[{self.set_id or self.case[0]}] fitted scalers {applied_scalers}')

        return features_scaler, labels_scaler, labels_eV_scaler


    def _transform_scale(self,
               features_scaler :Union[StandardScaler, MinMaxScaler, None],
               labels_scaler :Union[StandardScaler, MinMaxScaler, None],
               labels_eV_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        if self.config.features_transforms:
            self._transform_features()
        
        try:
            self.features = features_scaler.transform(self.features) if features_scaler else self.features

        except ValueError: ### triggered by Mixer which has been scaled already
            if self.config.debug:
                features_scaler = None
                print(f'[{self.set_id or self.case[0]}] [mixer_info] features_scaler was not applied as mixer_invariant_features_scaler was already applied')

        self.labels = labels_scaler.transform(self.labels) if labels_scaler else self.labels
        self.labels_eV = labels_eV_scaler.transform(self.labels_eV) if labels_eV_scaler else self.labels_eV
        self.tensor_features_linear = labels_eV_scaler.transform(self.tensor_features_linear) if labels_eV_scaler else self.tensor_features_linear

        if self.config.debug:
            applied_scalers = [
                scaler for scaler in [features_scaler, labels_scaler, labels_eV_scaler] 
                if scaler
            ]
            print(f'[{self.set_id or self.case[0]}] applied scalers {applied_scalers}')
    

    def _fit_mixer_scaler(self,
                          mixer_invariant_features_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        mixer_invariant_features_scaler.fit(self.features) if mixer_invariant_features_scaler else ...

        if self.config.debug:
            print(f'[{self.set_id or self.case[0]}] [mixer_info] fitted {mixer_invariant_features_scaler}')

        return mixer_invariant_features_scaler  


    def _scale_mixer(self,
                      mixer_invariant_features_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        self.features = mixer_invariant_features_scaler.transform(self.features) if mixer_invariant_features_scaler else self.features

        if self.config.debug:
            print(f'[{self.set_id or self.case[0]}] [mixer_info] applied {mixer_invariant_features_scaler}')


    def _build_mixer_features(self,
                              mixer_invariant_features_scaler :Union[StandardScaler, MinMaxScaler, None]):
       
        mixer_features = np.array(
            [
                [
                    [
                        0 for iii in range(3) ### 1 + 2 dimenions -> features Cx, Cy 
                    ] for ii in range(self.features.shape[1])
                ] for i in range(self.features.shape[0])
            ], dtype= np.float32
        )
        
        for i in range(self.features.shape[0]):           
            mixer_features[i,:,0] = mixer_invariant_features_scaler.transform(self.features[i].reshape(1,-1))[0] if mixer_invariant_features_scaler else self.features[i]
            mixer_features[i,:,1:] = np.tile(self.augmented_spatial_mixing_coords[i], (self.features.shape[1],1))

        self.features = mixer_features

        if self.config.debug:
            print(f'[{self.set_id or self.case[0]}] [mixer_info] building mixer features augmentend with spatial mixing with new shape: {self.features.shape}')


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


    def _export_for_stack(self):
        return (
            self.case,
            self.features,
            self.tensor_features,
            self.tensor_features_linear,
            self.labels,
            self.tensor_features_eV,
            self.labels_eV,
            self.Cx,
            self.Cy,
            self.u,
            self.v,
            self.augmented_spatial_mixing_coords
        )


    def _stack(self, *args):

        self.case.append(args[0])
        self.case = [_case if isinstance(_case, list) else [_case] for _case in self.case]
        self.case = sum(self.case, [])

        for arg, arg_value in zip(
            [
                'features',
                'tensor_features',
                'tensor_features_linear',
                'labels',
                'tensor_features_eV',
                'labels_eV',
                'Cx',
                'Cy',
                'u',
                'v',
                'augmented_spatial_mixing_coords',
            ],
            args[1:]
        ):
            updated_value = np.vstack((getattr(self, arg), arg_value))
            setattr(self, arg, updated_value)

        if self.config.debug:
            print(f'[{self.set_id}] sucessfully stacked case {args[0]} into {self.case[:-1]}')


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

    features_transforms = ['multi_sign_cbrt', 'same_sign_log']
    skip_features_transforms_for = ['I1_2', 'I1_5', 'I1_8','I1_15','I1_17', 'I1_19', 'q_1', 'q_2', 'q_3', 'q_4']

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
        features_cardinality = features_cardinality,
        tensor_features=tensor_features,
        tensor_features_linear=tensor_features_linear,
        labels='b',
        debug=True
    )
    all_features = standard_case_test_configuration.features_filter

    print('Standard case:')
    CaseSet(case, set_config=standard_case_test_configuration)
    print(f'All extracted features based on cardinality: {all_features}')

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
        features_transforms=features_transforms,
        skip_features_transforms_for=skip_features_transforms_for,
        debug=True
    )

    print('\nCustom turb dataset with features filter, no SHAT term and remove outliers, and features transforms:')
    CaseSet(case, set_config=extra_optional_case_test_configuration)