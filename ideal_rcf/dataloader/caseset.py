from ideal_rcf.dataloader.config import SetConfig

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from typing import List, Union, Optional
import numpy as np

class CaseSet(object):
    """
    A class for managing and processing a set of cases with various features and configurations.

    Methods
    -------
    __init__(case: str, set_config: SetConfig, set_id: Optional[str]=None, initialize_empty: Optional[bool]=False) -> None
        Initializes the CaseSet with the provided case, configuration, and optional settings.
    
    loadLabels(field: List[str])
        Loads labels for the specified field from numpy files.
    
    loadCombinedArray(field: List[str])
        Loads combined array data for the specified field from numpy files.
    
    loadCollumnStackFeatures(fields: List[str])
        Loads and stacks features for the specified fields.
    
    shuffle()
        Shuffles the features and labels.
    
    _filter_features()
        Filters the features based on the configuration.
    
    get_outliers_index()
        Identifies and returns the indices of outliers in the features.
    
    _remove_outliers()
        Removes outliers from the features and labels.
    
    _transform_features()
        Applies transformations to the features based on the configuration.
    
    _fit_scaler_oev(features_oev_scaler: Union[StandardScaler, MinMaxScaler, None], labels_oev_scaler: Union[StandardScaler, MinMaxScaler, None])
        Fits scalers to the features and labels for OEV (Operational Error Validation).
    
    _scale_oev(features_oev_scaler: Union[StandardScaler, MinMaxScaler, None], labels_oev_scaler: Union[StandardScaler, MinMaxScaler, None])
        Scales the features and labels for OEV.
    
    _fit_mixer_scaler(mixer_invariant_features_scaler: Union[StandardScaler, MinMaxScaler, None])
        Fits a scaler to the mixer invariant features.
    
    _fit_scaler(features_scaler: Union[StandardScaler, MinMaxScaler, None], labels_scaler: Union[StandardScaler, MinMaxScaler, None], mixer_invariant_features_scaler: Union[StandardScaler, MinMaxScaler, None])
        Fits scalers to the features and labels, and optionally the mixer invariant features.
    
    _scale(features_scaler: Union[StandardScaler, MinMaxScaler, None], labels_scaler: Union[StandardScaler, MinMaxScaler, None])
        Scales the features and labels.
    
    _build_mixer_features(mixer_invariant_features_scaler: Union[StandardScaler, MinMaxScaler, None])
        Builds mixer features augmented with spatial mixing coordinates.
    
    _ensure_oev_shapes()
        Ensures that the shapes of the labels and tensor features for OEV are compatible.
    
    _export_for_stack()
        Exports the case set data for stacking.
    
    _import_from_copy(*args)
        Imports data from a copy for various attributes.
    
    _stack(*args)
        Stacks a new case's data into the current case set.
    
    check_set()
        Checks the consistency and dimensions of the set attributes.
    """
    def __init__(self,
                 case :str,
                 set_config :SetConfig,
                 set_id :Optional[str]=None,
                 initialize_empty :Optional[bool]=False) -> None:
        
        if not isinstance(set_config, SetConfig):
            raise AssertionError(f'[config_error] set_config must be of instance {SetConfig()}')
        
        self.config = set_config

        self.set_id = set_id

        self.case = self.config.ensure_list_instance(case)
        
        self.features = self.loadCollumnStackFeatures(self.config.features)

        self.tensor_features = self.loadCombinedArray(self.config.tensor_features)
        self.tensor_features_linear = self.loadCombinedArray(self.config.tensor_features_linear)
        self.labels = self.loadLabels(self.config.labels)

        self.tensor_features_oev = self.loadCombinedArray(self.config.tensor_features_oev)
        self._ensure_oev_shapes()

        self.Cx = self.loadCombinedArray(self.config.Cx)
        self.Cy = self.loadCombinedArray(self.config.Cy)

        self.u = self.loadLabels(self.config.u)
        self.v = self.loadLabels(self.config.v)

        self.predictions = None
        self.predictions_oev = None
        self.labels_compiled = False

        if not initialize_empty:
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
        """
        Method based on code provided by McConkey *et al.* in [A curated dataset for data-driven
        turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500)
        """        
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
        """
        Method based on code provided by McConkey *et al.* in [A curated dataset for data-driven
        turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500)
        """        
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


    def shuffle(self):
            (
                self.features,
                self.tensor_features,
                self.tensor_features_linear,
                self.labels,
                self.tensor_features_oev,
                self.Cx,
                self.Cy,
                self.u,
                self.v,
                # self.augmented_spatial_mixing_coords
            ) = shuffle(
                self.features,
                self.tensor_features,
                self.tensor_features_linear,
                self.labels,
                self.tensor_features_oev,
                self.Cx,
                self.Cy,
                self.u,
                self.v,
                # self.augmented_spatial_mixing_coords,
                random_state=self.config.random_seed
            )


    def _filter_features(self,):
        indexes_union = [self.config.all_features.index(feature) for feature in self.config.features_filter]
        if self.config.debug:
            assert len(indexes_union) == len(self.config.features_filter)
            print(f'[{self.set_id or self.case[0]}] sucessfuly filtered features {self.config.features} to {self.config.features_filter}')

        self.features = self.features[:, indexes_union]


    def get_outliers_index(self):
        """
        Method based on code provided by McConkey *et al.* in [A curated dataset for data-driven
        turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500)
        """ 
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
        """
        Method based on code provided by McConkey *et al.* in [A curated dataset for data-driven
        turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500)
        """ 
        outliers_index = self.get_outliers_index()
        if self.config.debug:
            print(f'[{self.set_id or self.case[0]}] Found {len(outliers_index)} outliers in {self.config.features} feature set')

        self.features = np.delete(self.features, outliers_index, axis=0)
        self.tensor_features = np.delete(self.tensor_features, outliers_index, axis=0)
        self.tensor_features_linear = np.delete(self.tensor_features_linear, outliers_index, axis=0) if self.config.tensor_features_linear else None
        self.labels = np.delete(self.labels, outliers_index, axis=0)
        
        self.tensor_features_oev = np.delete(self.tensor_features_oev, outliers_index, axis=0) if self.config.tensor_features_oev else None
        
        self.Cx = np.delete(self.Cx, outliers_index, axis=0)
        self.Cy = np.delete(self.Cy, outliers_index, axis=0)

        self.u = np.delete(self.u, outliers_index, axis=0)
        self.v = np.delete(self.v, outliers_index, axis=0)

    
    def _transform_features(self):
        ### must be applied after features_filter
        print(f'[{self.set_id or self.case[0]}]') if self.config.debug else ...
        for i, feature in enumerate(self.config.features_filter):
            if feature in self.config.skip_features_transforms_for:
                continue            
            print(f'[transforms] {feature}:') if self.config.debug else ...
            for transform in self.config.features_transforms:
                self.features[:,i] = transform(self.features[:,i], self.config.debug)


    def _fit_scaler_oev(self,
                        features_oev_scaler :Union[StandardScaler, MinMaxScaler, None],
                        labels_oev_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        features_oev_scaler.fit(self.features) if features_oev_scaler else ...
        try: 
            bool(self.tensor_features_oev);
        
        except ValueError:         
            labels_oev_scaler.fit(self.labels) if labels_oev_scaler else ...

        return features_oev_scaler, labels_oev_scaler


    def _scale_oev(self,
                   features_oev_scaler :Union[StandardScaler, MinMaxScaler, None],
                   labels_oev_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        self.features = features_oev_scaler.transform(self.features) if features_oev_scaler else self.features

        try: 
           bool(self.tensor_features_oev);
        
        except ValueError:
            try:
                bool(self.labels);
            except ValueError:
                self.labels = labels_oev_scaler.transform(self.labels) if labels_oev_scaler else self.labels
            self.tensor_features_oev = labels_oev_scaler.transform(self.tensor_features_oev) if labels_oev_scaler else self.tensor_features_oev

        # return [scaled_features, scaled_tensor_features_oev], scaled_labels


    def _fit_mixer_scaler(self,
                          mixer_invariant_features_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        mixer_invariant_features_scaler.fit(self.features) if mixer_invariant_features_scaler else ...

        if self.config.debug:
            print(f'[{self.set_id or self.case[0]}] [mixer_info] fitted {mixer_invariant_features_scaler}')

        return mixer_invariant_features_scaler  


    def _fit_scaler(self,
                    features_scaler :Union[StandardScaler, MinMaxScaler, None],
                    labels_scaler :Union[StandardScaler, MinMaxScaler, None],
                    mixer_invariant_features_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        features_scaler.fit(self.features) if features_scaler else ...
        labels_scaler.fit(self.labels) if labels_scaler else ...
        mixer_invariant_features_scaler.fit(self.features) if mixer_invariant_features_scaler else ...
        
        if self.config.debug:
            applied_scalers = [
                scaler for scaler in [features_scaler, labels_scaler, mixer_invariant_features_scaler] 
                if scaler
            ]
            print(f'[{self.set_id or self.case[0]}] fitted scalers {applied_scalers}')

        return features_scaler, labels_scaler, mixer_invariant_features_scaler


    def _scale(self,
               features_scaler :Union[StandardScaler, MinMaxScaler, None],
               labels_scaler :Union[StandardScaler, MinMaxScaler, None]):
        
        self.features = features_scaler.transform(self.features) if features_scaler and not self.config.enable_mixer else self.features
        try:
            bool(self.labels);
        except ValueError:
            self.labels = labels_scaler.transform(self.labels) if labels_scaler else self.labels


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


    def _ensure_oev_shapes(self):
        if bool(self.config.labels) and bool(self.config.tensor_features_oev):
            if self.config.tensor_features_oev:
                tensor_shape = self.tensor_features_oev.shape[1]
                if self.labels.shape[1] > tensor_shape:
                    if self.config.debug:
                        print('')
                    self.labels = self.labels[:,:tensor_shape]

                elif self.labels_oev.shape[1] < tensor_shape:
                    raise ValueError(f'[{self.set_id or self.case[0]}] Config_Error: labels ({self.config.labels}) and tensor_features_oev ({self.config.tensor_features_oev} must have same dim 1 but have ({self.labels.shape[1] }) and ({tensor_shape})')

        # else:
        #     raise AssertionError(f'[{self.set_id or self.case[0]}] Config_Error: labels ({self.config.labels}) and tensor_features_oev ({self.config.tensor_features_oev} must be passed simultaneously)')


    def _export_for_stack(self):
        return (
            self.case,
            self.features,
            self.tensor_features,
            self.tensor_features_linear,
            self.labels,
            self.tensor_features_oev,
            # self.labels_oev,
            self.Cx,
            self.Cy,
            self.u,
            self.v,
            self.augmented_spatial_mixing_coords
        )


    def _import_from_copy(self,
                          *args):
    
        for arg, arg_value in zip(
            [
                'features',
                'tensor_features',
                'tensor_features_linear',
                'labels',
                'tensor_features_oev',
                'Cx',
                'Cy',
                'u',
                'v',
                'augmented_spatial_mixing_coords',
            ],
            args[1:]
        ):
            setattr(self, arg, arg_value)


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
                'tensor_features_oev',
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
            'tensor_features_oev',
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