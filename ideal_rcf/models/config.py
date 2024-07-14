from typing import Optional, Union, List, Any, Tuple
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.initializers import LecunNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


class BaseConfig(object):
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


    def ensure_int_instance(self, attribute):
        if isinstance(attribute, int) or not attribute:
            return attribute
        
        else:
            raise TypeError(f'{attribute} must be int but got {type(attribute)}')
        
    
    def ensure_is_instance(self, incoming, instance):
        if isinstance(incoming, instance) or not incoming:
            return incoming
        
        else:
            raise TypeError(f'{incoming} must be instance of {instance}')


    def ensure_attr_group(self, attr_group):
        passed_attr = []
        empty_attr = []
        for attr in attr_group:
            value = getattr(self, attr)
            passed_attr.append(attr) if value else empty_attr.append(attr)

        if not passed_attr or not empty_attr:
            return
        
        else:
            raise AttributeError(f'{passed_attr} have been passed, to ensure functional beahaviour also pass {empty_attr}')


class MixerConfig(BaseConfig):
    def __init__(self,
                 features_mlp_layers :int,
                 features_mlp_units :int,
                 normalization :Optional[str]='L',
                 dropout :Optional[int]=0,
                 initializer :Optional[Union[Any, None]]=LecunNormal(seed=0),
                 regularizer :Optional[Union[Any, None]]=L2(1e-8),
                 activations :Optional[Union[str, Any]]='selu'
                 ) -> None:
        ### initializer, regularizer and activations are taken from main nn
        self.features_mlp_layers = self.ensure_int_instance(features_mlp_layers)
        self.features_mlp_units = self.ensure_int_instance(features_mlp_units)
        self.dropout = self.ensure_int_instance(dropout)
        self.normalization = normalization
        self.initializer = initializer
        self.regularizer = regularizer
        self.activations = activations


class ModelConfig(BaseConfig):
    def __init__(self,
                 layers_tbnn :Optional[int]=None,
                 units_tbnn :Optional[int]=None, 
                 features_input_shape :Optional[Tuple[int]]=None, 
                 tensor_features_input_shape :Optional[Tuple[int]]=None,
                 layers_evnn :Optional[int]=None,
                 units_evnn :Optional[int]=None,                  
                 tensor_features_linear_input_shape :Optional[Union[Tuple[int],None]]=None,
                 layers_oevnn :Optional[int]=None,
                 units_oevnn :Optional[int]=None,
                 tensor_features_linear_oev_input_shape :Optional[Union[Tuple[int],None]]=None,                 
                 tbnn_mixer_config :Optional[Union[MixerConfig, None]]=None,
                 evnn_mixer_config :Optional[Union[MixerConfig, None]]=None,
                 oevnn_mixer_config :Optional[Union[MixerConfig, None]]=None,
                 optimizer :Optional[Any]=Adam,
                 regress_nl_labels :Optional[bool]=True,
                 loss :Optional[Any]=Huber(),
                 learning_rate :Optional[int]=None,
                 learning_rate_oevnn :Optional[int]=None,
                 batch :Optional[int]=None,
                 epochs :Optional[int]=None,
                 initializer :Optional[Union[Any, None]]=LecunNormal(seed=0),
                 regularizer :Optional[Union[Any, None]]=L2(1e-8),
                 tbnn_activations :Optional[Union[str, Any]]='selu',
                 evnn_activations :Optional[Union[str, Any]]='selu',
                 oevnn_activations :Optional[Union[str, Any]]='selu',
                 eV_activation :Optional[Union[str,Any]]='exponential',
                 metrics :Optional[List[Union[str, Any]]]=['mse', 'mae'],
                 keras_callbacks :Optional[Union[List[Union[ReduceLROnPlateau, EarlyStopping, Any]], None]]=None,
                 model_id :Optional[str]=None,
                 random_seed :Optional[int]=42,
                 verbose :Optional[int]=1,
                 shuffle :Optional[bool]=True,
                 debug :Optional[bool]=False,
                 ) -> None:

        self.layers_tbnn = self.ensure_int_instance(layers_tbnn)
        self.units_tbnn = self.ensure_int_instance(units_tbnn)
        try:
            self.features_input_shape = int(features_input_shape)
        except TypeError:
            self.features_input_shape = self.ensure_is_instance(features_input_shape,  tuple)
        self.tensor_features_input_shape = self.ensure_is_instance(tensor_features_input_shape, tuple)
        
        self.layers_evnn = self.ensure_int_instance(layers_evnn)
        self.units_evnn = self.ensure_int_instance(units_evnn)
        self.tensor_features_linear_input_shape = self.ensure_is_instance(tensor_features_linear_input_shape, tuple)
        
        self.layers_oevnn = self.ensure_int_instance(layers_oevnn)
        self.units_oevnn = self.ensure_int_instance(units_oevnn)
        self.tensor_features_linear_oev_input_shape = self.ensure_is_instance(tensor_features_linear_oev_input_shape, tuple)

        self.learning_rate = learning_rate
        self.batch = self.ensure_int_instance(batch)
        self.epochs = self.ensure_int_instance(epochs)
        self.learning_rate_oevnn = learning_rate_oevnn if learning_rate_oevnn else self.learning_rate
        
        self.regress_nl_labels = regress_nl_labels

        self.loss = loss
        self.optimizer = optimizer
        self.initializer = initializer
        self.regularizer = regularizer
        
        self.tbnn_activations = tbnn_activations
        self.ensure_attr_group(['layers_tbnn', 'units_tbnn', 'features_input_shape', 'tensor_features_input_shape'])

        self.evnn_activations = evnn_activations
        self.ensure_attr_group(['layers_evnn', 'units_evnn', 'tensor_features_linear_input_shape'])
        self._evtbnn = True if self.layers_evnn else False ### used to trigger between tbnn and evnn in framework

        self.oevnn_activations = oevnn_activations
        self.eV_activation = eV_activation
        self.ensure_attr_group(['layers_oevnn', 'units_oevnn', 'tensor_features_linear_oev_input_shape'])
        self._oevnltbnn = True if self.layers_oevnn else False ### used to activate oevnltbnn in framework

        self.metrics = self.ensure_list_instance(metrics)
        self.keras_callbacks = self.ensure_list_instance(keras_callbacks)

        self.model_id = self.ensure_str_instance(model_id) if model_id else ''
        self.verbose = self.ensure_int_instance(verbose)
        self.random_seed = self.ensure_int_instance(random_seed)

        self.tbnn_mixer_config = self.ensure_is_instance(tbnn_mixer_config, MixerConfig) 
        self.evnn_mixer_config = self.ensure_is_instance(evnn_mixer_config, MixerConfig) 
        self.ensure_attr_group(['tbnn_mixer_config', 'evnn_mixer_config'])if self._evtbnn else ...
        self.oevnn_mixer_config = self.ensure_is_instance(oevnn_mixer_config, MixerConfig) 
        self.ensure_attr_group(['tbnn_mixer_config', 'evnn_mixer_config', 'oevnn_mixer_config'])if self._oevnltbnn else ...
        
        try:
            len(self.features_input_shape)
            if not self.tbnn_mixer_config:
                raise ValueError(f'when mixer config is not passed features_input_shape should have len = 1, but got {self.features_input_shape}') 
        except TypeError:
             if self.tbnn_mixer_config:
                raise ValueError(f'when mixer config is passed features_input_shape should have len >= 1, but got {self.features_input_shape}') 
        
        self.shuffle = shuffle
        self.debug = debug