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
                 layers_tbnn :int,
                 units_tbnn :int, 
                 features_input_shape :Tuple[int], 
                 tensor_features_input_shape :Tuple[int],
                 layers_evtbnn :Optional[int]=None,
                 units_evtbnn :Optional[int]=None,                  
                 tensor_features_linear_input_shape :Optional[Union[Tuple[int],None]]=None,
                 layers_evnn :Optional[int]=None,
                 units_evnn :Optional[int]=None,
                 tensor_features_linear_eV_input_shape :Optional[Union[Tuple[int],None]]=None,                 
                 tbnn_mixer_config :Optional[Union[MixerConfig, None]]=None,
                 evtbnn_mixer_config :Optional[Union[MixerConfig, None]]=None,
                 evnn_mixer_config :Optional[Union[MixerConfig, None]]=None,
                 optimizer :Optional[Any]=Adam,
                 loss :Optional[Any]=Huber(),
                 learning_rate :Optional[int]=None,
                 batch :Optional[int]=None,
                 epochs :Optional[int]=None,
                 initializer :Optional[Union[Any, None]]=LecunNormal(seed=0),
                 regularizer :Optional[Union[Any, None]]=L2(1e-8),
                 tbnn_activations :Optional[Union[str, Any]]='selu',
                 evtbnn_activations :Optional[Union[str, Any]]='selu',
                 evnn_activations :Optional[Union[str, Any]]='selu',
                 eV_activation :Optional[Union[str,Any]]='exponential',
                 metrics :Optional[List[Union[str, Any]]]=['mse', 'mae'],
                 keras_callbacks :Optional[Union[List[Union[ReduceLROnPlateau, EarlyStopping, Any]], None]]=None,
                 model_id :Optional[str]=None,
                 random_seed :Optional[int]=42,
                 debug :Optional[bool]=False,
                 ) -> None:

        self.layers_tbnn = self.ensure_int_instance(layers_tbnn)
        self.units_tbnn = self.ensure_int_instance(units_tbnn)
        self.features_input_shape = self.ensure_is_instance(features_input_shape, tuple)
        self.tensor_features_input_shape = self.ensure_is_instance(tensor_features_input_shape, tuple)
        
        self.layers_evtbnn = self.ensure_int_instance(layers_evtbnn)
        self.units_evtbnn = self.ensure_int_instance(units_evtbnn)
        self.tensor_features_linear_input_shape = self.ensure_is_instance(tensor_features_linear_input_shape, tuple)
        
        self.layers_evnn = self.ensure_int_instance(layers_evnn)
        self.units_evnn = self.ensure_int_instance(units_evnn)
        self.tensor_features_linear_eV_input_shape = self.ensure_is_instance(tensor_features_linear_eV_input_shape, tuple)

        self.learning_rate = self.ensure_int_instance(learning_rate)
        self.batch = self.ensure_int_instance(batch)
        self.epochs = self.ensure_int_instance(epochs)
        
        self.loss = loss
        self.optimizer = optimizer
        self.initializer = initializer
        self.regularizer = regularizer
        
        self.tbnn_activations = tbnn_activations
        self.ensure_attr_group(['layers_tbnn', 'units_tbnn', 'features_input_shape', 'tensor_features_input_shape', 'tbnn_activations'])

        self.evtbnn_activations = evtbnn_activations
        self.ensure_attr_group(['layers_evtbnn', 'units_evtbnn', 'tensor_features_linear_input_shape'])
        self._evtbnn = True if self.layers_evtbnn else False ### used to trigger between tbnn and evtbnn in framework

        self.evnn_activations = evnn_activations
        self.eV_activation = eV_activation
        self.ensure_attr_group(['layers_evnn', 'units_evnn', 'tensor_features_linear_eV_input_shape'])
        self._oevnltbnn = True if self.layers_evnn else False ### used to activate oevnltbnn in framework

        self.metrics = self.ensure_list_instance(metrics)
        self.keras_callbacks = self.ensure_list_instance(keras_callbacks)

        self.model_id = self.ensure_str_instance(model_id) if model_id else ''
        self.random_seed = self.ensure_int_instance(random_seed)

        self.tbnn_mixer_config = self.ensure_is_instance(tbnn_mixer_config, MixerConfig) 
        self.evtbnn_mixer_config = self.ensure_is_instance(evtbnn_mixer_config, MixerConfig) 
        self.ensure_attr_group(['tbnn_mixer_config', 'evtbnn_mixer_config'])if self._evtbnn else ...
        self.evnn_mixer_config = self.ensure_is_instance(evnn_mixer_config, MixerConfig) 
        self.ensure_attr_group(['tbnn_mixer_config', 'evtbnn_mixer_config', 'evnn_mixer_config'])if self._oevnltbnn else ...
        
        self.debug = debug


if __name__ == '__main__':
    layers_tbnn = 3
    units_tbnn = 150
    features_input_shape = (15,3)
    tensor_features_input_shape = (20,3,3)

    layers_evtbnn = 2
    units_evtbnn = 150
    tensor_features_linear_input_shape = (3,)

    layers_evnn = 2
    units_evnn = 150
    tensor_features_linear_eV_input_shape = (3,)

    tbnn_mixer_config = MixerConfig(
        features_mlp_layers=5,
        features_mlp_units=150
    )

    evtbnn_mixer_config = MixerConfig(
        features_mlp_layers=3,
        features_mlp_units=150
    )

    evnn_mixer_config = MixerConfig(
        features_mlp_layers=5,
        features_mlp_units=150
    )

    TBNN_config = ModelConfig(
        layers_tbnn=layers_tbnn,
        units_tbnn=units_tbnn,
        features_input_shape=features_input_shape,
        tensor_features_input_shape=tensor_features_input_shape,
        tbnn_mixer_config=tbnn_mixer_config
    )
    assert TBNN_config._evtbnn == False
    assert TBNN_config._oevnltbnn == False
    print('Sucess creating mixer TBNN ModelConfig obj')
    
    eVTBNN_config = ModelConfig(
        layers_tbnn=layers_tbnn,
        units_tbnn=units_tbnn,
        features_input_shape=features_input_shape,
        tensor_features_input_shape=tensor_features_input_shape,
        layers_evtbnn=layers_evtbnn,
        units_evtbnn=units_evtbnn,
        tensor_features_linear_input_shape=tensor_features_linear_input_shape,
    )
    assert eVTBNN_config._evtbnn == True
    assert eVTBNN_config._oevnltbnn == False
    print('Sucess creating eVTBNN_config ModelConfig obj')

    OeVNLTBNN_config = ModelConfig(
        layers_tbnn=layers_tbnn,
        units_tbnn=units_tbnn,
        features_input_shape=features_input_shape,
        tensor_features_input_shape=tensor_features_input_shape,
        layers_evtbnn=layers_evtbnn,
        units_evtbnn=units_evtbnn,
        tensor_features_linear_input_shape=tensor_features_linear_input_shape,
        layers_evnn=layers_evnn,
        units_evnn=units_evnn,
        tensor_features_linear_eV_input_shape=tensor_features_linear_eV_input_shape,
        tbnn_mixer_config=tbnn_mixer_config,
        evtbnn_mixer_config=evtbnn_mixer_config,
        evnn_mixer_config=evnn_mixer_config
    )
    assert OeVNLTBNN_config._evtbnn == True
    assert OeVNLTBNN_config._oevnltbnn == True
    print('Sucess creating mixer OeVNLTBNN_config ModelConfig obj')





