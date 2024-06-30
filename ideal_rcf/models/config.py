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
        if isinstance(attribute, int):
            return attribute
        
        else:
            raise TypeError(f'{attribute} must be int but got {type(attribute)}')
        
    
    def ensure_is_instance(self, incoming, instance):
        if isinstance(incoming, instance) or not incoming:
            return incoming
        
        else:
            raise TypeError(f'{incoming} must be instance of {instance}')


    #### create functon that receives list of attribuytes and checks if they are all not none
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
    def __init__(self,):
        ...

        
        


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
                 weights_initializer :Optional[Union[Any, None]]=LecunNormal(seed=0),
                 regularizer :Optional[Union[Any, None]]=L2(1e-8),
                 tbnn_activations :Optional[Union[str, Any]]='seliu',                 
                 evtbnn_activations :Optional[Union[str, Any]]='seliu',                 
                 evnn_activations :Optional[Union[str, Any]]='seliu',
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
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer
        
        self.tbnn_activations = tbnn_activations
        self.ensure_attr_group(['layers_tbnn', 'units_tbnn', 'features_input_shape', 'tensor_features_input_shape', 'tbnn_activations'])
        
        self.evtbnn_activations = evtbnn_activations
        self.ensure_attr_group(['layers_evtbnn', 'units_evtbnn', 'tensor_features_linear_input_shape', 'evtbnn_activations'])

        self.evnn_activations = evnn_activations
        self.eV_activation = eV_activation
        self.ensure_attr_group(['layers_evnn', 'units_evnn', 'tensor_features_linear_eV_input_shape', 'evnn_activations', 'eV_activation'])

        self.metrics = self.ensure_list_instance(metrics)
        self.keras_callbacks = self.ensure_list_instance(keras_callbacks)

        self.model_id = self.ensure_str_instance(model_id) if model_id else ''
        self.random_seed = self.ensure_int_instance(random_seed)

        self.tbnn_mixer_config = self.ensure_is_instance(tbnn_mixer_config, MixerConfig) 
        self.evtbnn_mixer_config = self.ensure_is_instance(evtbnn_mixer_config, MixerConfig) 
        self.evnn_mixer_config = self.ensure_is_instance(evnn_mixer_config, MixerConfig) 
        
        self.debug = debug

if __name__ == '__main__':
    print('Sucess')





