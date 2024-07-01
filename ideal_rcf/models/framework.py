try:
    from models.config import ModelConfig, MixerConfig
    from models.tbnn import TBNN
    from models.evnn import eVNN
    from models.oevnn import OeVNN

except ModuleNotFoundError:
    from config import ModelConfig, MixerConfig
    from tbnn import TBNN
    from evnn import eVNN
    from oevnn import OeVNN

from tensorflow.keras.layers import Input

class FrameWork(object):
    def __init__(self,
                 model_config :ModelConfig):
        
        if not isinstance(model_config, ModelConfig):
            raise AssertionError(f'[config_error] model_config must be of instance {ModelConfig()}')
        
        self.config = model_config


    def build(self,):
        ### need input layers here and pass them to build methods
        input_features_layer = Input(
            shape=self.config.features_input_shape,
            name='features_input_layer'
        )

        input_tensor_features_layer = Input(
            shape=self.config.tensor_features_input_shape,
            name='tensor_features_input_layer'
        )

        tbnn_model = TBNN(self.model_config).build(input_features_layer, input_tensor_features_layer)
        
        if self.config._evtbnn:
            input_tensor_features_linear_layer = Input(
                shape=self.config.tensor_features_linear_input_shape,
                name='tensor_features_evnn_input_layer'
            )

            evnn_model = eVNN(self.model_config).build(input_features_layer, input_tensor_features_linear_layer)
            ### add tomorrow
            ### support for evtbnn here
            # evtbnn_model = eVTBNN(tbnn_model, evnn_model)
        
        if self.config._oevnltbnn:
            input_tensor_features_oev_linear_layer = Input(
                shape=self.config.tensor_features_linear_oev_input_shape,
                name='tensor_features_oevnn_input_layer'
            )

            oevnn_model = OeVNN(self.model_config).build(input_features_layer, input_tensor_features_oev_linear_layer)
            ### add tomorrow 
            ### support for oevnltbnn here
        

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





