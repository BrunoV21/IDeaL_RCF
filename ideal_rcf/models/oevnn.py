try:
    from models.base_model import BaseModel
    from models.config import ModelConfig
    from models.mixer import MixerResBlock

except ModuleNotFoundError:
    from base_model import BaseModel
    from config import ModelConfig
    from mixer import MixerResBlock

from tensorflow.keras.layers import Input, Dense, Reshape, Multiply
from tensorflow.keras import Model

class OeVNN(BaseModel):
    def __init__(self, 
                 model_config: ModelConfig) -> None:
        
        super().__init__(model_config)

        self.HiddenProcessing = MixerResBlock(self.config.oevnn_mixer_config).layers \
            if self.config.oevnn_mixer_config \
            else \
            Dense(
                self.config.units_oevnn,
                kernel_initializer=self.config.initializer,
                kernel_regularizer=self.config.regularizer, 
                activation = self.config.oevnn_activations
            )


    def build(self,
              input_features_layer :Input,
              input_tensor_features_oev_linear_layer :Input):
        
        hidden = input_features_layer 
        for i in range(self.config.layers_oevnn):
            hidden = self.HiddenProcessing(hidden)

        if self.config.oevnn_mixer_config:
            hidden = hidden[:,:,0] ### Invariant Selection

        output = Dense(
            1,
            kernel_initializer=self.config.initializer,
            kernel_regularizer=self.config.regularizer, 
            activation = self.config.eV_activation,
            )(hidden)
        
        output = -2*output
        
        optimal_viscosity = Multiply()([output, input_tensor_features_oev_linear_layer])
        reshaped_optimal_viscosity = Reshape((self.config.tensor_features_linear_oev_input_shape[0],1))(optimal_viscosity)

        model = Model(
            inputs=[
                input_features_layer,
                input_tensor_features_oev_linear_layer
            ],
            outputs=[
                reshaped_optimal_viscosity
            ]
        )

        model._name ='mixer_oevnn' if self.config.oevnn_mixer_config else 'oevnn'

        return model