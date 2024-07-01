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

class eVNN(BaseModel):
    def __init__(self, 
                 model_config: ModelConfig) -> None:
        
        super().__init__(model_config)

        self.HiddenProcessing = MixerResBlock(self.config.evnn_mixer_config) \
            if self.config.tbnn_mixer_config \
            else \
            Dense(
                self.config.units_evnn,
                kernel_initializer=self.config.initializer,
                kernel_regularizer=self.config.regularizer, 
                activation = self.config.evnn_activations
            )


    def build(self,
              input_features_layer :Input,
              input_tensor_features_linear_layer :Input):
        
        hidden = input_features_layer 
        for i in range(self.config.layers_evnn):
            hidden = self.HiddenProcessing(hidden)

        if self.config.evnn_mixer_config:
            hidden = hidden[:,:,0] ### Invariant Selection

        output = Dense(
            1,
            kernel_initializer=self.config.initializer,
            kernel_regularizer=self.config.regularizer, 
            activation = self.config.eV_activation,
            )(hidden)
        
        effective_viscosity = Multiply()([output, input_tensor_features_linear_layer])
        reshaped_effective_viscosity = Reshape((self.config.tensor_features_linear_eV_input_shape,1))(effective_viscosity)

        model = Model(
            inputs=[
                input_features_layer,
                input_tensor_features_linear_layer
            ],
            outputs=[
                reshaped_effective_viscosity
            ]
        )

        model._name = 'mixer_evnn' if self.config.evnn_mixer_config else 'evnn'

        return model