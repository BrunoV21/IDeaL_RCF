try:
    from ideal_rcf.models.base_model import BaseModel
    from ideal_rcf.models.config import ModelConfig
    from ideal_rcf.models.mixer import MixerResBlock

except ModuleNotFoundError:
    from base_model import BaseModel
    from config import ModelConfig
    from mixer import MixerResBlock

from tensorflow.keras.layers import Input, Dense, Reshape, Dot
from tensorflow.keras import Model

class TBNN(BaseModel):
    def __init__(self, 
                 model_config: ModelConfig) -> None:
        
        super().__init__(model_config)

        self.HiddenProcessing = MixerResBlock(self.config.tbnn_mixer_config).layers \
            if self.config.tbnn_mixer_config \
            else self.Dense


    def Dense(self, x):
        x = Dense(
                self.config.units_tbnn,
                kernel_initializer=self.config.initializer,
                kernel_regularizer=self.config.regularizer, 
                activation = self.config.tbnn_activations
        )(x)

        return x


    def build(self,
              input_features_layer :Input,
              input_tensor_features_layer :Input):
        
        hidden = input_features_layer
        for i in range(self.config.layers_tbnn):
            hidden = self.HiddenProcessing(hidden)

        if self.config.tbnn_mixer_config:
            hidden = hidden[:,:,0] ### Invariant Selection

        output = Dense(
            self.config.tensor_features_input_shape[0],
            kernel_initializer=self.config.initializer,
            kernel_regularizer=self.config.regularizer, 
            activation = self.config.tbnn_activations
            )(hidden)
        
        shaped_output = Reshape((self.config.tensor_features_input_shape[0],1,1))(output)
        anisotropy = Dot(axes=1)([shaped_output, input_tensor_features_layer])
        reshaped_anisotropy = Reshape((9,1))(anisotropy)

        model = Model(
            inputs=[
                input_features_layer,
                input_tensor_features_layer
            ],
            outputs=[
                reshaped_anisotropy
            ]
        )

        model._name = 'mixer_tbnn' if self.config.tbnn_mixer_config else 'tbnn'

        return model