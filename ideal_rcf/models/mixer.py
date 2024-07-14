try:
    from ideal_rcf.models.config import MixerConfig

except ModuleNotFoundError:
    from config import MixerConfig

from tensorflow.keras.layers import LayerNormalization, BatchNormalization, Dense, Dropout
import tensorflow as tf


class MixerResBlock(object):
    """
    based on [tsmixer](https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/tsmixer.py)
    """
    def __init__(self,
                 mixer_config :MixerConfig) -> None:
        
        if not isinstance(mixer_config, MixerConfig):
            raise AssertionError(f'[config_error] mixer_config must be of instance {MixerConfig()}')
        
        self.config = mixer_config


    def layers(self, inputs):
        norm = (
            LayerNormalization
            if self.config.normalization == 'L'
            else BatchNormalization
        )

        ### Temporal Linear
        x = norm(axis=[-2, -1])(inputs)
        x = tf.transpose(x, perm=[0, 2, 1])  ### [Batch, Channel, Input Length]
        x = Dense(
            x.shape[-1], 
            kernel_initializer=self.config.initializer, 
            kernel_regularizer=self.config.regularizer, 
            activation=self.config.activations)(x)
        x = tf.transpose(x, perm=[0, 2, 1])  ### [Batch, Input Length, Channel]
        x = Dropout(self.config.dropout)(x)
        res = x + inputs

        #### Feature Linear
        x = norm(axis=[-2, -1])(res)
        for _ in range(self.config.features_mlp_layers):
            x = Dense(
                self.config.features_mlp_units,
                kernel_initializer=self.config.initializer,
                kernel_regularizer=self.config.regularizer,
                activation=self.config.activations
            )(x)  ### [Batch, Input Length, FF_Dim]
        
        x = Dropout(self.config.dropout)(x)
        x = Dense(inputs.shape[-1])(x)  ### [Batch, Input Length, Channel]
        x = Dropout(self.config.dropout)(x)

        return x + res