from types import SimpleNamespace

try:
    from models.config import ModelConfig, MixerConfig
    from models.tbnn import TBNN
    from models.evnn import eVNN
    from models.oevnn import OeVNN
    from dataloader.dataset import DataSet
    from dataloader.caseset import CaseSet

except ModuleNotFoundError:
    from config import ModelConfig, MixerConfig
    from tbnn import TBNN
    from evnn import eVNN
    from oevnn import OeVNN

    DataSet = SimpleNamespace()
    DataSet.labels_scaler = None
    DataSet.labels_eV_scaler = None

    CaseSet = SimpleNamespace()

from tensorflow.keras.layers import Input, Lambda, Add, Concatenate
from tensorflow.keras import Model
from typing import Optional
import tensorflow as tf
import polars as pl

class FrameWork(object):
    def __init__(self,
                 model_config :ModelConfig):
        
        if not isinstance(model_config, ModelConfig):
            raise AssertionError(f'[config_error] model_config must be of instance {ModelConfig()}')
        
        self.config = model_config

        tf.random.set_seed(42)

        self.models = SimpleNamespace()
        self.history = SimpleNamespace()
        self.build()


    def build(self,):

        model = {}
        ### need input layers here and pass them to build methods
        input_features_layer = Input(
            shape=(self.config.features_input_shape) if type(self.config.features_input_shape)==int else self.config.features_input_shape,
            name='features_input_layer'
        )

        input_tensor_features_layer = Input(
            shape=self.config.tensor_features_input_shape,
            name='tensor_features_input_layer'
        )

        tbnn_model = TBNN(self.config).build(input_features_layer, input_tensor_features_layer)
        tbnn_output = tbnn_model([
            input_features_layer,
            input_tensor_features_layer
        ])

        tbnn_output_0 = Lambda(lambda x: x[:,0])(tbnn_output)
        tbnn_output_1 = Lambda(lambda x: x[:,1])(tbnn_output)
        tbnn_output_4 = Lambda(lambda x: x[:,4])(tbnn_output)


        if self.config._evtbnn: 
            input_tensor_features_linear_layer = Input(
                shape=self.config.tensor_features_linear_input_shape,
                name='tensor_features_evnn_input_layer'
            )

            evnn_model = eVNN(self.config).build(input_features_layer, input_tensor_features_linear_layer)
            evnn_output = evnn_model([
                input_features_layer,
                input_tensor_features_linear_layer
            ])

            evnn_output_0 = Lambda(lambda x: -x[:,0])(evnn_output)
            evnn_output_1 = Lambda(lambda x: -x[:,1])(evnn_output)
            evnn_output_4 = Lambda(lambda x: -x[:,2])(evnn_output)

            evtbnn_output_0 = Add()([tbnn_output_0, evnn_output_0])            
            evtbnn_output_1 = Add()([tbnn_output_1, evnn_output_1])
            evtbnn_output_4 = Add()([tbnn_output_4, evnn_output_4])

            evtbnn_output_6 = Add()([evtbnn_output_0, evtbnn_output_4])

            merged_output = Concatenate()([
                evtbnn_output_0,
                evtbnn_output_1,
                evtbnn_output_4,
                tf.math.negative(evtbnn_output_6)
            ])

            evtbnn = Model(
                inputs=[
                    input_features_layer,
                    input_tensor_features_layer,
                    input_tensor_features_linear_layer
                ],
                outputs=[
                    merged_output
                ]
            )
            evtbnn._name = 'evtbnn_framework'
            self.models.evtbnn=evtbnn

        else:
            tbnn_output_6 = Add()([tbnn_output_0, tbnn_output_4])

            merged_output = Concatenate()([
                tbnn_output_0,
                tbnn_output_1,
                tbnn_output_4,
                tf.math.negative(tbnn_output_6)
            ])

            tbnn = Model(
                inputs=[
                    input_features_layer,
                    input_tensor_features_layer,
                ],
                outputs=[
                    merged_output
                ]
            )

            tbnn._name = 'tbnn_framework'
            self.models.tbnn=tbnn

        
        if self.config._oevnltbnn:
            input_tensor_features_oev_linear_layer = Input(
                shape=self.config.tensor_features_linear_oev_input_shape,
                name='tensor_features_oevnn_input_layer'
            )

            oevnn_model = OeVNN(self.config).build(input_features_layer, input_tensor_features_oev_linear_layer)
            oevnn_output = oevnn_model([input_features_layer, input_tensor_features_oev_linear_layer])

            oevnn = Model(
                inputs=[
                    input_features_layer,
                    input_tensor_features_oev_linear_layer
                ],
                outputs=[
                    oevnn_output
                ]
            )
            oevnn._name = 'oevnn_framework'
            self.models.oevnn = oevnn
            self.models.nltbnn = self.models.evtbnn
            self.models.nltbnn._name = 'nltbnn_framework'
            del self.models.evtbnn

    ### in CaseSet/DataSet need to add support to transform and inverse transform features from eV
    

    def compile_models(self):
        for model_type, model in self.models.__dict__.items():
            model.compile(
                loss=self.config.loss,
                optimizer=self.config.optimizer(self.config.learning_rate_oevnn if model_type == 'oevnn' else self.config.learning_rate),
                metrics=self.config.metrics
            )
            setattr(self.models, model_type, model)

        if self.config.debug:
            for model in self.models.__dict__.values():
                print(model.summary())

    def train(self,
              dataset_obj :DataSet,
              train_caseset_obj :CaseSet,
              val_caseset_obj :Optional[CaseSet]=None):
        
        self.config.ensure_attr_group(['lr', 'epochs', 'batch'])
        
        if not isinstance(dataset_obj, DataSet):
            raise TypeError(f'dataset_obj must be {DataSet} instance')
        
        if not isinstance(train_caseset_obj, CaseSet):
            raise TypeError(f'train_caseset_obj must be {CaseSet} instance')
        
        if val_caseset_obj and not isinstance(val_caseset_obj, CaseSet):
            raise TypeError(f'val_caseset_obj must be {CaseSet} instance')
        
        x = [train_caseset_obj.features]
        y = [train_caseset_obj.labels]
        x_val = [val_caseset_obj.features]
        y_val = [val_caseset_obj.labels]
        
        if self.config._oevnltbnn:
            history = self.models.oevnn.fit(
                x=x,
                y=y,
                batch_size=self.config.batch,
                epochs=self.config.epochs,
                validation_data=(x_val, y_val) if val_caseset_obj else None,
                verbose=self.config.verbose,
                callbacks=self.config.keras_callbacks
            )

            ### extract evnn model from trained and store it for inference
            ### scale back with labels_ev_scaler
            ### subtract predictioins from anisotropy
            ### create a_!!, a_12, a_22, a_33 tensor
            ### use datasaset.labels.scaler.fit on train data again
            ### and dataset.labels.scaler.transform on all
            
        x.append(train_caseset_obj.tensor_features)
        if x_val:
            x_val.append(val_caseset_obj.tensor_features)

        if self.config._evtbnn:
            x.append(train_caseset_obj.tensor_features_linear)
            if x_val:
                x_val.append(val_caseset_obj.tensor_features_linearor)  

        for model_type, model in self.models._dict__.items():
            if model_type == 'oevnn':
                continue
            
            history = model.fit(
                x=x,
                y=y,
                batch_size=self.config.batch,
                epochs=self.config.epochs,
                validation_data=(x_val, y_val) if val_caseset_obj else None,
                verbose=self.config.verbose,
                callbacks=self.config.keras_callbacks
            )

            ### test this later
            # history_df = history.history
            # history_lr = round(model.optimizer.lr.numpy(), 5)
            setattr(self.model, model_type, model)
            # setattr(self.history, history_df)

            
            # self.history

        # else:
        #     ...

        return None


    def predict_oev(self,
                    dataset_obj :DataSet,
                    caseset_obj :CaseSet):
        
        
        return None
    
    def predict_anisotropy(self,
                           dataset_obj :DataSet,
                           caseset_obj :CaseSet):

        return None

        


if __name__ == '__main__':
    layers_tbnn = 3
    units_tbnn = 150
    features_input_shape = (15,3)
    tensor_features_input_shape = (20,3,3)

    layers_evnn = 2
    units_evnn = 150
    tensor_features_linear_input_shape = (3,)

    layers_oevnn = 2
    units_oevnn = 150
    tensor_features_linear_oev_input_shape = (3,)

    learning_rate=5e-4
    learning_rate_oevnn=1e-4

    tbnn_mixer_config = MixerConfig(
        features_mlp_layers=5,
        features_mlp_units=150
    )

    evnn_mixer_config = MixerConfig(
        features_mlp_layers=3,
        features_mlp_units=150
    )

    oevnn_mixer_config = MixerConfig(
        features_mlp_layers=5,
        features_mlp_units=150
    )

    TBNN_config = ModelConfig(
        layers_tbnn=layers_tbnn,
        units_tbnn=units_tbnn,
        features_input_shape=15,
        tensor_features_input_shape=tensor_features_input_shape,
        debug=True,
        # tbnn_mixer_config=tbnn_mixer_config
    )
    assert TBNN_config._evtbnn == False
    assert TBNN_config._oevnltbnn == False
    print('Sucess creating TBNN ModelConfig obj')
    tbnn = FrameWork(TBNN_config)
    tbnn.compile_models()
    
    eVTBNN_config = ModelConfig(
        layers_tbnn=layers_tbnn,
        units_tbnn=units_tbnn,
        features_input_shape=15,
        tensor_features_input_shape=tensor_features_input_shape,
        layers_evnn=layers_evnn,
        units_evnn=units_evnn,
        tensor_features_linear_input_shape=tensor_features_linear_input_shape,
        # tbnn_mixer_config=tbnn_mixer_config,
        # evnn_mixer_config=evnn_mixer_config,
        debug=True,
    )
    assert eVTBNN_config._evtbnn == True
    assert eVTBNN_config._oevnltbnn == False
    print('Sucess creating eVTBNN_config ModelConfig obj')
    evtbnn = FrameWork(eVTBNN_config)
    evtbnn.compile_models()

    OeVNLTBNN_config = ModelConfig(
        layers_tbnn=layers_tbnn,
        units_tbnn=units_tbnn,
        features_input_shape=features_input_shape,
        tensor_features_input_shape=tensor_features_input_shape,
        layers_evnn=layers_evnn,
        units_evnn=units_evnn,
        tensor_features_linear_input_shape=tensor_features_linear_input_shape,
        layers_oevnn=layers_oevnn,
        units_oevnn=units_oevnn,
        tensor_features_linear_oev_input_shape=tensor_features_linear_oev_input_shape,
        learning_rate=learning_rate,
        learning_rate_oevnn=learning_rate_oevnn,
        tbnn_mixer_config=tbnn_mixer_config,
        evnn_mixer_config=evnn_mixer_config,
        oevnn_mixer_config=oevnn_mixer_config,
        debug=True
    )
    assert OeVNLTBNN_config._evtbnn == True
    assert OeVNLTBNN_config._oevnltbnn == True
    print('Sucess creating mixer OeVNLTBNN_config ModelConfig obj')
    oevnltbnn = FrameWork(OeVNLTBNN_config)
    oevnltbnn.compile_models()


    ### put in place checl in config where if mixer config shape of features input shape >= 1, else int
    ### include train method