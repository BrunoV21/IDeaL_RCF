from types import SimpleNamespace

try:
    from ideal_rcf.models.config import ModelConfig, MixerConfig
    from ideal_rcf.models.tbnn import TBNN
    from ideal_rcf.models.evnn import eVNN
    from ideal_rcf.models.oevnn import OeVNN    
    from ideal_rcf.models.utils import MakeRealizable
    from ideal_rcf.dataloader.dataset import DataSet
    from ideal_rcf.dataloader.caseset import CaseSet

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
import matplotlib.pyplot as plt
from typing import Optional
import tensorflow as tf
import polars as pl
import numpy as np

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


    def build(self):

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
              val_caseset_obj :CaseSet=None):
        
        self.config.ensure_attr_group(['learning_rate', 'epochs', 'batch'])
        
        if not isinstance(dataset_obj, DataSet):
            raise TypeError(f'dataset_obj must be {DataSet} instance')
        
        if not isinstance(train_caseset_obj, CaseSet):
            raise TypeError(f'train_caseset_obj must be {CaseSet} instance')
        
        if val_caseset_obj and not isinstance(val_caseset_obj, CaseSet):
            raise TypeError(f'val_caseset_obj must be {CaseSet} instance')
        
        if self.config.shuffle:
            train_caseset_obj.shuffle()
            if val_caseset_obj:
                val_caseset_obj.shuffle()

        x = [train_caseset_obj.features]
        y = [train_caseset_obj.labels]

        if val_caseset_obj:
            x_val = [val_caseset_obj.features]
            y_val = [val_caseset_obj.labels]

        else:
            x_val = []
            y_val = []
        
        if self.config._oevnltbnn:
            x.append(train_caseset_obj.tensor_features_oev)
            if x_val:
                x_val.append(val_caseset_obj.tensor_features_oev)

            print('> starting oevnn traninng ...')
            history = self.models.oevnn.fit(
                x=x,
                y=y,
                batch_size=self.config.batch,
                epochs=self.config.epochs,
                validation_data=(x_val, y_val) if val_caseset_obj else None,
                verbose=self.config.verbose,
                callbacks=self.config.keras_callbacks
            )

            history_learning_rate = {'learning_rate': round(self.models.oevnn.optimizer.lr.numpy(), 5)}
            history_dict = history.history
            history_dict.update(history_learning_rate)
            self.history.oevnn = pl.from_dicts(history_dict)
            
            ### scale back labels used in to train oevnn with inverse_transform 
            ### get nl labels from oevnn model output, still using invariant_features + tensor_basis_oev
            self.calculate_nl_labels(dataset_obj, train_caseset_obj, dump=False)
            self.calculate_nl_labels(dataset_obj, val_caseset_obj, dump=False) if val_caseset_obj else None

            ### fit scaler
            train_caseset_obj.labels = dataset_obj.labels_scaler.fit_transform(train_caseset_obj.labels)
            if val_caseset_obj:
                val_caseset_obj.labels = dataset_obj.labels_scaler.transform(val_caseset_obj.labels)
            
            ### extract evnn model from trained and store it for inference in self.models.oevnn
            self.extract_oev()
            ### store oev predictions in each caseset althought they are shuffled
            ### so couldget away with doing it only later
            # self.predict_oev(train_caseset_obj, dump_predictions=False)
            # self.predict_oev(val_caseset_obj, dump_predictions=False) if val_caseset_obj else ...
            x = x[:-1]            
            y = [train_caseset_obj.labels]
            if x_val:
                x_val = x_val[:-1]                
                y_val = [val_caseset_obj.labels]

        x.append(train_caseset_obj.tensor_features)
        if x_val:
            x_val.append(val_caseset_obj.tensor_features)

        if self.config._evtbnn:
            x.append(train_caseset_obj.tensor_features_linear)
            if x_val:
                x_val.append(val_caseset_obj.tensor_features_linear)  

        for model_type, model in self.models.__dict__.items():
            if model_type == 'oevnn':
                continue

            print(f'> starting {model_type} traninng ...')
            history = model.fit(
                x=x,
                y=y,
                batch_size=self.config.batch,
                epochs=self.config.epochs,
                validation_data=(x_val, y_val) if val_caseset_obj else None,
                verbose=self.config.verbose,
                callbacks=self.config.keras_callbacks
            )
            history_learning_rate = {'learning_rate': round(model.optimizer.lr.numpy(), 5)}
            history_dict = history.history
            history_dict.update(history_learning_rate)

            setattr(self.models, model_type, model)
            setattr(self.history, model_type, pl.from_dicts(history_dict))
        
        train_caseset_obj.labels = dataset_obj.labels_scaler.inverse_transform(train_caseset_obj.labels)
        if y_val:
            val_caseset_obj.labels = dataset_obj.labels_scaler.inverse_transform(val_caseset_obj.labels)
            
        return tuple(obj for obj in [dataset_obj, train_caseset_obj, val_caseset_obj] if obj)
    

    def extract_oev(self):
        try:
            oev_model = self.models.oevnn.get_layer(name='oevnn')
        except ValueError:
            oev_model = self.models.oevnn.get_layer(name='mixer_oevnn')
        
        self.models.oevnn = Model(
            oev_model.layers[0].input, 
            oev_model.layers[-5].output
        )
        self.models.oevnn._name='oevnn'

        if self.config.debug:
            print(self.models.oevnn.summary())


    def calculate_nl_labels(self,
                            dataset_obj :DataSet,
                            caseset_obj :CaseSet,
                            dump :Optional[bool]=True):
        
        linear_term = dataset_obj.labels_oev_scaler.inverse_transform(
            self.models.oevnn([caseset_obj.features, caseset_obj.tensor_features_oev])[:,:,0]
        )

        full_term = dataset_obj.labels_oev_scaler.inverse_transform(caseset_obj.labels)

        nl_term = full_term - linear_term

        nl_term = np.transpose(
            [
                nl_term[:,0],
                nl_term[:,1],
                nl_term[:,2],
                -nl_term[:,0]-nl_term[:,2] ### 2D traceless condition
            ]
        )

        if dump:
            return nl_term

        else:
            caseset_obj.labels = nl_term
            # return caseset_obj

    def predict_oev(self,
                    caseset_obj :CaseSet,
                    dump_predictions :Optional[bool]=True):
        
        caseset_obj.predictions_oev = self.models.oevnn.predict([caseset_obj.features])[:,0]
        
        if dump_predictions:
            return caseset_obj.predictions_oev
        else:
            return None


    def inference(self,
                  dataset_obj : DataSet,
                  caseset_obj :CaseSet,
                  force_realizability :Optional[bool]=True,
                  dump_predictions :Optional[bool]=True):
        
        x = [caseset_obj.features, caseset_obj.tensor_features]
        if self.config._evtbnn:
            x.append(caseset_obj.tensor_features_linear)
        
        for model_type, model in self.models.__dict__.items():
            if model_type == 'oevnn':
                self.predict_oev(caseset_obj, dump_predictions=False)
                continue

            caseset_obj.predictions = dataset_obj.labels_scaler.inverse_transform(
                model.predict([x])
            )

            if force_realizability:
                caseset_obj.predictions = MakeRealizable(debug=self.config.debug).force_realizability(caseset_obj.predictions)

            if dump_predictions:
                return caseset_obj.predictions_oev, caseset_obj.predictions if self.config._oevnltbnn else caseset_obj.predictions
            else:
                return None


    def plot_metrics(self,
                     model_type :str,
                     metrics :pl.DataFrame):
    
        epochs = [i for i in range(metrics.shape[0])]
        metrics_types = metrics.columns

        metric_pairs = {}

        for i,_metric in enumerate(metrics_types):
            if _metric in list(metric_pairs.values()):
                continue
            for __metric in metrics_types[i+1:]:
                if _metric in __metric.split('_'):
                    metric_pairs[_metric]=__metric
        
        metric_pairs['learning_rate'] = 'loss'
        metric_pairs = [[key, value] for key,value in metric_pairs.items()]

        fig, axs = plt.subplots(1, len(metric_pairs), figsize=(len(metric_pairs)*10, 10))

        fig.suptitle(f'{model_type} training metrics', fontsize=40, y=1.0)
        for ax, pair_plot in zip(axs, metric_pairs):

            ax.plot(epochs, [metrics[pair_plot[0]].to_list(), metrics[pair_plot[1]].to_list()])
            ax.set_xlabel('Number of epochs', fontsize=40)
            ax.set_ylabel(' and '.join(pair_plot), fontsize=40)
            ax.legend(labels=pair_plot, fontsize=30)
            ax.set_xlim(0, len(epochs))
            if 'learning_rate' in pair_plot:
                ax.set_yscale('log')
            else :     
                ax.set_ylim(0, 2*np.mean(metrics[pair_plot[1]].to_list()))
        
        fig.show()




    def train_metrics(self):
        for model_type, history_metrics in self.history.__dict__.items():
            self.plot_metrics(model_type, history_metrics)


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
    oevnltbnn.extract_oev()

    ### put in place checl in config where if mixer config shape of features input shape >= 1, else int
    ### include train method