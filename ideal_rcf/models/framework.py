from ideal_rcf.models.config import ModelConfig, MixerConfig
from ideal_rcf.models.tbnn import TBNN
from ideal_rcf.models.evnn import eVNN
from ideal_rcf.models.oevnn import OeVNN    
from ideal_rcf.models.utils import MakeRealizable
from ideal_rcf.dataloader.dataset import DataSet
from ideal_rcf.dataloader.caseset import CaseSet

from tensorflow.keras.layers import Input, Lambda, Add, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.models import load_model, save_model
from sklearn.linear_model import LinearRegression
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
import tensorflow as tf
import polars as pl
import numpy as np
from copy import deepcopy
import os

class FrameWork(object):
    """
    A class for managing and training machine learning models based on provided configurations.

    Parameters
    ----------
    model_config : ModelConfig
        Configuration object for the model setup.
    _id : Optional[str], optional
        Identifier for the framework instance.

    Methods
    -------
    __init__(model_config: ModelConfig, _id: Optional[str]=None)
        Initializes the FrameWork with the provided model configuration and ID.
    
    build()
        Constructs the models based on the configuration.
    
    compile_models()
        Compiles the models with the specified loss, optimizer, and metrics.
    
    train(dataset_obj: DataSet, train_caseset: CaseSet, val_caseset: CaseSet=None, use_pretrained_oevnn: Optional[bool]=False, dry_run: Optional[bool]=False)
        Trains the models on the provided dataset and case sets.
    
    extract_oev()
        Extracts and redefines the oevnn model.
    
    regress_nl_labels(caseset: CaseSet)
        Regresses non-linear labels using the provided case set.
    
    calculate_nl_labels(dataset_obj: DataSet, caseset: CaseSet)
        Calculates non-linear labels using the provided dataset and case set.
    
    predict_oev(dataset_obj: DataSet, caseset: CaseSet, scaled_features: Optional[bool]=False, dump_predictions: Optional[bool]=True)
        Predicts OEV values for the provided case set.
    
    predict_evtbnn(dataset_obj: DataSet, caseset: CaseSet, model, force_realizability: bool, scaled_features: Optional[bool]=False)
        Predicts EVTBNN values for the provided case set using the specified model.
    
    inference(dataset_obj: DataSet, caseset: CaseSet, force_realizability: Optional[bool]=True, dump_predictions: Optional[bool]=False)
        Performs inference on the provided case set.
    
    plot_metrics(model_type: str, metrics: pl.DataFrame)
        Plots the training metrics for the specified model type.
    
    train_metrics()
        Plots training metrics for all models.
    
    load_from_dir(dir_path: Path)
        Loads models from the specified directory.
    
    dump_to_dir(dir_path: Path)
        Dumps models to the specified directory.
    """
    def __init__(self,
                 model_config :ModelConfig,
                 _id :Optional[str]=None):
        
        if not isinstance(model_config, ModelConfig):
            raise AssertionError(f'[config_error] model_config must be of instance {ModelConfig()}')
        
        self.config = model_config
        self._id = _id

        tf.random.set_seed(42)

        self.compiled_from_files = False
        self.models = SimpleNamespace()
        self.history = SimpleNamespace()


    def build(self):
        """
        Build the neural network models based on the provided configuration.

        This method constructs either a TBNN (Tensor Basis Neural Network), 
        eVTBNN (effective Viscosity Tensor Basis Neural Network), or oEVNN (optimal Eddy Viscosity Tensor Basis Neural Network)
        model depending on the configuration settings provided in `self.config`.

        The method sets up the input layers, constructs the model architectures,
        and saves the constructed models in `self.models`.

        Inputs:
        - `input_features_layer`: Layer for general input features.
        - `input_tensor_features_layer`: Layer for tensor input features.
        - Optionally, if eVTBNN is configured:
            - `input_tensor_features_linear_layer`: Layer for linear tensor input features.
        - Optionally, if oEVNN is configured:
            - `input_tensor_features_oev_linear_layer`: Layer for output enriched vector input features.

        Outputs:
        - `merged_output`: The concatenated output of different components of the model,
        with specific manipulations such as additions and negations applied to individual elements.
        
        Models:
        - `tbnn`: Tensor Basis Neural Network model.
        - Optionally, if EVTBNN is configured:
            - `evtbnn`: effective Viscosity Tensor Basis Neural Network model.
        - Optionally, if OEvNN is configured:
            - `oevnn`: optimal Eddy Viscosity Tensor Basis Neural Network model.
            - `nltbnn`: Nonlinear Tensor Basis Neural Network model (derived from EVTBNN).
        """
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
        
        self.build() if not self.compiled_from_files else ...

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
              train_caseset :CaseSet,
              val_caseset:CaseSet=None,
              use_pretrained_oevnn :Optional[bool]=False,
              dry_run :Optional[bool]=False):
        
        self.config.ensure_attr_group(['learning_rate', 'epochs', 'batch'])
        
        if not isinstance(dataset_obj, DataSet):
            raise TypeError(f'dataset_obj must be {DataSet} instance')
        
        if not isinstance(train_caseset, CaseSet):
            raise TypeError(f'train_caseset_obj must be {CaseSet} instance')
        
        if val_caseset and not isinstance(val_caseset, CaseSet):
            raise TypeError(f'val_caseset_obj must be {CaseSet} instance')
        
        train_caseset_obj = CaseSet(
                    case=train_caseset.case,
                    set_config=train_caseset.config, 
                    set_id=train_caseset.set_id,
                    initialize_empty=True
                )                
        train_caseset_obj._import_from_copy(*deepcopy(train_caseset._export_for_stack()))

        if val_caseset:
            val_caseset_obj = CaseSet(
                        case=val_caseset.case,
                        set_config=val_caseset.config, 
                        set_id=val_caseset.set_id,
                        initialize_empty=True
                    )                
            val_caseset_obj._import_from_copy(*deepcopy(val_caseset._export_for_stack()))
        
        if self.config.shuffle:
            train_caseset_obj.shuffle()
            val_caseset_obj.shuffle()
        
        if self.config._oevnltbnn:
            if train_caseset.config.features_transforms:
                train_caseset_obj._transform_features()

            dataset_obj.features_oev_scaler, dataset_obj.labels_oev_scaler = train_caseset_obj._fit_scaler_oev(dataset_obj.features_oev_scaler, dataset_obj.labels_oev_scaler)
            train_caseset_obj._scale_oev(dataset_obj.features_oev_scaler, dataset_obj.labels_oev_scaler)
            
            if self.config.oevnn_mixer_config:
                train_caseset_obj._fit_mixer_scaler(dataset_obj.mixer_invariant_oev_features_scaler)
                train_caseset_obj._build_mixer_features(dataset_obj.mixer_invariant_oev_features_scaler)
            
            if val_caseset:
                if val_caseset.config.features_transforms:
                    val_caseset_obj._transform_features()
                val_caseset_obj._scale_oev(dataset_obj.features_oev_scaler, dataset_obj.labels_oev_scaler)
                val_caseset_obj._build_mixer_features(dataset_obj.mixer_invariant_oev_features_scaler) if self.config.oevnn_mixer_config else ...
            
            x = [train_caseset_obj.features, train_caseset_obj.tensor_features_oev]
            y = [train_caseset_obj.labels]

            if val_caseset:
                x_val = [val_caseset_obj.features, val_caseset_obj.tensor_features_oev]
                y_val = [val_caseset_obj.labels]

            else:
                x_val = []
                y_val = []

            if not use_pretrained_oevnn:
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
                self.extract_oev()
                
            self.predict_oev(dataset_obj, train_caseset)
            self.regress_nl_labels(train_caseset) if self.config.regress_nl_labels else self.calculate_nl_labels(dataset_obj, train_caseset)

            if val_caseset:
                self.predict_oev(dataset_obj, val_caseset)
                self.regress_nl_labels(val_caseset) if self.config.regress_nl_labels else self.calculate_nl_labels(dataset_obj, val_caseset)

            train_caseset_obj._import_from_copy(*deepcopy(train_caseset._export_for_stack()))
            val_caseset_obj._import_from_copy(*deepcopy(val_caseset._export_for_stack())) if val_caseset else ...

            if self.config.shuffle:
                train_caseset_obj.shuffle()
                val_caseset_obj.shuffle()

        dataset_obj.features_scaler, dataset_obj.labels_scaler, dataset_obj.mixer_invariant_features_scaler = train_caseset_obj._fit_scaler(dataset_obj.features_scaler, dataset_obj.labels_scaler, dataset_obj.mixer_invariant_features_scaler)
        if self.config.tbnn_mixer_config and self.config.evnn_mixer_config:
            train_caseset_obj._build_mixer_features(dataset_obj.mixer_invariant_features_scaler)
        if train_caseset.config.features_transforms:
            train_caseset_obj._transform_features()
        train_caseset_obj._scale(dataset_obj.features_scaler, dataset_obj.labels_scaler)

        if val_caseset:
            if self.config.tbnn_mixer_config and self.config.evnn_mixer_config:
                val_caseset_obj._build_mixer_features(dataset_obj.mixer_invariant_features_scaler)
            if val_caseset.config.features_transforms:
                val_caseset_obj._transform_features()
            val_caseset_obj._scale(dataset_obj.features_scaler, dataset_obj.labels_scaler)
        
        x = [train_caseset_obj.features, train_caseset_obj.tensor_features]
        y = [train_caseset_obj.labels]
        if val_caseset:
            x_val = [val_caseset_obj.features, val_caseset_obj.tensor_features]
            y_val = [val_caseset_obj.labels]

        if self.config._evtbnn:
            x.append(train_caseset_obj.tensor_features_linear)
            if x_val:
                x_val.append(val_caseset_obj.tensor_features_linear)

        if dry_run:
            return tuple(obj for obj in [dataset_obj, train_caseset_obj, val_caseset_obj] if obj)

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


    def regress_nl_labels(self,
                          caseset :CaseSet):
        
        if caseset.labels_compiled:
            return
        
        caseset.tensor_features_oev        
        reg_nnls = LinearRegression(positive=True)
        oev_labels = np.array(
            [
                reg_nnls.fit(-2*_S.reshape(-1,1), _a.reshape(-1,1)).coef_ 
                for _S, _a in zip(caseset.tensor_features_oev, caseset.labels)
            ]
        ).reshape(caseset.tensor_features_oev.shape[0], 1)
        
        caseset.labels += 2*oev_labels*caseset.tensor_features_oev

        caseset.labels = np.transpose(
                                    [
                                        caseset.labels[:,0],
                                        caseset.labels[:,1],
                                        caseset.labels[:,2],
                                        -caseset.labels[:,0]-caseset.labels[:,2] ### 2D traceless condition
                                    ]
                                )
        if self.config.debug:
            print('[nl labels] regressing with nnls')

        caseset.labels_compiled=True        


    def calculate_nl_labels(self,
                            dataset_obj :DataSet, 
                            caseset: CaseSet):
        if caseset.labels_compiled:
            return
        
        caseset.labels -= dataset_obj.labels_oev_scaler.inverse_transform(
            dataset_obj.labels_oev_scaler.transform(caseset.tensor_features_oev)*-2*caseset.predictions_oev.reshape(-1,1)
        )
        
        caseset.labels = np.transpose(
                                    [
                                        caseset.labels[:,0],
                                        caseset.labels[:,1],
                                        caseset.labels[:,2],
                                        -caseset.labels[:,0]-caseset.labels[:,2] ### 2D traceless condition
                                    ]
                                )
        
        caseset.labels_compiled=True


    def predict_oev(self,
                    dataset_obj :DataSet,
                    caseset :CaseSet,
                    scaled_features :Optional[bool]=False,
                    dump_predictions :Optional[bool]=True):
        
        caseset_obj = CaseSet(
                    case=caseset.case,
                    set_config=caseset.config, 
                    set_id=caseset.set_id,
                    initialize_empty=True
                )        
        caseset_obj._import_from_copy(*deepcopy(caseset._export_for_stack()))
        if not scaled_features:
            if caseset.config.features_transforms:
                caseset_obj._transform_features()
            caseset_obj._scale_oev(dataset_obj.features_oev_scaler, dataset_obj.labels_oev_scaler)
            if self.config.oevnn_mixer_config:
                caseset_obj._build_mixer_features(dataset_obj.mixer_invariant_oev_features_scaler)

        caseset.predictions_oev = self.models.oevnn.predict([caseset_obj.features])[:,0]
        
        if dump_predictions:
            return caseset.predictions_oev
        else:
            return None


    def predict_evtbnn(self,
                       dataset_obj :DataSet,
                       caseset :CaseSet,
                       model,
                       force_realizability :bool,
                       scaled_features :Optional[bool]=False,):
        
        caseset_obj = CaseSet(
                    case=caseset.case,
                    set_config=caseset.config, 
                    set_id=caseset.set_id,
                    initialize_empty=True
                )        
        caseset_obj._import_from_copy(*deepcopy(caseset._export_for_stack()))

        if not scaled_features:
            if self.config.tbnn_mixer_config and self.config.evnn_mixer_config:
                caseset_obj._build_mixer_features(dataset_obj.mixer_invariant_features_scaler)
            if caseset.config.features_transforms:
                caseset_obj._transform_features()
            caseset_obj._scale(dataset_obj.features_scaler, dataset_obj.labels_scaler)
        
        x = [caseset_obj.features, caseset_obj.tensor_features]
        if self.config._evtbnn:
            x.append(caseset_obj.tensor_features_linear)

        caseset.predictions = dataset_obj.labels_scaler.inverse_transform(
                model.predict(x)
        )
        
        if force_realizability:
            caseset.predictions = MakeRealizable(debug=self.config.debug).force_realizability(caseset.predictions)


    def inference(self,
                  dataset_obj :DataSet,
                  caseset :CaseSet,
                  force_realizability :Optional[bool]=True,
                  dump_predictions :Optional[bool]=False):
        """
        Perform inference using the stored neural network models on the provided dataset and caseset.

        This method iterates over the models stored in `self.models` and performs predictions or regressions
        based on the model type and configuration settings.

        Args:
        - `dataset_obj`: DataSet object containing input data for inference.
        - `caseset`: CaseSet object containing cases on which predictions or regressions are to be performed.
        - `force_realizability`: Optional boolean flag indicating whether to force realizability of predictions.
        - `dump_predictions`: Optional boolean flag indicating whether to dump predictions.

        Returns:
        - If `dump_predictions` is True, returns predictions stored in `caseset.predictions_oev` or `caseset.predictions`
        depending on the configuration.
        - If `dump_predictions` is False, returns None.

        Raises:
        - ValueError: If predictions or labels are missing or invalid during operations.
        """       
        for model_type, model in self.models.__dict__.items():
            if model_type == 'oevnn':
                try:
                    bool(caseset.predictions_oev);                
                    self.predict_oev(dataset_obj, caseset)
                    try:
                        bool(caseset.labels);
                    except ValueError:
                        self.regress_nl_labels(caseset) if self.config.regress_nl_labels else self.calculate_nl_labels(dataset_obj, caseset)
                except ValueError:
                    ...
                continue

            self.predict_evtbnn(dataset_obj, caseset, model, force_realizability)

            if dump_predictions:
                return caseset.predictions_oev, caseset.predictions if self.config._oevnltbnn else caseset.predictions
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
        
        plt.show(block=False)


    def train_metrics(self):
        for model_type, history_metrics in self.history.__dict__.items():
            self.plot_metrics(model_type, history_metrics)


    def load_from_dir(self, 
                      dir_path :Path):
        
        models_dir = f'{dir_path}/models'
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f'Ensure {models_dir} exists and contains the model files')
        for model_file in os.listdir(Path(models_dir)):
            model_type = model_file.split('.')[0]
            model = load_model(f'{models_dir}/{model_file}', compile=False)
            setattr(self.models, model_type, model)
            print(f'[{model_type}] loaded sucessfully')
        
        self.compiled_from_files=True


    def dump_to_dir(self, 
                    dir_path :Path):
        
        models_dir = f'{dir_path}/models'
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        for model_type, model in self.models.__dict__.items():
            save_model(model, f'{models_dir}/{model_type}.h5')
            print(f'[{model_type}] dumped sucessfully')