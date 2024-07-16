# IDeaL_RCF
An Invariant Deep Learning RANS Closure Framework provides a unified way to interact with [A curated dataset for data-driven
turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500) by McConkey *et al.* allowing for data loading, preprocessing, model training and experimenting, inference, evaluation, integration with openfoam via exporting and postprocessing openfoam files.

The framework uses [tensorflow](https://www.tensorflow.org/api_docs/python/tf/all_symbols) and [keras](https://keras.io/api/) for the machine learning operations and [scikit-learn](https://scikit-learn.org/stable/index.html) for metrics generation. Plotting is done using [matplotlib](https://matplotlib.org/).

The provided models leverage Galilean Invariance when predicting the Anisotropy Tensor and an Eddy Viscosity which can then be injected into a converged [RANS](https://en.wikipedia.org/wiki/Reynolds-averaged_Navier%E2%80%93Stokes_equations) simulation using [OpenFOAM v2006](https://www.openfoam.com/news/main-news/openfoam-v20-06) and converging towards the [DNS](https://en.wikipedia.org/wiki/Direct_numerical_simulation) velocity field.


The physics behind this framework can be found [here](https://fenix.tecnico.ulisboa.pt/cursos/meaer21/dissertacao/1972678479056448).

Support for SSTBNNZ (a semi Supervised Zonal Approach) will be made avalable in the future.


## Instalation


```bash
conda create --name ML_Turb python=3.9
conda activate ML_Turb
pip install git+https://github.com/BrunoV21/IDeaL_RCF.git
```

## Dowloading the dataset
The original dataset can be downloaded directly from kaggle
```bash
kaggle datasets download -d ryleymcconkey/ml-turbulence-dataset
mkdir ml-turbulence-dataset
unzip ml-turbulence-dataset.zip -d ml-turbulence-dataset
```
The expanded dataset can be included with
```bash
gdown https://drive.google.com/uc?id=1rb2-7vJQtp_nLqxjmnGJI2aRQx8u9W6B
unzip a_3_1_2_NL_S_DNS_eV.zip -d ml-turbulence-dataset/komegasst
```


## Usage
The package is structure across three core objects CaseSet, DataSet and FrameWork.
A series of other modules are available for extended functionality such as evaluation, visualization and integration with OpenFOAM, all of which interact with a CaseSet obj. Before starting make sure that [A curated dataset for data-driven
turbulence modelling](https://doi.org/10.34740/kaggle/dsv/2637500) by McConkey *et al.* is present in your system. The version used in the present work was augmented using [these tools](...) and can be found [here](...).

### CaseSet
A CaseSet must be created via a SetConfig obj which contains the params to be loaded such as features and labels.

```python
from ideal_rcf.dataloader.config import SetConfig
from ideal_rcf.dataloader.caseset import CaseSet

set_config = SetConfig(...)N
caseset_obj = CaseSet('PHLL_case_1p2', set_config=set_config)
```
View the [creating_casesets.ipynb](./examples/creating_casesets.ipynb) example for more.

### DataSet
A DataSet receives the same type of SetConfig as the CaseSet but handles different parameters such as trainset, valset and tesset which are used to split the DataSet into the required sets for the supervised training. The DataSet object fits and stores the scalers built from the trainset.
```python
from ideal_rcf.dataloader.config import SetConfig
from ideal_rcf.dataloader.dataset import DataSet

set_config = SetConfig(...)
dataset_obj = DataSet(set_config=set_config)
train, val, test = dataset_obj.split_train_val_test()
```
View the [creating_datasets.ipynb](./examples/creating_datasets.ipynb) example for more.


### FrameWork
The FrameWork receives a ModelConfig obj which will determine the model to be used. Currently three models are supported:
1. TBNN - Tensor Based Neural Networks - proposed originally by Ling *et al.* [[paper]](https://www.osti.gov/servlets/purl/1333570) [code](https://github.com/sandialabs/tbnn)
2. eVTBN - Effective Viscosity Tensor Based Neural Network - proposed by ... [paper][thesis][[wiki]](https://github.com/BrunoV21/IDeaL_RCF/wiki)
3. OeVNLTBNN - Optimal Eddy Viscosity + Non Linear Tensor Based Neural Network:
    1. orginally proposed by Wang *et al.* [[paper]](https://arxiv.org/abs/1701.07102)
    2. improved by McConkey *et al.* [[paper]](https://arxiv.org/abs/2201.01710) to always be non-negative
    3. expanded by ... [paper][thesis] [[wiki]](https://github.com/BrunoV21/IDeaL_RCF/wiki) to be coupled with the anisotropy tensor via the strain rate during training but decoupled for inference and eVTBNN for non linear-term

Each model builds on the previous, so an eVTBNN is a TBNN combined with an eVNN while the OeVNLTBNN is an eVTBNN paired with an oEVNN.

View the [creating_models.ipynb](./examples/creating_models.ipynb) example for more.

#### TBNN
```python
from ideal_rcf.models.config import ModelConfig
from ideal_rcf.models.framework import FrameWork

TBNN_config = ModelConfig(
    layers_tbnn=layers_tbnn,
    units_tbnn=units_tbnn,
    features_input_shape=features_input_shape,
    tensor_features_input_shape=tensor_features_input_shape,
)
tbnn = FrameWork(TBNN_config)
tbnn.compile_models()
### acess compiled model
print(tbnn.models.tbnn.summary())
```

#### eVTBNN
```python
from ideal_rcf.models.config import ModelConfig
from ideal_rcf.models.framework import FrameWork

eVTBNN_config = ModelConfig(
    layers_tbnn=layers_tbnn,
    units_tbnn=units_tbnn,
    features_input_shape=features_input_shape,
    tensor_features_input_shape=tensor_features_input_shape,
    layers_evnn=layers_evnn,
    units_evnn=units_evnn,
    tensor_features_linear_input_shape=tensor_features_linear_input_shape,
)
evtbnn = FrameWork(eVTBNN_config)
evtbnn.compile_models()
### acess compiled model
print(evtbnn.models.evtbnn.summary())
```

#### OeVNLTBNN
```python
from ideal_rcf.models.config import ModelConfig
from ideal_rcf.models.framework import FrameWork

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
)
oevnltbnn = FrameWork(OeVNLTBNN_config)
oevnltbnn.compile_models()
### after training you can extract oev model from oevnn so that S_DNS is not required to run inference
### this is done automatically inside the train module
oevnltbnn.extract_oev()
### acess compiled model
print(oevnltbnn.models.oevnn.summary())
print(oevnltbnn.models.nltbnn.summary())
```

#### Mixer:
All models support the Mixer Architecture which is based on the concept introduced by Chen *et al.* in [TSMixer: An All-MLP Architecture for Time Series Forecasting](https://arxiv.org/abs/2303.06053) [[code]](https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/tsmixer.py) and adapted to work with spatial features while preserving invariance. The architecture and explanation are available in the [[wiki]](https://github.com/BrunoV21/IDeaL_RCF/wiki).

```python
from ideal_rcf.models.config import ModelConfig, MixerConfig
from ideal_rcf.models.framework import FrameWork

tbnn_mixer_config = MixerConfig(
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
tbnn = FrameWork(TBNN_config)
tbnn.compile_models()
### acess compiled model
print(tbnn.models.tbnn.summary())
```

#### train
```python
oevnltbnn.train(dataset_obj, train, val)
```

#### inference
```python
### the predictions are saved in the test obj
oevnltbnn.inference(dataset_obj, test)
```

### evaluate
```python
from ideal_rcf.infrastructure.evaluator import Evaluator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

metrics_list = [mean_squared_error, r2_score, mean_absolute_error]
eval_instance = Evaluator(metrics_list)
eval_instance.calculate_metrics(test)
```

### export to openfoam
```python
from ideal_rcf.foam.preprocess import FoamParser

### dump the predictions into openfoam compatible files
foam = FoamParser(PHLL_case_1p2)
foam.dump_predictions(dir_path)
```

## Examples
More use cases are covered in the [examples](./examples/) directory:
1. [FrameWork Training](./examples/training_oevnltbnn.ipynb)
2. [Setting Up Cross Validtion](./examples/cross_val_load_inference.py)
3. [Inference on loaded DataSet, Framework and exporting to openfoam](./examples/loading_inference_foam_export.py)
4. [Post Processing resulting foam files](./examples/load_foam_fields.ipynb)


## OpenFOAM Integration
The solvers and configurations used for injecting the predictions are available [here](./openfoam/)
