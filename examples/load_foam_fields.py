from ideal_rcf.dataloader.config import SetConfig, set_dataset_path
from ideal_rcf.dataloader.caseset import CaseSet

from ideal_rcf.foam.postprocess import FoamLoader
from ideal_rcf.foam.visualization import FoamPlottingTools

from utils import fetch_experiments

import matplotlib.pyplot as plt
import os


def load_foam_fields_example():

    dataset_path = os.getenv('DATASET_PATH')
    turb_dataset = 'komegasst'
    custom_turb_dataset = 'a_3_1_2_NL_S_DNS_eV'

    case = [
        'PHLL_case_0p5',
        'PHLL_case_0p8',
        'PHLL_case_1p0',
        'PHLL_case_1p2',
        'PHLL_case_1p5'
    ]

    trainset = [
        'PHLL_case_0p5',
        'PHLL_case_0p8',
        'PHLL_case_1p5'
    ]

    valset = [
        'PHLL_case_1p0',
    ]

    testset = [
        'PHLL_case_1p2',
    ]

    features_filter = ['I1_1', 'I1_2', 'I1_3', 'I1_4', 'I1_5', 'I1_6', 'I1_8', 'I1_9', 'I1_15', 'I1_17', 'I1_19', 'I2_3', 'I2_4', 'q_1', 'q_2']

    features = ['I1', 'I2', 'q']
    features_cardinality = [20, 20, 4]

    tensor_features = ['Tensors']
    tensor_features_linear = ['Shat']
    labels = ['a']

    tensor_features_oev = ['S_DNS']


    features_transforms = ['same_sign_log']
    skip_features_transforms_for = ['I1_2', 'I1_5', 'I1_8','I1_15','I1_17', 'I1_19', 'q_1', 'q_2', 'q_3', 'q_4']


    no_labes_CaseSetConfig = SetConfig(
            cases=case,
            turb_dataset=turb_dataset,
            dataset_path=dataset_path,
            trainset=trainset,
            valset=valset,
            testset=testset,
            features=features,
            tensor_features=tensor_features,
            tensor_features_linear=tensor_features_linear,
            # labels=labels,
            custom_turb_dataset=custom_turb_dataset,
            tensor_features_oev=tensor_features_oev,
            features_filter=features_filter,
            features_cardinality=features_cardinality,
            features_transforms=features_transforms,
            skip_features_transforms_for=skip_features_transforms_for,
            enable_mixer=True,
            debug=True,
        )

    PHLL_case_1p2 = CaseSet(case='PHLL_case_1p2', set_config=no_labes_CaseSetConfig)

    FoamLoader(PHLL_case_1p2).read_from_dir('./experiments/final_results_cross_val/fold_1_test/foam/PHLL_case_1p2')

    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    metrics_list = [mean_squared_error, r2_score, mean_absolute_error]

    foamplotter = FoamPlottingTools(metrics_list, exp_id='OEVNLTBNN')

    foamplotter.parity_plots(PHLL_case_1p2)

    foamplotter.velocity_plots(PHLL_case_1p2)

    foamplotter.plot_velocity_profiles(PHLL_case_1p2)
    
    foamplotter.get_plots_error(PHLL_case_1p2, foamplotter.velocity_abs_error, cmap_id='seismic')
    
    foamplotter.plot_wall_sheer_stress(PHLL_case_1p2, wall='bottom')

    plt.show()


if __name__ == '__main__':
    ### Set DataSet Path ad Environ Var
    set_dataset_path()
    fetch_experiments()
    load_foam_fields_example()