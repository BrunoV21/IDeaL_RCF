from ideal_rcf.models.config import MixtureConfig, ClassifierConfig, ModelConfig
from ideal_rcf.models.framework import FrameWork
from typing import Optional


class SemiSupervisedZonalFramework(FrameWork):
    def __init__(self,
                 mixture_config :MixtureConfig,
                 classifier_config :ClassifierConfig,
                 model_config :ModelConfig,
                 _id :Optional[str]=None) -> None:
        
        super().__init__(model_config, _id=_id)

        ### use same structure as framework
        ### build / compile
        ### train
        ### save to dir
        ### load from dir