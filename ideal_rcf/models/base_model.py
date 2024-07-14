try:
    from ideal_rcf.models.config import ModelConfig

except ModuleNotFoundError:
    from config import ModelConfig


class BaseModel(object):
    def __init__(self,
                 model_config :ModelConfig) -> None:
        
        if not isinstance(model_config, ModelConfig):            
            raise AssertionError(f'model_config must of instance {ModelConfig()}')
        
        self.config = model_config
        self.history = None


    def build(self):
        self.model = ...
        ### The ida is that eVNN, eVTBNN heredit from here and 
        ### but myabe it is best to implement logic at framework level
        ### and each individual model is built in each individual class
        ### and merged inside framework
        ...


    def complie(self):
        self.config.ensure_attr_group(['loss', 'optimizer', 'learning_rate'])

        self.model.compile(
            loss=self.config.loss,
            optimizer=self.config.optimizer(learning_rate=self.config.learning_rate),
            metrics=self.config.metrics)
        
        if self.config.debug:
            print(self.model.summary())


    def train(self):
        self.config.ensure_attr_group(['batch', 'epochs'])
        self.hisotry = self.model.train(

        )
        ### should be moved to framework or infrastucture level as it requires
        ### integration with CaseSets
        ...