from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import os

class ParamGroup:
    def __init__(self, cfg: DictConfig):
        for key, value in cfg.items():
            setattr(self, key, value)

    def extract(self):
        return self

class ConfigHandler:
    def __init__(self, cfg: DictConfig):
        self.hydra_out = HydraConfig.get().run.dir
        self.dataset = ParamGroup(cfg.dataset)
        self.training = ParamGroup(cfg.training)
        self.debug = ParamGroup(cfg.debug)
        self.model = ParamGroup(cfg.model)
        self.optimization = ParamGroup(cfg.optimization)
        self.pipeline = ParamGroup(cfg.pipeline)

    def extract(self):
        return {
            "dataset": self.dataset.extract(),
            "training": self.training.extract(),
            "debug": self.debug.extract(),
            "model": self.model.extract(),
            "optimization": self.optimization.extract(),
            "pipeline": self.pipeline.extract(),
        }
    
class TriangulationConfigHandler:
    def __init__(self, cfg: DictConfig):
        self.hydra_out = HydraConfig.get().run.dir
        self.dataset = ParamGroup(cfg.dataset)
        self.debug = ParamGroup(cfg.debug)
    def extract(self):
        return {
            "dataset": self.dataset.extract(),
            "debug": self.debug.extract(),
        }