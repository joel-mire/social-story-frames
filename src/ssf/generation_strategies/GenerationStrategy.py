from abc import ABC, abstractmethod
from typing import List, Dict
from ssf.generation_strategies.configs import ModelConfig, GenerationConfig

class GenerationStrategy(ABC):
  def __init__(self, 
               model_config: ModelConfig,
               generation_config: GenerationConfig):
    self.model_config = model_config
    self.generation_config = generation_config
  
  @abstractmethod
  def generate(self,
               prompts: List[Dict[str, str]]):
    pass
  