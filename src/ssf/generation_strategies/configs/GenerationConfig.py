from attr import dataclass


@dataclass
class GenerationConfig:
  max_new_tokens: int = 1000
  temperature: float = 0.0
  top_p: float = 1.0

  # only supported for VllmGenerationStrategy
  batch_size: int = None
  max_model_len: int = None