from pydantic import BaseModel, Field
from typing import Dict
import yaml

class ForceRedo(BaseModel):
  subreddit_metadata_sum: bool = False
  corpus_init: bool = False
  ref_plausible_inf_gen: bool = False
  ref_implausible_inf_gen: bool = False
  ssf_gen_ft: bool = False
  ssf_gen_inf: bool = False
  ref_inf_class_val: bool = False
  ref_inf_class_test: bool = False
  ref_inf_class_train: bool = False
  ssf_class_ft: bool = False
  ssf_class_val: bool = False
  ssf_class_test: bool = False
  ssf_class_all: bool = False
  ablations: bool = False

class ConvoContext(BaseModel):
  max_utt_summary_sentences: int = Field(gt=0)
  context_chain_utts: int = Field(gt=0)

class SubstantialTextMinChars(BaseModel):
  story: int = Field(gt=0)
  context_utt: int = Field(gt=0)
  initial_post_title: int = Field(gt=0)

class FineTuning(BaseModel):
  base_model: str
  ssf_gen_ft_script_base_name: str
  ssf_class_ft_script_base_name: str

class TaxonomyClassification(BaseModel):
  demo_counts: Dict[str, int]
  sampling_strategy: str

class RandomSeeds(BaseModel):
  default: int
  alternative: int

class Models(BaseModel):
  ssf_base: str
  openai_default: str
  openai_tax_class: str
  sbert_model: str

class DataDirs(BaseModel):
  subreddit_metadata: str
  corpus: str
  ref_inf_gen: str
  ssf_gen_ft: str
  ssf_gen_inf: str
  ref_inf_class: str
  ssf_class_ft: str
  ssf_class_inf: str
  annotations: str
  ablations: str
  analysis: str
  similarity: str
  plausibility_error_analyses: str
  ssf_sim_global_validation: str

class Dirs(BaseModel):
  data: DataDirs

class Inference(BaseModel):
  ssf_batch_size: int
  ssf_max_model_len: int

class Corpus(BaseModel):
  original_name: str
  extension_name: str
  source_conversations_coverage: float
  convo_context: ConvoContext
  story_seeker_threshold: float
  perspective_toxicity_threshold: float
  perspective_sexually_explicit_threshold: float
  disqualification_strings: list[str]
  substantial_text_min_chars: SubstantialTextMinChars
  community_meta_key: str

class Context(BaseModel):
  include_community_name: bool = True
  include_community_description: bool = True
  include_community_values: bool = True
  include_progenitor_summary: bool = True
  include_conversation_summary: bool = True

class Config(BaseModel):
  id: str
  force_redo: ForceRedo
  ft: FineTuning
  tax_class: TaxonomyClassification
  random_seeds: RandomSeeds
  dirs: Dirs
  models: Models
  inference: Inference
  corpus: Corpus
  context: Context


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)