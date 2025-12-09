import argparse
from ssf.CorpusAugmenter import CorpusAugmenter
from ssf.helpers import CorpusHelper
from ssf.community_preprocessors import CommunityPreprocessor
from ssf.generation_strategies.configs import ModelConfig, GenerationConfig
from ssf.generation_strategies import OpenaiGenerationStrategy
from ssf.generation_strategies.VllmGenerationStrategy import VllmGenerationStrategy
from ssf.corpus_splitters.PreassignedSplitCorpusSplitter import PreassignedSplitCorpusSplitter
import os
import numpy as np
import random
from ssf.Configs import load_config
from ssf.Constants import *
from ssf.Taxonomy import Taxonomy
from ssf.community_preprocessors.SubredditPreprocessor import SubredditPreprocessor
import json
from ssf.utils import PdUtils, AnnotationUtils, FtUtils, TorchUtils, InferenceUtils, TaxonomyUtils
from ssf.task_managers.BatchMultiOutputTaskManager import BatchMultiOutputTaskManager
from ssf.task_managers.SingleOutputTaskManager import SingleOutputTaskManager
from ssf.task_managers.ImplausibleSingleOutputTaskManager import ImplausibleSingleOutputTaskManager
from ssf.Exceptions import *
import pandas as pd
from ssf.TaxClassStageOrchestrator import TaxClassStageOrchestrator
from ssf.classifiers.BatchTaxonomyClassifier import BatchTaxonomyClassifier
from ssf.helpers import TaxonomyEvaluator
import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, required=True)
  return parser.parse_args()

def ensure_data_dirs(config: Config):
  for dir in config.dirs.data.model_dump().values():
    os.makedirs(dir, exist_ok=True)

def set_random_seeds(config):
  np.random.seed(config.random_seeds.default)
  random.seed(config.random_seeds.default)

def get_default_openai_generation_strategy(config):
  model_config = ModelConfig(model_name=config.models.openai_default)
  generation_config = GenerationConfig()
  return OpenaiGenerationStrategy(model_config=model_config,
                                  generation_config=generation_config)

def get_tax_class_openai_generation_strategy(config):
  model_config = ModelConfig(model_name=config.models.openai_tax_class)
  generation_config = GenerationConfig()
  return OpenaiGenerationStrategy(model_config=model_config,
                                  generation_config=generation_config)

def swap_plausible_with_implausible(df, 
                                    taxonomy, 
                                    model_name_normalized,
                                    random_seed, 
                                    id_swap_dim_dict=None):
  swapped_values = []
  swap_dims = []
  df = df.reset_index(drop=True)

  if id_swap_dim_dict is not None:
    # Use dict-based swap
    for i, row in df.iterrows():
      id = row['id']
      dim = id_swap_dim_dict.get(id)
      if dim is None:
        raise ValueError(f"No swap dim found for story ID {id}")
      implausible_col = f"implausible_{model_name_normalized}${dim}_gen0"
      plausible_col = f"{model_name_normalized}_{dim}_genRand"
      implausible_value = row[implausible_col]
      plausible_value = row[plausible_col]
      df.at[i, plausible_col] = implausible_value
      swap_dims.append(dim)
      swapped_values.append(plausible_value)
  else:
    # Use random swap
    random_dims = random.choices(taxonomy.get_dims(), k=len(df))
    for i, dim in enumerate(random_dims):
      implausible_col = f"implausible_{model_name_normalized}${dim}_gen0"
      plausible_col = f"{model_name_normalized}_{dim}_genRand"
      implausible_value = df.at[i, implausible_col]
      plausible_value = df.at[i, plausible_col]
      df.at[i, plausible_col] = implausible_value
      swap_dims.append(dim)
      swapped_values.append(plausible_value)

  df['swap_value'] = swapped_values
  df['swap_dim'] = swap_dims
  df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
  return df

def sample_plausible_responses(df, 
                               model_name_normalized: str, 
                               taxonomy) -> pd.DataFrame:
  df = df.copy()
  for dim in taxonomy.get_dims():
    sampled_response_list = []
    sampled_response_idx_list = []
    for _, row in df.iterrows():
      options = {
        j: row[f"{model_name_normalized}_{dim}_gen{j}"]
        for j in range(3)
        if f"{model_name_normalized}_{dim}_gen{j}" in row and row[f"{model_name_normalized}_{dim}_gen{j}"] != ""
      }
      if options:
        random_key, random_value = random.choice(list(options.items()))
        sampled_response_list.append(random_value)
        sampled_response_idx_list.append(random_key)
      else:
        sampled_response_list.append("")
        sampled_response_idx_list.append(None)
    df[f"{model_name_normalized}_{dim}_genRand"] = sampled_response_list
    df[f"{model_name_normalized}_{dim}_genRandIdx"] = sampled_response_idx_list
  return df
  
def postprocess_stage3(df, 
                       out_path, 
                       output_col_prefix, 
                       taxonomy):
  gen_suffix = "gen0"
  batch_results = {}
  with open(out_path, 'r') as f:
    for line in f:
      line = line.strip()
      result = json.loads(line)
      custom_id = result['custom_id']
      output = result['output']

      custom_id_parts = custom_id.split('_')
      instance_id = custom_id_parts[0]
      dim = '_'.join(custom_id_parts[1:-1])  # Handle multi-word dimensions
      gen_name = custom_id_parts[-1]  # gen0, gen1, gen2
      batch_results[custom_id] = {
        'instance_id': instance_id,
        'dim': dim,
        'gen_name': gen_name,
        'output': output
    }
    
    for dim in taxonomy.get_dims():
      target_col = f"{output_col_prefix}_{dim}_{gen_suffix}"
      df[target_col] = [None for _ in range(len(df))]
    
    for _, row in df.iterrows():
      if row['split'] == 'train':  # Only process train split
        instance_id = str(row['id'])
        for dim in taxonomy.get_dims():
          custom_id = f"{instance_id}_{dim}_{gen_suffix}"
          target_col = f"{output_col_prefix}_{dim}_{gen_suffix}"
          parsed_output = InferenceUtils.parse_json(batch_results[custom_id]['output'].lstrip("```json").rstrip("```"))
          response_val = parsed_output['response']
          df.at[row.name, target_col] = response_val
    
    return df

def filter_and_align_dataframes(target_df, reference_df, verbose=True):
  """
  Filter target_df to only include instances that appear in reference_df,
  and reorder rows to match the order in reference_df.
  
  Args:
    target_df: DataFrame to be filtered
    reference_df: DataFrame providing the filter criteria and ordering
    verbose: Whether to print debugging information
    
  Returns:
    Filtered and reordered DataFrame
  """
  if verbose:
    print(f"Before filtering - target_df length: {len(target_df)}")
    print(f"Before filtering - reference_df length: {len(reference_df)}")
    print(f"Before filtering - target_df first 5 ids: {target_df['id'].head().tolist()}")
    print(f"Before filtering - reference_df first 5 ids: {reference_df['id'].head().tolist()}")
  
  # Create filtered dataframe with only instances that appear in reference_df
  target_df_filtered = target_df[target_df['id'].isin(reference_df['id'])].copy()
  
  # Reorder rows to match the order in reference_df
  target_df_filtered = target_df_filtered.set_index('id').loc[reference_df['id']].reset_index()
  
  if verbose:
    print(f"After filtering - target_df_filtered length: {len(target_df_filtered)}")
    print(f"After filtering - target_df_filtered first 5 ids: {target_df_filtered['id'].head().tolist()}")
    print(f"After filtering - reference_df first 5 ids: {reference_df['id'].head().tolist()}")
  
  return target_df_filtered

if __name__ == "__main__":
  args = parse_args()

  """
  Load Configs
  """
  config = load_config(args.config)
  ensure_data_dirs(config)
  
  taxonomy = Taxonomy(taxonomy_dir=TAXONOMY_DIR)
  taxonomy.to_latex_table(path=TAXONOMY_TEX_PATH)

  set_random_seeds(config)

  default_openai_generation_strategy = get_default_openai_generation_strategy(config)
  tax_class_openai_generation_strategy = get_tax_class_openai_generation_strategy(config)
  
  """
  Preprocess Corpus
  """
  community_preprocessor: CommunityPreprocessor = SubredditPreprocessor(subreddits_path=SUBREDDITS_PATH,
                                                                        subreddit_desc_rules_path=SUBREDDIT_DESC_RULES_PATH,
                                                                        dir=config.dirs.data.subreddit_metadata,
                                                                        generation_strategy=default_openai_generation_strategy,
                                                                        force_rebuild=config.force_redo.subreddit_metadata_sum)
  community_allowlist = community_preprocessor.get_community_allowlist()
  community_data_dict = community_preprocessor.get_community_data_dict()

  corpus_utils = CorpusHelper(config=config.corpus, corpus_dir=config.dirs.data.corpus)
  corpus = corpus_utils.load_corpus(force_rebuild=config.force_redo.corpus_init)

  corpus_augmenter = CorpusAugmenter(corpus_utils=corpus_utils, corpus_dir=config.dirs.data.corpus)
  corpus = corpus_augmenter.apply_ck_transformer_pipeline(corpus=corpus, 
                                                          community_allowlist=community_allowlist,
                                                          community_data_dict=community_data_dict,
                                                          generation_strategy=default_openai_generation_strategy,
                                                          force_redo_corpus=config.force_redo.corpus_init)
  
  ssf_df = corpus_utils.get_ssf_df(corpus=corpus,
                                   community_allowlist=community_allowlist)
  
  # filter out r/wow which should not have made it past subreddit filtering
  ssf_df = ssf_df[ssf_df['meta.subreddit'] != 'wow']
  ssf_df = ssf_df.reset_index(drop=True)

  # Potato annotation prep
  ssf_df['_text'] = ssf_df['text']
  ssf_df['text'] = ssf_df.apply(lambda x: AnnotationUtils.fmt_text_for_potato(x), axis=1)
  ssf_df['label_suggestions'] = ssf_df.apply(lambda x: json.dumps(taxonomy.get_label_suggestions()), axis=1)

  # add train/val/test splits using authoritative predefined splits
  corpus_splitter = PreassignedSplitCorpusSplitter(
      train_split_path=f"{config.dirs.data.corpus}/predefined_splits/train.csv",
      test_split_path=f"{config.dirs.data.corpus}/predefined_splits/test.csv",
      val_split_path=f"{config.dirs.data.corpus}/predefined_splits/val.csv"
  )
  ssf_df, train_metadata, test_metadata, val_metadata = corpus_splitter.add_split_column(ssf_df, random_state=config.random_seeds.default)

  # save ssf_df
  ssf_path = f"{config.dirs.data.corpus}/{SSF_DF_PATH}"
  PdUtils.save_df(ssf_df, ssf_path)

  """
  Reference Inference Generation
  """
  ssf_split_df = ssf_df[ssf_df['split'].notna()]

  # Add (putatively plausible) llm reference generations to ssf_split_df
  batch_manager = BatchMultiOutputTaskManager(
      taxonomy=taxonomy,
      model_name=default_openai_generation_strategy,
      out_dir=config.dirs.data.ref_inf_gen,
      force_redo=config.force_redo.ref_plausible_inf_gen,
      disambiguator=config.models.openai_default,
      context_config=config.context
  )
  openai_default_model_normalized = config.models.openai_default.replace("-", "_")
  ssf_split_df = batch_manager.run_task(stories_df=ssf_split_df)
  cols_to_check = [f"{openai_default_model_normalized}_{dim}_gen0" for dim in taxonomy.get_dims()]
  ssf_split_df = PdUtils.remove_nan_rows(ssf_split_df, cols_to_check)
  ssf_split_df.to_csv(f"{config.dirs.data.corpus}/ssf_split.csv", index=False)

  ssf_split_test_df = ssf_split_df[ssf_split_df['split'] == 'test']
  ssf_split_val_df = ssf_split_df[ssf_split_df['split'] == 'val']
  ssf_split_train_df = ssf_split_df[ssf_split_df['split'] == 'train']

  # Randomly sample 1 of the (up to) 3 plausible generations to be presented to human annotators
  ssf_split_test_df = sample_plausible_responses(df=ssf_split_test_df, 
                                                 model_name_normalized=openai_default_model_normalized,
                                                 taxonomy=taxonomy)
  ssf_split_test_df.to_csv(f"{config.dirs.data.corpus}/ssf_split_test.csv", index=False)
  
  # Generate known-implausible inferences for human annotation task
  implausible_llm_ref_task_manager = ImplausibleSingleOutputTaskManager(
      taxonomy=taxonomy,
      generation_strategy=default_openai_generation_strategy,
      out_dir=config.dirs.data.ref_inf_gen,
      force_redo=config.force_redo.ref_implausible_inf_gen,
      disambiguator=f"implausible_{openai_default_model_normalized}",
      context_config=config.context
  )
  implausible_llm_ref_task_manager.run_task(stories_df=ssf_split_test_df)
  ssf_split_test_df = implausible_llm_ref_task_manager.postprocess_results(stories_df=ssf_split_test_df)

  # For each story in ssf_split_test_df, swap plausible inference with implausible inference for one random dimension.
  id_swap_dim_dict_path = f"{config.dirs.data.ref_inf_gen}/id_swap_dim_dict.json"
  id_swap_dim_dict = json.load(open(id_swap_dim_dict_path, 'r')) if os.path.exists(id_swap_dim_dict_path) else None
  ssf_split_test_df = swap_plausible_with_implausible(df=ssf_split_test_df, 
                                                      taxonomy=taxonomy,
                                                      model_name_normalized=openai_default_model_normalized,
                                                      random_seed=config.random_seeds.default,
                                                      id_swap_dim_dict=id_swap_dim_dict)
  ssf_split_test_df.to_csv(f"{config.dirs.data.corpus}/ssf_split_test.csv", index=False)
  ssf_split_df.to_csv(f"{config.dirs.data.corpus}/ssf_split.csv", index=False)

  PdUtils.save_df(ssf_split_df, f"{config.dirs.data.corpus}/ssf_split.csv")
  PdUtils.save_df(ssf_split_test_df, f"{config.dirs.data.corpus}/ssf_split_test.csv")
  PdUtils.save_df(ssf_split_val_df, f"{config.dirs.data.corpus}/ssf_split_val.csv")

  """
  SSF-Generator Finetuning
  """
  # prompt_col_suffix = PROMPT_COL_SUFFIX_FULL_CONTEXT
  for prompt_col_suffix in ALL_PROMPT_COL_SUFFIXES:
    single_output_task_manager = SingleOutputTaskManager(taxonomy=taxonomy,
                                                        generation_strategy=None,
                                                        out_dir=config.dirs.data.corpus,
                                                        force_redo=True,
                                                        disambiguator=prompt_col_suffix,
                                                        context_config=config.context)
    for dim in taxonomy.get_dims():
      ssf_df[f"{dim}${prompt_col_suffix}$single_output_prompt"] = single_output_task_manager.get_prompts(stories_df=ssf_df, dim=dim, include_id=False)
      ssf_split_df[f"{dim}${prompt_col_suffix}$single_output_prompt"] = single_output_task_manager.get_prompts(stories_df=ssf_split_df, dim=dim, include_id=False)
    PdUtils.save_df(ssf_df, ssf_path)
    PdUtils.save_df(ssf_split_df, f"{config.dirs.data.corpus}/ssf_split.csv")

  for prompt_col_suffix in ALL_PROMPT_COL_SUFFIXES:
    dataset_dir = f"{config.dirs.data.ssf_gen_ft}/ssf-model-datasets/{prompt_col_suffix}/dataset"
    FtUtils.stage_llamafactory_finetuning(dims=taxonomy.get_dims(),
                                                    dataset_dir=dataset_dir,
                                                    df=ssf_split_df,
                                                    instruction_col_suffix=f"{prompt_col_suffix}$single_output_prompt",
                                                    output_col_prefix=openai_default_model_normalized,
                                                    output_col_suffixes=[f"gen{i}" for i in range(3)],
                                                    shuffle=True,
                                                    random_seed=config.random_seeds.default)
    FtUtils.run_finetuning_jobs(script_base_name=config.ft.ssf_gen_ft_script_base_name,
                                          output_base_dir=f"{config.dirs.data.ssf_gen_ft}/ft-models",
                                          dataset_dir=dataset_dir,
                                          disambiguator=prompt_col_suffix,
                                          do_eval=True,
                                          force_redo=config.force_redo.ssf_gen_ft)
    
  """
  SSF-Generator Inference Generation
  """
  for prompt_col_suffix in ALL_PROMPT_COL_SUFFIXES:
    df = ssf_df if prompt_col_suffix == PROMPT_COL_SUFFIX_FULL_CONTEXT else ssf_split_test_df
    df_path = ssf_path if prompt_col_suffix == PROMPT_COL_SUFFIX_FULL_CONTEXT else f"{config.dirs.data.corpus}/ssf_split_test.csv"  
    ssf_generator_path = f"{config.dirs.data.ssf_gen_ft}/ft-models/{config.ft.ssf_gen_ft_script_base_name}-train-{prompt_col_suffix}"
    generation_strategy = VllmGenerationStrategy(
      ModelConfig(model_name=ssf_generator_path),
      GenerationConfig(max_new_tokens=1000, max_model_len=2048 + 256),
      base_model_name=config.ft.base_model,
      lora_adapter_path=ssf_generator_path
    )
    single_output_task_manager = SingleOutputTaskManager(taxonomy=taxonomy,
                                                        generation_strategy=generation_strategy,
                                                        out_dir=f"{config.dirs.data.ssf_gen_inf}/{prompt_col_suffix}",
                                                        force_redo=False,
                                                        disambiguator=prompt_col_suffix,
                                                        context_config=config.context)
    single_output_task_manager.run_task(stories_df=df)
    df = single_output_task_manager.postprocess_results(stories_df=df)

    if prompt_col_suffix == PROMPT_COL_SUFFIX_FULL_CONTEXT:
      ssf_df = df
    else:
      ssf_split_test_df = df
    TorchUtils.clear_torch_memory(generation_strategy=generation_strategy)

  # Save after all iterations
  PdUtils.save_df(ssf_df, ssf_path)
  PdUtils.save_df(ssf_split_test_df, f"{config.dirs.data.corpus}/ssf_split_test.csv")

  # Add prompt_default gens (stored in ssf_df) to ssf_split_test_df
  ssf_df = pd.read_csv(ssf_path)
  prompt_default_gen_cols = [c for c in ssf_df.columns if c.startswith(f"{PROMPT_COL_SUFFIX_FULL_CONTEXT}$") and c.endswith("_gen0")]
  ssf_split_test_df = ssf_split_test_df.merge(
      ssf_df[['id'] + prompt_default_gen_cols],
      on='id',
      how='left'
  )
  PdUtils.save_df(ssf_split_test_df, f"{config.dirs.data.corpus}/ssf_split_test.csv")

  # Prep for last human eval of ssf-gen
  ssf_gen_eval_df = ssf_df[ssf_df['id'].isin(ssf_split_test_df['id'])]

  # Sort to match the order of ml_test_stories_final_df['id']
  ssf_gen_eval_df['id'] = pd.Categorical(
      ssf_gen_eval_df['id'],
      categories=ssf_split_test_df['id'],
      ordered=True
  )
  ssf_gen_eval_df = ssf_gen_eval_df.sort_values('id').reset_index(drop=True)

  swap_dims = []
  swapped_values = []
  for i, row in ssf_gen_eval_df.iterrows():
    id = row['id']
    dim = id_swap_dim_dict.get(id)
    if dim is None:
      raise ValueError(f"No swap dim found for story ID {id}")
    implausible_col = f"implausible_{openai_default_model_normalized}${dim}_gen0"
    implausible_value = ssf_split_test_df.iloc[i][implausible_col]
    plausible_col = f"prompt_default${dim}_gen0"
    plausible_value = row[plausible_col]
    ssf_gen_eval_df.at[i, plausible_col] = implausible_value
    swap_dims.append(dim)
    swapped_values.append(plausible_value)
  ssf_gen_eval_df['swap_value'] = swapped_values
  ssf_gen_eval_df['swap_dim'] = swap_dims

  # Rename all columns starting with "prompt_default$" by replacing "$" with "_". This is required for POTATO
  ssf_gen_eval_df.rename(
      columns={
          col: col.replace("$", "_")
          for col in ssf_gen_eval_df.columns
          if col.startswith("prompt_default$")
      },
      inplace=True
  )
  ssf_gen_eval_df.to_csv(f"{config.dirs.data.corpus}/ssf_gen_eval.csv", index=False)

  """
  Reference Inference Classification
  """
  ann_dir = f"{config.dirs.data.annotations}/tax_class"
  orchestrator = TaxClassStageOrchestrator(ann_dir=ann_dir,
                                           taxonomy=taxonomy,
                                           model_name=config.models.openai_tax_class,
                                            ssf_gen_base_model=config.models.openai_default,
                                           random_seed=config.random_seeds.default,
                                           show_progress=True)
  
  ref_val_output_dir = f"{config.dirs.data.ref_inf_class}/ref_val"
  ref_test_output_dir = f"{config.dirs.data.ref_inf_class}/ref_test"
  ref_train_output_dir = f"{config.dirs.data.ref_inf_class}/ref_train"
  os.makedirs(ref_val_output_dir, exist_ok=True)
  os.makedirs(ref_test_output_dir, exist_ok=True)
  os.makedirs(ref_train_output_dir, exist_ok=True)

  # Stage 1 (_ref_val)
  orchestrator.run_classification_with_evaluation(eval_split='val', 
                                                  dim_k_dict=config.tax_class.demo_counts, 
                                                  diversity_strategy=config.tax_class.sampling_strategy, 
                                                  force_redo=config.force_redo.ref_inf_class_val, 
                                                  output_dir=ref_val_output_dir, 
                                                  use_batch_api=False, 
                                                  random_seed=config.random_seeds.default)

  # Stage 2 (_ref_test)
  print("Starting Stage 2")
  orchestrator.run_classification_with_evaluation(eval_split='test', 
                                                  dim_k_dict=config.tax_class.demo_counts,
                                                  diversity_strategy=config.tax_class.sampling_strategy, 
                                                  force_redo=config.force_redo.ref_inf_class_test, 
                                                  output_dir=ref_test_output_dir, 
                                                  use_batch_api=True, 
                                                  random_seed=config.random_seeds.default)


  # # # Stage 3 (_ref_train - inference only)
  print("Starting Stage 3")
  stage3_classifier = BatchTaxonomyClassifier(taxonomy=taxonomy,
                                              few_shot_data=orchestrator.val_ann_df,
                                              model_name=config.models.openai_tax_class, 
                                              ssf_gen_base_model=config.models.openai_default,
                                              dim_k_dict=config.tax_class.demo_counts,
                                              diversity_strategy=config.tax_class.sampling_strategy,
                                              show_progress=False,
                                              similarity_index=orchestrator.similarity_index)

  InferenceUtils.run_stage_with_batch_handling(
    stage3_classifier.classify_generations,
    "Stage 3",
    stories_df=ssf_split_train_df,
    output_dir=ref_train_output_dir,
    force_redo=config.force_redo.ref_inf_class_train
  )

  output_col_prefix = "tax_class_zero_shot_output"
  ssf_split_train_df = postprocess_stage3(df=ssf_split_train_df,
                                           out_path=f"{ref_train_output_dir}/all_generations_outputs.jsonl",
                                           output_col_prefix=output_col_prefix,
                                           taxonomy=taxonomy)
  PdUtils.save_df(ssf_split_train_df, f"{config.dirs.data.corpus}/ssf_split_train.csv")


  os.makedirs(config.dirs.data.ssf_class_ft, exist_ok=True)

  # # Stage 4 (use _ref_train to create sft_model)
  print("Starting Stage 4")
  # prep output columns for finetuning
  instruction_col_suffix="tax_class_zero_shot_input_gen0"
  for dim in taxonomy.get_dims():
    response_col_name = f"{openai_default_model_normalized}_{dim}_gen0"
    instruction_col = f"{dim}${instruction_col_suffix}"
    ssf_split_train_df[instruction_col] = TaxonomyUtils.generate_prompts(
      texts=ssf_split_train_df[response_col_name].tolist(),
      dims=[dim] * len(ssf_split_train_df),
      taxonomy=taxonomy,
      few_shot_data=None,
      similarity_index=None,
      diversity_strategy=None,
      dim_k_dict=None,
      verbose=True
    )
  PdUtils.save_df(ssf_split_train_df, f"{config.dirs.data.corpus}/ssf_split_train.csv")

  # set up finetuning utils call
  dataset_dir_name = "tax_class_zero_shot_gen0"
  dataset_dir=f"{config.dirs.data.ssf_class_ft}/{dataset_dir_name}"
  
  FtUtils.stage_llamafactory_finetuning(dims=taxonomy.get_dims(),
                                                  dataset_dir=dataset_dir,
                                                  df=ssf_split_train_df,
                                                  instruction_col_suffix=instruction_col_suffix,
                                                  output_col_prefix=output_col_prefix,
                                                  output_col_suffixes=["gen0"],
                                                  shuffle=True,
                                                  random_seed=config.random_seeds.default)
  FtUtils.run_finetuning_jobs(script_base_name=config.ft.ssf_class_ft_script_base_name,
                                       output_base_dir=f"{config.dirs.data.ssf_class_ft}/ft-models",
                                       dataset_dir=dataset_dir,
                                       disambiguator="zero-shot-gen0",
                                       do_eval=False,
                                       force_redo=config.force_redo.ssf_class_ft)

  # # Stage 5 (eval sft_model on _ref_val) + Stage 6 (eval sft_model on _ref_test)
  print("Starting Stages 5 and 6")
  ssf_tax_classifier_path = f"{config.dirs.data.ssf_class_ft}/ft-models/{config.ft.ssf_class_ft_script_base_name}-train-zero-shot-gen0"
  generation_strategy = VllmGenerationStrategy(
    ModelConfig(model_name=ssf_tax_classifier_path),
    GenerationConfig(max_new_tokens=1000, max_model_len=2048 + 256),
    base_model_name=config.ft.base_model,
    lora_adapter_path=ssf_tax_classifier_path
  )

  tups = [
      (ssf_split_val_df, orchestrator.val_ann_df, f"{config.dirs.data.ssf_class_inf}/_tax_class_outputs/val", config.force_redo.ssf_class_val),
      (ssf_split_test_df, orchestrator.test_ann_df, f"{config.dirs.data.ssf_class_inf}/_tax_class_outputs/test", config.force_redo.ssf_class_test)
  ]
  for ml_df, eval_df_full, dir_name, force_redo in tups:
    for dim in taxonomy.get_dims():
      prompts_path = f"{dir_name}/prompts/{dim}.jsonl"
      outputs_path = f"{dir_name}/outputs/{dim}.jsonl"

      if os.path.exists(outputs_path) and os.path.exists(prompts_path) and not force_redo:
        print(f"Using existing results for {dir_name}, dimension {dim}")
        continue

      eval_df = eval_df_full[eval_df_full['dim'] == dim]
      ml_df = ml_df.copy()
      ml_df_filtered = filter_and_align_dataframes(target_df=ml_df, reference_df=eval_df, verbose=False)

      response_col_name = f"{openai_default_model_normalized}_{dim}_gen0"
      instruction_col = f"{dim}${instruction_col_suffix}"
      ml_df_filtered[instruction_col] = TaxonomyUtils.generate_prompts(
        texts=ml_df_filtered[response_col_name].tolist(),
        dims=[dim] * len(ml_df_filtered),
        taxonomy=taxonomy,
        few_shot_data=None,
        similarity_index=None,
        diversity_strategy=None,
        dim_k_dict=None,
        verbose=True
      )
      prompts_list = ml_df_filtered[instruction_col].tolist()
      InferenceUtils.save_prompts_as_jsonl(prompts_list, prompts_path)
      generation_strategy.generate(input_path=prompts_path,
                                  output_path=outputs_path)
  
    taxonomy_evaluator = TaxonomyEvaluator(taxonomy=taxonomy)
    dim_dict = taxonomy_evaluator.get_dim_eval_inputs_from_dir(pred_dir=dir_name,
                                                               ground_truth_data=eval_df_full)
  
    all_predictions = {}
    for dim, data in dim_dict.items():
      if dim not in all_predictions:
          all_predictions[dim] = {'refs': [], 'preds': [], 'texts': []}
      all_predictions[dim]['refs'].extend(data['refs'])
      all_predictions[dim]['preds'].extend(data['preds'])
      all_predictions[dim]['texts'].extend(data['texts'])
    
    # Save predictions        
    results_cache_path = f"{dir_name}/final_results.json"
    predictions_cache_path = f"{dir_name}/all_predictions.json"

    print("Evaluating results...")
    final_results = taxonomy_evaluator.evaluate_all_predictions(all_predictions)

    # Cache results for future use
    with open(results_cache_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    with open(predictions_cache_path, 'w') as f:
        json.dump(all_predictions, f, indent=2)


  # Ensure ssf_df order is aligned with canonical stories.csv order
  # This is critical because cached JSONL outputs assume this order
  print("Aligning ssf_df with canonical story order from stories.csv...")
  canonical_stories_path = "/usr2/jmire/ssf/notebooks/stories.csv"
  canonical_stories_df = pd.read_csv(canonical_stories_path, usecols=['id'])
  ssf_df = filter_and_align_dataframes(target_df=ssf_df, reference_df=canonical_stories_df, verbose=True)
  PdUtils.save_df(ssf_df, ssf_path)
  print(f"✓ Successfully aligned and saved {len(ssf_df)} stories with canonical order")

  # Stage 7: ssf_classifier -> inference on full stories set (full context)
  print("Starting Stage 7")
  output_base_dir = f"{config.dirs.data.ssf_class_inf}/_tax_class_outputs/all"
  prompt_col_suffix = PROMPT_COL_SUFFIX_FULL_CONTEXT
  for dim in taxonomy.get_dims():
    for gen_idx in ["gen0"]:
      output_col = f"{prompt_col_suffix}${dim}_{gen_idx}_cats"
      prompts_path = f"{output_base_dir}/prompts/{dim}_{gen_idx}.jsonl"
      outputs_path = f"{output_base_dir}/outputs/{dim}_{gen_idx}.jsonl"
      if not (os.path.exists(prompts_path) and os.path.exists(outputs_path)) or config.force_redo.ssf_class_all: 
        print(f"Running inference for {dim}_{gen_idx}")
        os.makedirs(os.path.dirname(prompts_path), exist_ok=True)
        os.makedirs(os.path.dirname(outputs_path), exist_ok=True)

        # Create input column for prompts
        instruction_col = f"{prompt_col_suffix}${dim}_{gen_idx}_cats_input"
        response_col_name = f"{prompt_col_suffix}${dim}_{gen_idx}"
        # Generate prompts for this dimension and generation
        ssf_df[instruction_col] = TaxonomyUtils.generate_prompts(
          texts=ssf_df[response_col_name].tolist(),
          dims=[dim] * len(ssf_df),
          taxonomy=taxonomy,
          few_shot_data=None,
          similarity_index=None,
          diversity_strategy=None,
          dim_k_dict=None,
          verbose=False
        )
        # Save prompts and run inference
        prompts_list = ssf_df[instruction_col].tolist()
        InferenceUtils.save_prompts_as_jsonl(prompts_list, prompts_path)
        generation_strategy.generate(input_path=prompts_path, output_path=outputs_path)
      else:
        print(f"Using existing results for {dim}_{gen_idx}")
      
      # Load predictions and parse them
      predictions = InferenceUtils.read_jsonl(outputs_path)
      # Parse predictions and store in output column
      parsed_predictions = []
      for output in predictions:
        output_text = output['output']
        if output_text == NO_OP_MSG:
          print("Don't think this should happen...")
          parsed_predictions.append(None)  # Use None for missing values, not empty list
        else:
          try:
            parsed = InferenceUtils.parse_json(
              output_text.lstrip("```json").rstrip("```").strip()
            )
            categories = parsed['response']
            if not isinstance(categories, list):
              categories = [categories] if categories else []
            parsed_predictions.append(categories)
          except Exception as e:
            print(f"Error parsing prediction for {dim}_{gen_idx}: {e}")
            parsed_predictions.append(None)  # Use None for parse errors too
      
      ssf_df[output_col] = parsed_predictions

  # Save updated stories_df with both input and output columns
  PdUtils.save_df(ssf_df, ssf_path)