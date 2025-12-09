import os
import json
import random
import shutil
import subprocess
from ssf.Constants import *

def stage_llamafactory_finetuning(dims, 
                                   dataset_dir, 
                                   df, 
                                   instruction_col_suffix, 
                                   output_col_prefix, 
                                   output_col_suffixes,
                                   shuffle=False,
                                   random_seed=None):
  os.makedirs(dataset_dir, exist_ok=True)

  dataset_info = {}
  for split in ['train', 'val', 'test']:
    dataset_split_path = f"{dataset_dir}/{split}.json"
    df_split = df[df['split'] == split]
    if len(df_split) == 0:
      continue
    _create_llamafactory_dataset(dims=dims,
                                 df_split=df_split, 
                                 dataset_split_path=dataset_split_path, 
                                 instruction_col_suffix=instruction_col_suffix,
                                 output_col_prefix=output_col_prefix,
                                 output_col_suffixes=output_col_suffixes,
                                 shuffle=shuffle,
                                 random_seed=random_seed)
    dataset_info[split] = {
        "file_name": f"{split}.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
  with open(f"{dataset_dir}/dataset_info.json", 'w') as f:
      json.dump(dataset_info, f, indent=2)

def _create_llamafactory_dataset(dims, 
                                 df_split, 
                                 dataset_split_path, 
                                 instruction_col_suffix, 
                                 output_col_prefix,
                                 output_col_suffixes,
                                 shuffle=False,
                                 random_seed=None):
  formatted_data = []
  for dim in dims:
    instruction_col = f"{dim}${instruction_col_suffix}"
    for _, row in df_split.iterrows():
      instruction = row[instruction_col]
      for output_col_suffix in output_col_suffixes:
        output_col = f"{output_col_prefix}_{dim}_{output_col_suffix}"
        output_val = row[output_col]
        if output_val == "": 
          continue
        output = json.dumps({"response": output_val})
        formatted_data.append({
          "instruction": instruction,
          "input": "",
          "output": output,
          "id": row['id'],
          "dim": dim
        })
  if shuffle:
    if random_seed is None:
      raise ValueError("random_seed must be provided if shuffle is True")
    random.seed(random_seed)
    random.shuffle(formatted_data)
  with open(dataset_split_path, 'w') as f:
    json.dump(formatted_data, f, indent=2)

def run_finetuning_jobs(script_base_name, output_base_dir, dataset_dir, disambiguator, do_eval=True, force_redo=False):
  os.makedirs(output_base_dir, exist_ok=True)
  # Define output directories
  train_output_dir = f"{output_base_dir}/{script_base_name}-train-{disambiguator}"
  eval_output_dir = f'{output_base_dir}/{script_base_name}-eval-{disambiguator}'

  if not os.path.exists(train_output_dir) or force_redo:
    if os.path.exists(train_output_dir):
      shutil.rmtree(train_output_dir)
    # Train
    train_script_path = f"{SCRIPTS_DIR}/train/{script_base_name}.sh"
    train_args = [dataset_dir, train_output_dir]
    print(f"Training {script_base_name}")
    _run_subprocess(train_script_path, train_args)
  else:
    print(f"Using cached train job: {script_base_name} - {disambiguator}")

  if do_eval:
    if not os.path.exists(eval_output_dir) or force_redo:
      if os.path.exists(eval_output_dir):
        shutil.rmtree(eval_output_dir)
      # Eval
      eval_script_path = f"{SCRIPTS_DIR}/eval/{script_base_name}.sh"
      eval_args = [dataset_dir, eval_output_dir, train_output_dir]
      print(f"Eval {script_base_name}")
      _run_subprocess(eval_script_path, eval_args)
    else:
      print(f"Using cached eval job: {script_base_name} - {disambiguator}")

def _run_subprocess(script_path, args):
  try:
    process = subprocess.run([script_path, *args],
                             check=True,
                             text=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    print(f"Task completed successfully.")
    print(process.stdout)
    return 0
  except subprocess.CalledProcessError as e:
    print(f"Task failed with error code {e.returncode}")
    print(f"Error output: {e.stderr}")
    return e.returncode