from typing import List, Dict
import json
import os
import re

def read_jsonl(path: str) -> List[Dict[str, str]]:
  with open(path, "r") as f:
    return [json.loads(line) for line in f]

def write_results(path: str, prompts: List[str], outputs: List[str]):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    for prompt, output in zip(prompts, outputs):
      # Handle both dict format (with id) and string format
      if isinstance(prompt, dict):
        result = {"id": prompt['id'], "prompt": prompt['prompt'], "output": output}
      else:
        result = {"prompt": prompt, "output": output}
      f.write(json.dumps(result) + "\n")

def save_prompts_as_jsonl(prompts, path):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, 'w', encoding='utf-8') as file:
    for prompt in prompts:
      # Handle both dict format (with id) and string format (backward compatibility)
      if isinstance(prompt, dict):
        json_line = prompt  # Already has 'id' and 'prompt' fields
      else:
        json_line = {"prompt": prompt}
      file.write(json.dumps(json_line) + '\n')

def remove_template_markers(text, template_start_marker='<TEMPLATE>', template_end_marker='</TEMPLATE>'):
    if text.startswith(template_start_marker):
      text = text[len(template_start_marker):]
    if text.endswith(template_end_marker):
      text = text[:-len(template_end_marker)]
    return text

def parse_json(s):
  """Intended for GenerationStrategy outputs in the format {"response": "..."}"""
  try:
      return json.loads(s)
  except json.JSONDecodeError as e:
      # print(s)
      # Fix common escaping issues (use raw strings)
      s = s.replace(r"\$", "$")
      s = s.replace(r"\&", "&")
      s = s.replace(r"\*", "*")
      s = s.replace(r"\_", "_")
      s = s.replace(r"\u00", "°")
      s = s.replace("\\(", "(")
      s = s.replace("\\)", ")")
      s = s.replace("\\:", ":")
      s = s.replace("'", r"\'")
      s = s.replace("{{", "{").replace("}}", "}") # Fix double braces that may break parsing
      
      # Handle missing closing brace
      if s.strip().endswith('"') and not s.strip().endswith('"}'):
          s = s.strip() + "}"

      # Handle missing value cases like {"narr_future_action"}
      lone_key_match = re.match(r'^\s*\{\s*"(\w+)"\s*\}\s*$', s)
      if lone_key_match:
        key = lone_key_match.group(1)
        return {"response": key}
      
      # Escape unescaped inner quotes in value positions only
      try:
        match = re.match(r'^\s*\{\s*"(\w+)"\s*:\s*"(.*)"\s*\}\s*$', s)
        if match:
          key, val = match.group(1), match.group(2)
          val = val.replace('\\"', '"')  # Unescape first
          val = val.replace('"', r'\"')  # Re-escape double quotes
          val = val.replace("'", r"\'")  # Escape single quotes
          return json.loads(f'{{"{key}": "{val}"}}')
      except Exception:
        pass

      raise ValueError("Input string is not a valid JSON")

def run_stage_with_batch_handling(stage_func, stage_name, **kwargs):
  """Helper to run a stage with proper batch API error handling."""
  try:
    return stage_func(**kwargs)
  except Exception as e:
    if "BatchNotReadyError" in str(type(e)) or "Batch job" in str(e):
      print(f"{stage_name} batch jobs submitted. Please run again later.")
      exit(0)
    else:
      raise e