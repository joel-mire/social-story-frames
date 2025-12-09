from ssf.generation_strategies.configs import ModelConfig, GenerationConfig
from ssf.generation_strategies import GenerationStrategy
import os
from vllm import LLM, SamplingParams
from typing import List, Dict
from ssf.utils import InferenceUtils
from ssf.Constants import *
from transformers import AutoTokenizer
from vllm.lora.request import LoRARequest

def prep_messages(prompts: List[Dict[str, str]], tokenizer) -> List[str]:
    formatted_prompts = []
    for item in prompts:
      raw_prompt = item["prompt"]
      messages = [{"role": "user", "content": raw_prompt}]
      formatted_prompt = tokenizer.apply_chat_template(messages, 
                                                       tokenize=False, 
                                                       add_generation_prompt=True)
      formatted_prompts.append(formatted_prompt)
    return formatted_prompts

class VllmGenerationStrategy(GenerationStrategy):
  def __init__(self,
               model_config: ModelConfig,
               generation_config: GenerationConfig,
               base_model_name: str,
               lora_adapter_path=None):
    self.model_config = model_config
    self.generation_config = generation_config
    self.lora_adapter_path = lora_adapter_path
    # Skip validation - vLLM handles both local paths and HuggingFace IDs
    
    self.model = LLM(
      model=base_model_name,
      enable_lora=bool(self.lora_adapter_path),
      trust_remote_code=True,
      max_model_len=self.generation_config.max_model_len,
      disable_log_stats=False)

    # Load tokenizer to get stop tokens and for chat formatting
    self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    stop_token_ids = []
    if self.tokenizer.eos_token_id:
      stop_token_ids.append(self.tokenizer.eos_token_id)
    if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id:
      stop_token_ids.append(self.tokenizer.pad_token_id)
    
    self.sampling_params = SamplingParams(
      temperature=self.generation_config.temperature,
      top_p=self.generation_config.top_p,
      max_tokens=self.generation_config.max_new_tokens,
      stop_token_ids=stop_token_ids if stop_token_ids else None,
      stop=["</s>", "[/INST]"],
      skip_special_tokens=True)

    self.lora_request = (LoRARequest("adapter", 1, self.lora_adapter_path)
                         if self.lora_adapter_path else None)

  def process_batch(self, 
                    batch: List[Dict[str, str]]) -> List[str]:
    messages = prep_messages(batch, self.tokenizer)

    # Filter out placeholder messages
    valid_indices = [i for i, msg in enumerate(messages) if msg != NO_OP_MSG]
    valid_messages = [messages[i] for i in valid_indices]
    if not valid_messages:
      return [NO_OP_MSG] * len(messages)
    
    # Conditionally pass lora_request only if using LoRA
    generate_kwargs = {
      'prompts': valid_messages,
      'sampling_params': self.sampling_params
    }
    if self.lora_request:
      generate_kwargs['lora_request'] = self.lora_request

    outputs = self.model.generate(**generate_kwargs)
    valid_responses = [output.outputs[0].text for output in outputs]
    
    # Map responses back to original indices
    response_map = {}
    for i, msg in enumerate(messages):
      if msg == NO_OP_MSG:
        response_map[i] = NO_OP_MSG
      else:
        valid_idx = valid_indices.index(i)
        response_map[i] = valid_responses[valid_idx]
    return [response_map[i] for i in range(len(messages))]

  def _generate(self, prompts: List[Dict[str, str]]) -> List[str]:
      responses = []
      for i in range(0, len(prompts), self.generation_config.batch_size):
        batch = prompts[i:i + self.generation_config.batch_size]
        batch_responses = self.process_batch(batch)
        responses.extend(batch_responses)
      return responses
  
  def generate(self, 
               input_path: str, 
               output_path: str):
    prompts = InferenceUtils.read_jsonl(input_path)
    responses = self._generate(prompts)
    InferenceUtils.write_results(output_path, prompts, responses)