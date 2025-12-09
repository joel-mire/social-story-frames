from ssf.generation_strategies.configs import ModelConfig, GenerationConfig
from ssf.generation_strategies import GenerationStrategy
from typing import List, Dict
from openai import OpenAI
from ssf.utils import InferenceUtils
from tqdm import tqdm
from ssf.Constants import *

def prep_messages(prompts: List[Dict[str, str]]) -> List[Dict[str, str]]:
  return [{"role": "user", "content": item["prompt"]} for item in prompts]

class OpenaiGenerationStrategy(GenerationStrategy):
  def __init__(self,
               model_config: ModelConfig,
               generation_config: GenerationConfig,
               show_progress: bool = True):
    self.model_config = model_config
    self.generation_config = generation_config
    self.show_progress = show_progress
    self.client = OpenAI()
  
  def _generate(self, messages: List[Dict[str, str]]):
    responses = []
    iterator = tqdm(messages) if self.show_progress else messages
    for message in iterator:
      if message['content'] == NO_OP_MSG:
        responses.append(NO_OP_MSG)
      elif message['content'].startswith(SKIP_INFERENCE_PREFIX):
        message['content'] = message['content'][len(SKIP_INFERENCE_PREFIX):]
        responses.append(message['content'])
      else:
        chat_completion = self.client.chat.completions.create(
          messages=[message],
          model=self.model_config.model_name,
          temperature=self.generation_config.temperature,
          top_p=self.generation_config.top_p,
          max_tokens=self.generation_config.max_new_tokens
        )
        responses.append(chat_completion.choices[0].message.content)
    return responses
    
  def generate(self, input_path: str, output_path: str):
    prompts = InferenceUtils.read_jsonl(input_path)
    messages = prep_messages(prompts)
    responses = self._generate(messages)
    InferenceUtils.write_results(output_path, prompts, responses)