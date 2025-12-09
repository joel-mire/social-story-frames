import os
from convokit import Transformer, Corpus
from ssf.helpers import get_utt_texts
from ssf.Constants import *
from ssf.generation_strategies.GenerationStrategy import GenerationStrategy
from ssf.utils import InferenceUtils
from ssf.Exceptions import *

class SummaryTransformer(Transformer):

  def __init__(self, 
               max_sentences,
               generation_strategy: GenerationStrategy,
               dir,
               force_rebuild,
               meta_key=SUMMARY_META_KEY):
    self.max_sentences = max_sentences
    self.meta_key = meta_key
    self.dir = dir
    self.generation_strategy = generation_strategy
    self.force_rebuild = force_rebuild

  def _get_prompts(self, utt_texts):
    prompts = []
    for utt_text in utt_texts:
      prompts.append(f"The following text comes from a social media forum. Summarize the text in a maximum of {self.max_sentences} sentences. Do not hallucinate and do not say that the text is too short to summarize.\n\n{utt_text}")
    return prompts

  def fit(self, corpus: Corpus, selector=None):
    return
  
  def transform(self, corpus: Corpus, selector=None):
    prompts_path = f'{self.dir}/prompts/summary{self.max_sentences}.jsonl'
    outputs_path = f'{self.dir}/outputs/summary{self.max_sentences}.jsonl'
    
    print(prompts_path, outputs_path)
    if not (os.path.exists(prompts_path) and os.path.exists(outputs_path)) or self.force_rebuild:
      utt_texts = get_utt_texts(corpus, selector=selector)
      prompts = self._get_prompts(utt_texts)
      InferenceUtils.save_prompts_as_jsonl(prompts, prompts_path)
      self.generation_strategy.generate(prompts_path, outputs_path)
    outputs = InferenceUtils.read_jsonl(outputs_path)
    for utt, output in zip(corpus.iter_utterances(selector=selector), outputs):
      utt.add_meta(key=self.meta_key, 
                   value=output['output'])
    return corpus