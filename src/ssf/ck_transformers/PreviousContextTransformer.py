from convokit import Transformer, Corpus
from ssf.helpers import CorpusHelper
from ssf.Constants import *
from ssf.utils import InferenceUtils
from ssf.Exceptions import *
import os
from ssf.generation_strategies.GenerationStrategy import GenerationStrategy

class PreviousContextTransformer(Transformer):

  def __init__(self, 
               dir,
               force_rebuild,
               k,
               generation_strategy: GenerationStrategy,
               corpus_utils: CorpusHelper,
               meta_key=PREVIOUS_CONTEXT_META_KEY):
    self.dir = dir
    self.force_rebuild = force_rebuild
    self.k = k
    self.meta_key = meta_key
    self.corpus_utils = corpus_utils
    self.generation_strategy = generation_strategy

  def _get_prompts(self, summaries_list):
    prompts = []
    for summaries in summaries_list:
      summary_count = len(summaries)
      if summary_count == 0:
        prompts.append(NO_OP_MSG)
        continue
      prompt = f"Below are {summary_count} summaries of a chain of social media comments under a single parent post/comment. Your task is to generate a global summary of the overall chain based on the local summaries in three sentences or less."
      for summary in summaries:
        prompt += f"\n- {summary}"
      prompts.append(prompt)
    return prompts

  def fit(self, corpus: Corpus, selector=None):
    return
  
  def transform(self, corpus: Corpus, selector=None):
    prompts_path = f'{self.dir}/prompts/previouscontext{self.k}.jsonl'
    outputs_path = f'{self.dir}/outputs/previouscontext{self.k}.jsonl'
    
    if not (os.path.exists(prompts_path) and os.path.exists(outputs_path)) or self.force_rebuild:
      utt_summaries_list = self.corpus_utils.get_k_closest_substantial_context_utt_summaries_list(corpus=corpus,
                                                                                                  context_utt_meta_key=PREV_UTT_IDS_META_KEY,
                                                                                                  summary_meta_key=SUMMARY_META_KEY,
                                                                                                  k=self.k,
                                                                                                  selector=selector)
      
      prompts = self._get_prompts(utt_summaries_list)
      InferenceUtils.save_prompts_as_jsonl(prompts, prompts_path)
      self.generation_strategy.generate(prompts_path, outputs_path)

    outputs = InferenceUtils.read_jsonl(outputs_path)
    for utt, output in zip(corpus.iter_utterances(selector=selector), outputs):
      utt.add_meta(key=self.meta_key, 
                   value=output['output'])
    return corpus
  