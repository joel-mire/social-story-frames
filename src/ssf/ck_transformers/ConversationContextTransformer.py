from convokit import Transformer, Corpus
from ssf.Constants import *
from ssf.utils import InferenceUtils
from ssf.Exceptions import *
import os
from ssf.generation_strategies.GenerationStrategy import GenerationStrategy
from tqdm import tqdm
from ssf.prompt_builders.ConversationContextConsolidationPromptBuilder import ConversationContextConsolidationPromptBuilder

class ConversationContextTransformer(Transformer):
  def __init__(self,
               generation_strategy: GenerationStrategy,
               dir,
               redo,
               meta_key=CONVERSATION_CONTEXT_META_KEY,
               summary_meta_key=SUMMARY_META_KEY,
               ancestral_context_meta_key=ANCESTRAL_CONTEXT_META_KEY,
               previous_context_meta_key=PREVIOUS_CONTEXT_META_KEY):
    self.dir = dir
    self.redo = redo
    self.meta_key = meta_key
    self.summary_meta_key=summary_meta_key
    self.ancestral_context_meta_key = ancestral_context_meta_key
    self.previous_context_meta_key = previous_context_meta_key
    self.generation_strategy = generation_strategy

  def _get_prompts(self, ancestral_contexts, previous_contexts):
    prompts = []
    for ancestral_context, previous_context in zip(ancestral_contexts, previous_contexts):
      prompt = (ConversationContextConsolidationPromptBuilder()
                                              .ancestral_summary(ancestral_context if ancestral_context != NO_OP_MSG else None)
                                              .preceding_peers_summary(previous_context if previous_context != NO_OP_MSG else None)
                                              .build())
      prompts.append(prompt)
    return prompts

  def fit(self, corpus: Corpus, selector=None):
    return
  
  def transform(self, corpus: Corpus, selector=None):
    prompts_path = f'{self.dir}/prompts/conversationcontext.jsonl'
    outputs_path = f'{self.dir}/outputs/conversationcontext.jsonl'

    if not (os.path.exists(prompts_path) and os.path.exists(outputs_path)) or self.redo:
      ancestral_contexts, previous_contexts, utt_texts = [], [], []
      for utt in tqdm(corpus.iter_utterances(selector=selector), total=len(corpus.utterances)):
        ancestral_contexts.append(utt.get_meta()[self.ancestral_context_meta_key])
        previous_contexts.append(utt.get_meta()[self.previous_context_meta_key])
        utt_texts.append(utt.text)
      
      prompts = self._get_prompts(ancestral_contexts=ancestral_contexts,
                                  previous_contexts=previous_contexts)
      
      InferenceUtils.save_prompts_as_jsonl(prompts, prompts_path)
      self.generation_strategy.generate(prompts_path, outputs_path)

    outputs = InferenceUtils.read_jsonl(outputs_path)
    for utt, output in zip(corpus.iter_utterances(selector=selector), outputs):
      consolidated_context = InferenceUtils.remove_template_markers(output['output'])
      utt.add_meta(key=self.meta_key, 
                  value=consolidated_context)
      
    return corpus
  