from convokit import Transformer, Corpus
from ssf.helpers import get_progenitor_utt
from ssf.Constants import *
from ssf.prompt_builders.ProgenitorContextConsolidationPromptBuilder import ProgenitorContextConsolidationPromptBuilder
from ssf.utils import InferenceUtils
from ssf.Exceptions import *
import os
from ssf.generation_strategies.GenerationStrategy import GenerationStrategy
from tqdm import tqdm

class ProgenitorContextTransformer(Transformer):
  def __init__(self,
               dir,
               force_rebuild,
               generation_strategy: GenerationStrategy,
               meta_key=PROGENITOR_CONTEXT_META_KEY,
               summary_meta_key=SUMMARY_META_KEY):
    self.dir = dir
    self.force_rebuild = force_rebuild
    self.meta_key = meta_key
    self.summary_meta_key=summary_meta_key
    self.generation_strategy = generation_strategy

  def _get_prompts(self, top_level_utts, utt_texts, utt_is_progenitor_flags, corpus):
    prompts = []
    for top_level_utt, utt_text, utt_is_progenitor_flag in zip(top_level_utts, utt_texts, utt_is_progenitor_flags):
      convo = corpus.get_conversation(top_level_utt.conversation_id)
      progenitor_title = convo.get_meta()['title'] if (top_level_utt.text != utt_text and top_level_utt.text.strip() != "") else None       # TODO - fix
      progenitor_summary = top_level_utt.get_meta()[self.summary_meta_key] if (top_level_utt.text != utt_text and top_level_utt.text.strip() != "") else None

      prompt = (ProgenitorContextConsolidationPromptBuilder(cur_utt_is_progenitor=utt_is_progenitor_flag, corpus_manager=self)
                .progenitor_title(progenitor_title)
                .progenitor_summary(progenitor_summary)
                .build())
      
      prompts.append(prompt)
    return prompts

  def fit(self, corpus: Corpus, selector=None):
    return
  
  def transform(self, corpus: Corpus, selector=None):
    prompts_path = f'{self.dir}/prompts/progenitorcontext.jsonl'
    outputs_path = f'{self.dir}/outputs/progenitorcontext.jsonl'

    if not (os.path.exists(prompts_path) and os.path.exists(outputs_path)) or self.force_rebuild:
      top_level_utts, utt_texts, utt_is_progenitor_flags = [], [], []
      for utt in tqdm(corpus.iter_utterances(selector=selector), total=len(corpus.utterances)):
        progenitor_utt = get_progenitor_utt(utt, corpus)
        top_level_utts.append(progenitor_utt)
        utt_texts.append(utt.text)
        utt_is_progenitor_flags.append(utt.id == progenitor_utt.id)
      
      prompts = self._get_prompts(top_level_utts=top_level_utts,
                                  utt_texts=utt_texts, 
                                  utt_is_progenitor_flags=utt_is_progenitor_flags,
                                  corpus=corpus)
      
      InferenceUtils.save_prompts_as_jsonl(prompts, prompts_path)
      self.generation_strategy.generate(prompts_path, outputs_path)

    outputs = InferenceUtils.read_jsonl(outputs_path)
    for utt, output in zip(corpus.iter_utterances(selector=selector), outputs):
      progenitor_context = InferenceUtils.remove_template_markers(output['output'])
      utt.add_meta(key=self.meta_key, 
                  value=progenitor_context)
      
    return corpus
  