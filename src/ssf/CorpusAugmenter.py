from convokit import Corpus, TextCleaner
from ssf.Constants import *
from ssf.Exceptions import *
from ssf.SelectorType import SelectorType
import os
import inspect
from ssf.ck_transformers.AncestralUttIdsTransformer import AncestralUttIdsTransformer
from ssf.ck_transformers.PrevUttIdsTransformer import PrevUttIdsTransformer
from ssf.ck_transformers.PerspectiveTransformer import PerspectiveTransformer
from ssf.ck_transformers.StoryProbTransformer import StoryProbTransformer
from ssf.ck_transformers.SummaryTransformer import SummaryTransformer
from ssf.ck_transformers.CommunityDescriptionContextTransformer import CommunityDescriptionContextTransformer
from ssf.ck_transformers.CommunityValuesContextTransformer import CommunityValuesContextTransformer
from ssf.ck_transformers.ProgenitorContextTransformer import ProgenitorContextTransformer
from ssf.ck_transformers.AncestralContextTransformer import AncestralContextTransformer
from ssf.ck_transformers.PreviousContextTransformer import PreviousContextTransformer
from ssf.ck_transformers.ConversationContextTransformer import ConversationContextTransformer

class CorpusAugmenter:
  def __init__(self, corpus_utils, corpus_dir):
      self.corpus_utils = corpus_utils
      self.corpus_dir = corpus_dir

  def apply_ck_transformer_pipeline(self, 
                                      corpus, 
                                      community_allowlist,
                                      community_data_dict,
                                      generation_strategy,
                                      force_redo_corpus):
      # Apply transformer pipeline for data cleaning, utterance summarization, context selection, context summarization, and context consolidation
      global_transformer_tuples = [
          (TextCleaner(save_original=True, verbosity=0), ORIGINAL_UTT_TEXT_META_KEY),
          (AncestralUttIdsTransformer(), ANCESTRAL_UTT_IDS_META_KEY),
          (PrevUttIdsTransformer(), PREV_UTT_IDS_META_KEY), 
          (StoryProbTransformer(), STORY_SEEKER_META_KEY),
      ]
      all_selector = self.corpus_utils.get_selector(SelectorType.ALL, corpus=corpus, community_allowlist=community_allowlist)
      corpus = self._apply_transformers(transformer_tuples=global_transformer_tuples, 
                                          corpus=corpus, 
                                          corpus_dir=self.corpus_dir,
                                          selector=all_selector)

      stories_superset_transformer_tuples = [
          (PerspectiveTransformer(gcloud_api_key=os.getenv(GCLOUD_API_KEY_ENV_VAR_NAME)), PERSPECTIVE_META_KEY),
      ]
      stories_superset_selector = self.corpus_utils.get_selector(SelectorType.STORIES_SUPERSET, corpus=corpus, community_allowlist=community_allowlist)
      corpus = self._apply_transformers(transformer_tuples=stories_superset_transformer_tuples, 
                                          corpus=corpus, 
                                          corpus_dir=self.corpus_dir,
                                          selector=stories_superset_selector)

      stories_and_contexts_transformer_tuples = [
          (SummaryTransformer(max_sentences=self.corpus_utils.config.convo_context.max_utt_summary_sentences, 
                              generation_strategy=generation_strategy, 
                              dir=self.corpus_dir, 
                              force_rebuild=force_redo_corpus), SUMMARY_META_KEY),
      ]
      stories_and_contexts_selector = self.corpus_utils.get_selector(SelectorType.STORIES_AND_CONTEXTS, corpus=corpus, community_allowlist=community_allowlist)
      corpus = self._apply_transformers(transformer_tuples=stories_and_contexts_transformer_tuples, 
                                          corpus=corpus, 
                                          corpus_dir=self.corpus_dir,
                                          selector=stories_and_contexts_selector)
      
      stories_transformer_tuples = [
          (CommunityDescriptionContextTransformer(community_data_dict=community_data_dict), COMMUNITY_DESCRIPTION_META_KEY),
          (CommunityValuesContextTransformer(community_data_dict=community_data_dict), COMMUNITY_VALUES_META_KEY),
          (ProgenitorContextTransformer(dir=self.corpus_dir, force_rebuild=force_redo_corpus,
                                        generation_strategy=generation_strategy), PROGENITOR_CONTEXT_META_KEY),
          (AncestralContextTransformer(dir=self.corpus_dir,
                                      force_rebuild=force_redo_corpus,
                                      k=self.corpus_utils.config.convo_context.context_chain_utts,
                                      generation_strategy=generation_strategy,
                                      corpus_utils=self.corpus_utils), ANCESTRAL_CONTEXT_META_KEY),
          (PreviousContextTransformer(dir=self.corpus_dir,
                                      force_rebuild=force_redo_corpus,
                                      k=self.corpus_utils.config.convo_context.context_chain_utts,
                                      generation_strategy=generation_strategy,
                                      corpus_utils=self.corpus_utils), PREVIOUS_CONTEXT_META_KEY),
          (ConversationContextTransformer(generation_strategy=generation_strategy,
                                          dir=self.corpus_dir,
                                          redo=force_redo_corpus), CONVERSATION_CONTEXT_META_KEY),
      ]
      stories_selector = self.corpus_utils.get_selector(SelectorType.STORIES, corpus=corpus, community_allowlist=community_allowlist)
      corpus = self._apply_transformers(transformer_tuples=stories_transformer_tuples, 
                                          corpus=corpus,
                                          corpus_dir=self.corpus_dir,
                                          selector=stories_selector)
      corpus.print_summary_stats()
      return corpus

  def _apply_transformers(self, transformer_tuples, corpus, corpus_dir, selector=None):
      for ck_transformer, ck_transformer_meta_key in transformer_tuples:
          print(f'Starting ck_transformer: {ck_transformer.__class__.__name__}')
          if not self._ck_transformer_applied(corpus, ck_transformer_meta_key, selector=selector):
              if "selector" in inspect.signature(ck_transformer.fit).parameters:
                  ck_transformer.fit(corpus, selector=selector)
              else:
                  ck_transformer.fit(corpus)
              if "selector" in inspect.signature(ck_transformer.transform).parameters:
                  corpus = ck_transformer.transform(corpus, selector=selector)
              else:
                  corpus = ck_transformer.transform(corpus)
              self.corpus_utils.save_corpus(corpus, base_path=corpus_dir)
          print(f'Finished ck_transformer: {ck_transformer.__class__.__name__}')
      return corpus

  def _ck_transformer_applied(self, corpus: Corpus, transformer_meta_key: str, selector=None):
      for utt in corpus.iter_utterances(selector=selector):
          return transformer_meta_key in utt.get_meta()