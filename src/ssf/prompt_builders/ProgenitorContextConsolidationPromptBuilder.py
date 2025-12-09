from ssf.helpers import CorpusHelper
from ssf.Constants import *

class ProgenitorContextConsolidationPromptBuilder:
  def __init__(self, cur_utt_is_progenitor, corpus_utils: CorpusHelper):
    self.cur_utt_is_progenitor = cur_utt_is_progenitor
    self.corpus_utils = corpus_utils
    self._progenitor_title = None
    self._progenitor_summary = None

  def progenitor_title(self, progenitor_title):
    self._progenitor_title = progenitor_title
    return self
  
  def progenitor_summary(self, progenitor_summary):
    self._progenitor_summary = progenitor_summary
    return self

  def _have_context(self):
    return any(var is not None for var in self._get_contexts_vars())

  def _get_contexts_vars(self):
    return [self._progenitor_title, self._progenitor_summary]

  def _get_task_statement(self):
    return "Your task is to distill the provided context about the top-level post in a subreddit conversation into a succinct 1-sentence summary."
  
  def _get_context_types_overview(self):
    context_types_overview = "Context Types:"
    if self._progenitor_title is not None and self.corpus_utils.is_substantial_text(self._progenitor_title, min_char_length=10):
      context_types_overview += f"\n- Top-level Post Title: the title of the initial top-level post"
    if self._progenitor_summary is not None:
      context_types_overview += f"\n- Top-level Post Summary: a summary of the initial top-level post"
    return context_types_overview

  def _get_context_content(self):
    context_content = "Context:"
    if self._progenitor_title is not None and self.corpus_utils.is_substantial_text(self._progenitor_title, min_char_length=10):
      context_content += f"\n- Top-level Post Title: {self._progenitor_title}"
    if self._progenitor_summary is not None:
      context_content += f"\n- Top-level Post Summary: {self._progenitor_summary}"
    return context_content
  
  def _get_text_type(self):
    return "post" if self._progenitor_summary is None else "comment"
  
  def _build_output_instructions(self):
    return "Write a 1-sentence summary of the provided context. Output just the summary and no other text. Start your response with 'The first post...'."

  def build(self):
    if self.cur_utt_is_progenitor:
      return f"{SKIP_INFERENCE_PREFIX}The current text is the first post in the conversation."
    elif not self._have_context():
      return f"{SKIP_INFERENCE_PREFIX}Details about the first post in the conversation are unavailable."

    task_statement = self._get_task_statement()
    context_types_overview = self._get_context_types_overview()
    context_content = self._get_context_content()
    output_instructions = self._build_output_instructions()

    return f"{task_statement}\n\n{context_types_overview}\n\n{context_content}\n\n{output_instructions}"



