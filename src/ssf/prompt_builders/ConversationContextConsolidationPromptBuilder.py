from ssf.Constants import *

class ConversationContextConsolidationPromptBuilder:

  def __init__(self):
    self._ancestral_summary = None
    self._preceding_peers_summary = None
  
  def ancestral_summary(self, ancestral_summary):
    self._ancestral_summary = ancestral_summary
    return self
  
  def preceding_peers_summary(self, preceding_peers_summary):
    self._preceding_peers_summary = preceding_peers_summary
    return self

  def _have_context(self):
    return any(var is not None for var in self._get_contexts_vars())

  def _get_contexts_vars(self):
    return [self._ancestral_summary, self._preceding_peers_summary]

  def _get_task_statement(self):
    return "Your task is to distill the provided context about the conversational context into a 1-2 sentence summary."
  
  def _get_context_types_overview(self):
    context_types_overview = "Conversational Context Types:"
    if self._ancestral_summary is not None:
      context_types_overview += f"\n- Ancestors Summary: a summary of the chain of texts formed by a parent-child relationship leading up to the current text"
    if self._preceding_peers_summary is not None:
      context_types_overview += f"\n- Preceding Peers Summary: a summary of the chronologically-ordered comments preceding the current text under the same parent"
    return context_types_overview

  def _get_context_content(self):
    context_content = "Conversational Context:"
    if self._ancestral_summary is not None:
      context_content += f"\n- Ancestors Summary: {self._ancestral_summary}"
    if self._preceding_peers_summary is not None:
      context_content += f"\n- Preceding Peers Summary: {self._preceding_peers_summary}"
    return context_content
  
  def _get_text_type(self):
    return "post" if self._progenitor_summary is None else "comment"

  def _build_output_instructions(self):
    return "Summarize the provided conversational context in 1-3 sentences. Output just the summary and no other text. Start your response with 'The conversation so far...'."

  def build(self):
    if not self._have_context():
      return f"{SKIP_INFERENCE_PREFIX}There is no prior conversational context."
    
    task_statement = self._get_task_statement()
    context_types_overview = self._get_context_types_overview()
    context_content = self._get_context_content()
    output_instructions = self._build_output_instructions()

    return f"{task_statement}\n\n{context_types_overview}\n\n{context_content}\n\n{output_instructions}"