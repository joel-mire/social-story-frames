from ssf.Constants import *
import json

class InferenceGenerationPromptBuilder:

  def __init__(self, taxonomy, dim, text, single_output):
    self._taxonomy = taxonomy
    self._dim = dim
    self._text = text
    self.single_output = single_output

    self._community_name = None
    self._community_description = None
    self._community_values = None
    self._progenitor_summary = None
    self._conversation_summary = None

  def community_name(self, community_name):
    self._community_name = community_name
    return self

  def community_description(self, community_description):
    self._community_description = community_description
    return self

  def community_values(self, community_values):
    self._community_values = community_values
    return self
  
  def progenitor_summary(self, progenitor_summary):
    self._progenitor_summary = progenitor_summary
    return self
  
  def conversation_summary(self, conversation_summary):
    self._conversation_summary = conversation_summary
    return self
  
  def text(self, text):
    self._text = text
    return self
  
  def _build_task_statement(self):
    if self.single_output:
      return f"Your task is to use commonsense to generate one contextually plausible description of the {TAXONOMY_TASK_SINGULAR_DICT[self._dim]} in a social media conversation."
    else:
      return f"Your task is to use commonsense to generate contextually plausible description(s) of the {TAXONOMY_TASK_PLURAL_DICT[self._dim]} in a social media conversation."
  
  def _build_dim_overview(self):
    return f"General (non-exhaustive) information to help scaffold your thinking about {self._dim} in the context of social media storytelling:\n{self._taxonomy.dim_summaries_dict[self._dim]}"

  def _build_context_types_overview(self):
    context_types_overview = "The following conversational context types are available:"
    if self._community_name:
      context_types_overview += f"\n- Subreddit Name: the Reddit community where the conversation takes place"
    if self._community_description:
      context_types_overview += f"\n- Subreddit Description: a brief overview of the subreddit topic"
    if self._community_values:
      context_types_overview += f"\n- Subreddit Values: a high-level summary of key values, norms, or rules in the subreddit"
    if self._progenitor_summary:
      context_types_overview += f"\n- Top-level Post Summary: a summary of the first, top-level post in the conversation thread"
    if self._conversation_summary:
      context_types_overview += f"\n- Conversation Summary: a summary of the prior conversation leading up to the current text"
    context_types_overview += f"\n- Current Text: the current text to analyze. The text necessarily contains storytelling (even if the story is short or banal)."
    return context_types_overview
  
  def _build_conversation_context(self):
    conversational_context = "Conversational Context:"
    if self._community_name:
      conversational_context += f"\n- Subreddit Name: {self._community_name}"
    if self._community_description:
      conversational_context += f"\n- Subreddit Description: {self._community_description}"
    if self._community_values:
      conversational_context += f"\n- Subreddit Values: {self._community_values}"
    if self._progenitor_summary:
      conversational_context += f"\n- Top-level Post Summary: {self._progenitor_summary}"
    if self._conversation_summary:
      conversational_context += f"\n- Conversation Summary: {self._conversation_summary}"
    return conversational_context
  
  def _build_text_input(self):
    return f"Current Text:\n{self._text}"
  
  def _build_output_instructions(self):
    if self.single_output:
      return self._build_single_output_instructions()
    else:
      return self._build_multi_output_instructions()

  def _build_single_output_instructions(self):
    dim_template = self._taxonomy.dim_templates_dict[self._dim]
    response_json_dict = {
      "response": dim_template
    }
    response_json = json.dumps(response_json_dict, indent=2)
    output_instructions = (
        'Output Instructions:'
        f'\nRemember: {self._build_task_statement()}'
        '\nFill in the JSON list below with **exactly 1 contextually plausible and likely response** to the current text in conversational context, as if you were a member of the subreddit who wants to construct a coherent understanding of the story and understand why the events and states in the story are mentioned.'
        f'\nYou may use the provided info about {self._dim} as background but do **not** force your response to fit it. You must not copy directly from the provided info if you can answer more precisely in your own words.'
        '\n\n**IMPORTANT RULES (READ CAREFULLY):**'
        '\n- ONLY edit inside double-braced placeholders like `{{...}}`. DO NOT MODIFY ANY TEXT OUTSIDE `{{}}`.'
        '\n- DO NOT change or correct the template’s wording, punctuation, or singular/plural mismatches. FOLLOW THE TEMPLATE EXACTLY.'
        '\n- DO NOT modify the JSON structure. Use valid JSON with double quotes only.'
        '\n- OUTPUT ONLY the completed template below — NO EXTRA TEXT, HEADINGS, OR COMMENTS.'
        '\n- IF YOU BREAK THESE RULES, THE OUTPUT WILL BE UNUSABLE.'

        f"\n{response_json}"
    )
    return output_instructions

  def _build_multi_output_instructions(self):
    dim_template = self._taxonomy.dim_templates_dict[self._dim]
    response_json_dict = {
        "responses": [
            dim_template,
            dim_template,
            dim_template
        ]
    }
    response_json = json.dumps(response_json_dict, indent=2)
    output_instructions = (
        'Output Instructions:'
        f'\nRemember: {self._build_task_statement()}'
        '\nFill in the JSON list below with **up to 3 contextually plausible and likely responses** to the current text in conversational context, as if you were a member of the subreddit who wants to construct a coherent understanding of the story and understand why the events and states in the story are mentioned.'
        '**All responses must be in a single JSON list** (e.g., `["A", "B", "C"]`), **not separate lists** (❌ `["A"] ["B"] ["C"]`). The responses must be unique/independent. If you cannot think of three unique responses that are highly plausible/likely, you may include fewer responses in the output list.'
        f'\nYou may use the provided info about {self._dim} as background but do **not** force your response(s) to fit it. You must not copy directly from the provided info if you can answer more precisely in your own words.'
        '\n\n**IMPORTANT RULES (READ CAREFULLY):**'
        '\n- ONLY edit inside double-braced placeholders like `{{...}}`. DO NOT MODIFY ANY TEXT OUTSIDE `{{}}`.'
        '\n- DO NOT change or correct the template’s wording, punctuation, or singular/plural mismatches. FOLLOW THE TEMPLATE EXACTLY.'
        '\n- DO NOT modify the JSON structure. Use valid JSON with double quotes only.'
        '\n- OUTPUT ONLY the completed template below — NO EXTRA TEXT, HEADINGS, OR COMMENTS.'
        '\n- IF YOU BREAK THESE RULES, THE OUTPUT WILL BE UNUSABLE.'
        f"\n{response_json}"
    )
    return output_instructions

  def build(self):
    task_statement = self._build_task_statement()
    dim_overview = self._build_dim_overview()
    context_types_overview = self._build_context_types_overview()
    conversation_context = self._build_conversation_context()
    text_input = self._build_text_input()
    output_instructions = self._build_output_instructions()

    prompt = f"{task_statement}\n\n{dim_overview}\n\n{context_types_overview}\n\n{conversation_context}\n\n{text_input}\n\n{output_instructions}"
    return prompt
