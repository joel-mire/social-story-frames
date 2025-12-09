from ssf.Constants import *
import json
import pandas as pd


def parse_labels(labels_str):
  if pd.isna(labels_str) or labels_str.strip() == '':
    return []
  return [label.strip() for label in labels_str.split(',') if label.strip()]

class InferenceClassificationPromptBuilder:
  """
  Enhanced prompt builder with semantic similarity-based few-shot example selection.
  """

  def __init__(self, 
                taxonomy, 
                text: str, 
                dim: str, 
                k: int = 0, 
                few_shot_df = None,
                similarity_index = None,
                diversity_strategy: str = 'other',
                diversity_weight: float = 0.5,
                exclude_ids: list = None):
      self.taxonomy = taxonomy
      self.text = text
      self._dim = dim
      self.k = k
      self.few_shot_df = few_shot_df
      self.similarity_index = similarity_index
      self.diversity_strategy = diversity_strategy
      self.diversity_weight = diversity_weight
      self.exclude_ids = exclude_ids or []


  def _build_task_statement(self):
    return f"Using the taxonomy and tips below, classify the following description of the {TAXONOMY_TASK_PLURAL_DICT[self._dim]} in a social media conversation."
  
  def _build_dim_overview(self):
    dim_overview = f"Taxonomy for {self._dim}:"
    for category, data in self.taxonomy.dim_data_dict[self._dim].items():
      dim_overview += f"\n- {category}: {data['definition']}"
    return dim_overview
  
  def _build_dim_tips(self):
    dim_tips = f"Classification Tips for {self._dim}:"
    dim_tips += f"\n{TAXONOMY_CLASSIFICATION_TIPS_DICT[self._dim]}"
    # for category, data in self.taxonomy.dim_data_dict[self._dim].items():
    #   dim_overview += f"\n- {category}: {data['definition']}"
    return dim_tips

  def _build_examples(self) -> str:
      """
      Build examples section
      """
      if self.k == 0:
          return ""
          
      # Use semantic similarity if index is available
      if self.similarity_index and self._dim in self.similarity_index.indices and self.diversity_strategy != 'random':
          sampled_df = self.similarity_index.get_similar_examples(
              query_text=self.text,
              dim=self._dim,
              k=self.k,
              diversity_strategy=self.diversity_strategy,
              diversity_weight=self.diversity_weight,
              exclude_ids=self.exclude_ids  # EFFICIENT LOOCV EXCLUSION
          )
      else:
          # Fallback to random sampling
          dim_df = self.few_shot_df[self.few_shot_df['dim'] == self._dim]
          
          # Apply exclusion filter if provided
          if self.exclude_ids:
              dim_df = dim_df[~dim_df['id'].isin(self.exclude_ids)]
          
          sampled_df = dim_df.sample(n=min(self.k, len(dim_df)), random_state=25)

      examples_overview = f"\n\nExamples:"
      for i, row in sampled_df.iterrows():
          examples_overview += f"\nInput: {row['response']}"
          # Labels are already lists, no need to parse
          response = {"response": row['labels']}
          examples_overview += f"\nOutput: {json.dumps(response)}\n"
          
      return examples_overview
  
  def _build_text_input(self):
    return f"Text to classify:\n{self.text}"
  
  def _build_output_instructions(self):
    response_json_dict = {
      "response": ["category_a", "..."]
    }
    response_json = json.dumps(response_json_dict, indent=2)
    output_instructions = (
        'Output Instructions:'
        f'\nRemember: {self._build_task_statement()}'
        '\nFill in the JSON list below with *ALL* of the categories that apply to the text. Many texts span multiple categories—please include every one that applies, not just the most obvious.'

        '\n\n**IMPORTANT RULES (READ CAREFULLY):**'
        '\n- DO NOT modify the JSON structure. Use valid JSON with double quotes only.'
        '\n- OUTPUT ONLY the completed template below — NO EXTRA TEXT, HEADINGS, OR COMMENTS.'

        f"\n{response_json}"
    )
    return output_instructions

  def build(self):
    task_statement = self._build_task_statement()
    dim_overview = self._build_dim_overview()
    dim_tips = self._build_dim_tips()
    examples = self._build_examples() if self.k > 0 else ""
    text_input = self._build_text_input()
    output_instructions = self._build_output_instructions()

    return f"{task_statement}\n\n{dim_overview}\n\n{dim_tips}{examples}\n\n{text_input}\n\n{output_instructions}"