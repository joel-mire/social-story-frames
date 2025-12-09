import pandas as pd
from ssf.Constants import *
import os
from ssf.community_preprocessors.CommunityPreprocessor import CommunityPreprocessor
from ssf.utils import InferenceUtils
from ssf.generation_strategies import GenerationStrategy

class SubredditPreprocessor(CommunityPreprocessor):
  
  def __init__(self, 
               subreddits_path, 
               subreddit_desc_rules_path,
               dir, 
               generation_strategy: GenerationStrategy,
               force_rebuild=False):
    self.prompts_dir = f'{dir}/prompts'
    self.outputs_dir = f'{dir}/outputs'
    self.generation_strategy = generation_strategy

    subreddits_df = pd.read_csv(subreddits_path, dtype={'subreddit': str,
                                                        'is_niche': bool,
                                                        'is_unsafe': bool, 
                                                        'is_image_based': bool})                                         
    subreddit_desc_rules_df = pd.read_csv(subreddit_desc_rules_path) # cols: id,name,public_description,subscribers,rules,cleaned_rules,created_utc,topic_label,has_ai_rule_label,is_topical_question_and_answer_ca_label,is_learning_and_perspective_broadening_ca_label,is_social_support_ca_label,is_content_generation_ca_label,is_affiliation_with_an_entity_ca_label
    
    initial_subreddit_allowlist = self._build_subreddit_allowlist(subreddits_df)
    subreddit_desc_rules_dict = self._build_subreddit_desc_rules_dict(subreddit_desc_rules_df, 
                                                                      initial_subreddit_allowlist)
    self.subreddit_data_dict = self._build_subreddit_data_dict(subreddit_desc_rules_dict, force_rebuild=force_rebuild)
    self.finalized_subreddit_allowlist = set(self.subreddit_data_dict.keys())
    self.subreddit_topic_dict = self._build_subreddit_topic_dict(subreddit_desc_rules_df)

  def _build_subreddit_allowlist(self, subreddits_df):
    bool_cols = [col for col in subreddits_df.columns if col.startswith('is_')]
    allowlisted_subreddits_df = subreddits_df[~subreddits_df[bool_cols].any(axis=1)]
    return set(allowlisted_subreddits_df['subreddit'].tolist())
  
  def _build_subreddit_desc_rules_dict(self, subreddit_desc_rules_df, subreddit_allowlist):
    subreddit_desc_rules_dict = {}
    for _, row in subreddit_desc_rules_df.iterrows():
      subreddit = row['name'][2:] # skip 'r/'
      if subreddit in subreddit_allowlist:
        subreddit_desc_rules_dict[subreddit] = {
          "description": row['public_description'],
          "rules": row['cleaned_rules'],
        }
    return subreddit_desc_rules_dict
  
  def _build_subreddit_data_dict(self, subreddit_desc_rules_dict, force_rebuild):
    desc_prompts_path = f'{self.prompts_dir}/descriptions.jsonl'
    desc_outputs_path = f'{self.outputs_dir}/descriptions.jsonl'
    if not (os.path.exists(desc_prompts_path) and os.path.exists(desc_outputs_path)) or force_rebuild:
      desc_prompts = self._get_description_prompts(subreddit_desc_rules_dict)
      InferenceUtils.save_prompts_as_jsonl(desc_prompts, desc_prompts_path)
      self.generation_strategy.generate(desc_prompts_path, desc_outputs_path)
    desc_outputs = InferenceUtils.read_jsonl(desc_outputs_path)
    descriptions = [output['output'] for output in desc_outputs]

    values_prompts_path = f'{self.prompts_dir}/values.jsonl'
    values_outputs_path = f'{self.outputs_dir}/values.jsonl'
    if not (os.path.exists(values_prompts_path) and os.path.exists(values_outputs_path)) or force_rebuild:
      values_prompts = self._get_values_prompts(subreddit_desc_rules_dict)
      InferenceUtils.save_prompts_as_jsonl(values_prompts, values_prompts_path)
      self.generation_strategy.generate(values_prompts_path, values_outputs_path)
    values_outputs = InferenceUtils.read_jsonl(values_outputs_path)
    values_list = [output['output'] for output in values_outputs]

    subreddit_data_dict = {}
    for subreddit, description, values in zip(subreddit_desc_rules_dict.keys(), descriptions, values_list):
      subreddit_data_dict[subreddit] = {
        "description": description,
        "values": values
      }
    
    return subreddit_data_dict

  def _build_subreddit_topic_dict(self, subreddit_desc_rules_df):
    subreddit_topic_dict = {}
    for _, row in subreddit_desc_rules_df.iterrows():
      subreddit = row['name'][2:] # skip 'r/'
      if subreddit in self.finalized_subreddit_allowlist:
        subreddit_topic_dict[subreddit] = row['topic_label']
    return subreddit_topic_dict

  def _get_description_prompts(self, subreddit_desc_rules_dict):
    prompts = []
    for subreddit, data in subreddit_desc_rules_dict.items():
      desc = data['description']
      prompts.append(f"Summarize the following description of the r/{subreddit} subreddit in 1 sentence. Do not hallucinate and do not say the text is too short to summarize. Output the summary and no other text.\n\n{desc}")
    return prompts

  def _get_values_prompts(self, subreddit_desc_rules_dict):
    prompts = []
    for subreddit, data in subreddit_desc_rules_dict.items():
      desc = data['description']
      rules = data['rules']
      prompts.append(f"Summarize key values or norms of the r/{subreddit} subreddit that are either explicitly stated or strongly evidenced by the following description and rules for the subreddit. Do not hallucinate. Output a 1 sentence summary and no other text.\n\nDescription: {desc}\n\nRules: {rules}")
    return prompts
  
  def get_community_allowlist(self):
    return self.finalized_subreddit_allowlist

  def get_community_data_dict(self):
    return self.subreddit_data_dict

  def get_community_topic_dict(self):
    return self.subreddit_topic_dict
  

  