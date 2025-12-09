"""
GPT-based taxonomy classifier using composition pattern.
"""
from ssf.classifiers.TaxonomyClassifier import TaxonomyClassifier
from ssf.utils import TaxonomyUtils, InferenceUtils
import os
import pandas as pd
from ssf.generation_strategies.OpenaiGenerationStrategy import OpenaiGenerationStrategy
from ssf.generation_strategies.configs import ModelConfig, GenerationConfig
from ssf.Constants import *
import os

class GptTaxonomyClassifier(TaxonomyClassifier):
    """
    GPT-based taxonomy classifier.
    Pure classification functionality - no training, no evaluation.
    """
    
    def __init__(self, 
                 taxonomy, 
                 model_name,
                 few_shot_data=None, 
                 dim_k_dict=None, 
                 diversity_strategy=None, 
                 show_progress=False, 
                #  model_name=None,
                 similarity_index=None):
        super().__init__(taxonomy)
        self.few_shot_data = few_shot_data
        self.dim_k_dict = dim_k_dict if dim_k_dict is not None else {}
        self.diversity_strategy = diversity_strategy
        self.show_progress = show_progress
        self.similarity_index = similarity_index
        
        # Setup generation strategy
        generation_config = GenerationConfig(max_new_tokens=200)
        model_config = ModelConfig(model_name=model_name)
        self.generation_strategy = OpenaiGenerationStrategy(model_config, generation_config, show_progress=show_progress)

        # Filter few-shot data to target dimensions if 'dim' column exists
        target_dims = taxonomy.get_dims()
        if 'dim' in self.few_shot_data.columns:
            self.few_shot_data = self.few_shot_data[self.few_shot_data['dim'].isin(target_dims)]

    def classify_texts(self, texts, dims, instance_ids, output_path, force_redo=False):
        """
        Classify a list of texts for given dimensions.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use LOOCV-aware prompt generation if instance_ids provided
        if instance_ids is not None:
            prompts = []
            for text, dim, instance_id in zip(texts, dims, instance_ids):
                # Generate one prompt at a time to use different exclude_ids
                prompt_batch = TaxonomyUtils.generate_prompts(
                    [text], [dim], self.taxonomy, self.few_shot_data,
                    self.similarity_index, self.diversity_strategy, self.dim_k_dict,
                    exclude_ids=[instance_id]  # LOOCV: Exclude current instance
                )
                prompts.extend(prompt_batch)
        else:
            # No LOOCV needed
            prompts = TaxonomyUtils.generate_prompts(
                texts, dims, self.taxonomy, self.few_shot_data, 
                self.similarity_index, self.diversity_strategy, self.dim_k_dict
            )
    
        # Save prompts
        prompts_path = output_path.replace('outputs.jsonl', 'prompts.jsonl')
        print("PROMPTS PATH", prompts_path)
        InferenceUtils.save_prompts_as_jsonl(prompts, prompts_path)
        
        # Generate classifications
        if force_redo or not os.path.exists(output_path):
            if self.show_progress:
                print(f"Generating outputs for {len(prompts)} prompts...")
            self.generation_strategy.generate(prompts_path, output_path)
        elif self.show_progress:
            print(f"Using cached results from {output_path}")
            
        return output_path
    
    def classify_generations(self, ml_stories_df, output_dir, force_redo=False):
        """
        Classify multiple generations per instance across dimensions.
        Specialized method for Stage 3 - matches batch implementation logic.
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each generation column and dimension
        for col_name in ['gen0', 'gen1', 'gen2']:
            if col_name not in ml_stories_df.columns:
                continue
            
            for dim in self.taxonomy.get_dims():
                # Get texts for this generation and dimension
                texts = []
                for _, row in ml_stories_df.iterrows():
                    text = row[col_name]
                    # Skip empty generations
                    if pd.isna(text) or text == '' or str(text).strip() == '':
                        texts.append('')  # Keep index alignment
                    else:
                        texts.append(text)
                
                # Create dimension list
                dims = [dim] * len(texts)
                
                # Classify texts
                outputs_path = f'{output_dir}/{col_name}_{dim}_outputs.jsonl'
                self.classify_texts(texts, dims, outputs_path, force_redo)