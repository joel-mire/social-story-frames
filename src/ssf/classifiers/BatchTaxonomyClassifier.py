"""
Batch API-based taxonomy classifier using composition with BatchProcessor.
"""
import os
import pandas as pd
from ssf.classifiers.TaxonomyClassifier import TaxonomyClassifier
from ssf.classifiers.BatchProcessor import BatchProcessor
from ssf.utils import TaxonomyUtils
from ssf.Constants import *

class BatchTaxonomyClassifier(TaxonomyClassifier):
    """
    OpenAI Batch API-based taxonomy classifier.
    Supports both LOOCV and non-LOOCV few-shot sampling strategies.
    """
    
    def __init__(self, 
                 taxonomy, 
                 few_shot_data, 
                 model_name, 
                 ssf_gen_base_model,
                 dim_k_dict=None, 
                 diversity_strategy=None, 
                 show_progress=False, 
                 completion_window="24h", 
                 similarity_index=None):
        super().__init__(taxonomy)
        self.few_shot_data = few_shot_data
        self.model_name = model_name
        self.ssf_gen_base_model = ssf_gen_base_model
        self.dim_k_dict = dim_k_dict if dim_k_dict is not None else {}
        self.diversity_strategy = diversity_strategy
        self.similarity_index = similarity_index
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            model_name=model_name,
            completion_window=completion_window,
            show_progress=show_progress
        )
        
        # Filter few-shot data to target dimensions if 'dim' column exists
        target_dims = taxonomy.get_dims()
        if 'dim' in self.few_shot_data.columns:
            self.few_shot_data = self.few_shot_data[self.few_shot_data['dim'].isin(target_dims)]

    def _generate_prompts(self, texts, dims, instance_ids):
        """Generate prompts for standard stages using consistent few-shot examples."""
        prompts = []
        custom_ids = []
        
        for text, dim, instance_id in zip(texts, dims, instance_ids):
            # Skip empty texts
            if pd.isna(text) or text == '' or str(text).strip() == '':
                continue
                
            prompts_for_instance = TaxonomyUtils.generate_prompts(
                [text], [dim], self.taxonomy, self.few_shot_data,
                self.similarity_index, self.diversity_strategy, self.dim_k_dict
            )
            
            if prompts_for_instance[0] != NO_OP_MSG:
                prompts.append(prompts_for_instance[0])
                custom_ids.append(f"{instance_id}_{dim}")
            
        return prompts, custom_ids

    def classify_texts(self, texts, dims, instance_ids, output_path, force_redo=False):
        """
        Classify texts using OpenAI Batch API.
        Delegates batch processing to BatchProcessor, focuses on prompt generation.
        """
        prompts, custom_ids = self._generate_prompts(texts, dims, instance_ids)
        
        # Use batch processor for submission and monitoring
        return self.batch_processor.process_batch(prompts, custom_ids, output_path, force_redo)
    
    def classify_generations(self, stories_df, output_dir, force_redo=False):
        """
        Classify ML Stories generations using batch API with few-shot examples.
        Uses a single batch for all (dimension, generation) combinations.
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect ALL texts, dims, and custom_ids for batch submission
        all_texts = []
        all_dims = []
        all_custom_ids = []
        
        # Process each dimension and generation combination
        model_prefix = self.model_name.replace('-', '_')
        
        dims = self.taxonomy.get_dims()
        print(f"[BatchTaxonomyClassifier] Generating prompts for {len(dims)} dimensions...")
        
        for dim_idx, dim in enumerate(dims):
            print(f"[BatchTaxonomyClassifier] Processing dimension {dim_idx+1}/{len(dims)}: {dim}")
                
            for gen_suffix in ['gen0']:
                # Construct the actual column name in the DataFrame
                col_name = f"{self.ssf_gen_base_model.replace('-', '_')}_{dim}_{gen_suffix}"
                
                if col_name not in stories_df.columns:
                    print(f"[BatchTaxonomyClassifier] Column {col_name} not found, skipping...")
                    continue
                
                print(f"[BatchTaxonomyClassifier] Processing {col_name} ({len(stories_df)} stories)...")
                
                for _, row in stories_df.iterrows():
                    instance_id = row['id']
                    text = row[col_name]
                    
                    # Skip empty generations (will be handled by utility function)
                    all_texts.append(text)
                    all_dims.append(dim)
                    all_custom_ids.append(f"{instance_id}_{dim}_{gen_suffix}")
        
        # Use existing utility function to generate prompts
        prompts = TaxonomyUtils.generate_prompts(
            all_texts, 
            all_dims, 
            self.taxonomy, 
            self.few_shot_data,
            self.similarity_index, 
            self.diversity_strategy, 
            self.dim_k_dict,
            verbose=True
        )
        
        # Filter out NO_OP_MSG prompts and corresponding custom_ids
        filtered_prompts = []
        filtered_custom_ids = []
        for prompt, custom_id in zip(prompts, all_custom_ids):
            if prompt != NO_OP_MSG:
                filtered_prompts.append(prompt)
                filtered_custom_ids.append(custom_id)
        
        # Submit single batch for all prompts
        unified_output_path = f"{output_dir}/all_generations_outputs.jsonl"
        self.batch_processor.process_batch(
            filtered_prompts, 
            filtered_custom_ids, 
            unified_output_path, 
            force_redo
        )