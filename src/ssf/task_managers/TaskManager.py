from abc import ABC, abstractmethod
import os
from ssf.utils import InferenceUtils
from ssf.Constants import *
from ssf.Configs import Context

class TaskManager(ABC):
    """Base class for synchronous task managers"""
    
    def __init__(self, 
                 taxonomy, 
                 generation_strategy, 
                 out_dir, 
                 force_redo, 
                 disambiguator,
                 context_config: Context):
        
        self.taxonomy = taxonomy
        self.generation_strategy = generation_strategy
        self.out_dir = out_dir
        self.force_redo = force_redo
        self.disambiguator = disambiguator
        self.context = context_config

    def run_task(self, stories_df):
        """Execute task: generate prompts and run inference"""
        prompt_dir = f'{self.out_dir}/prompts/{self.disambiguator}'
        output_dir = f'{self.out_dir}/outputs/{self.disambiguator}'
        os.makedirs(prompt_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        for dim in self.taxonomy.get_dims():
            prompts_path = f'{prompt_dir}/{dim}.jsonl'
            outputs_path = f'{output_dir}/{dim}.jsonl'
            
            if not (os.path.exists(prompts_path) and os.path.exists(outputs_path)) or self.force_redo:
                prompts = self.get_prompts(stories_df, dim)
                InferenceUtils.save_prompts_as_jsonl(prompts, prompts_path)
                self.generation_strategy.generate(prompts_path, outputs_path)

    def get_prompts(self, stories_df, dim, include_id=True):
        """Generate prompts for all rows with associated IDs

        Args:
            stories_df: DataFrame containing story data
            dim: Dimension to generate prompts for
            include_id: If True, return list of dicts with 'id' and 'prompt' keys.
                       If False, return list of prompt strings only.
        """
        prompts = []
        for _, row in stories_df.iterrows():
            prompt = self.get_prompt(dim, row)
            if include_id:
                story_id = row['id']
                prompts.append({'id': story_id, 'prompt': prompt})
            else:
                prompts.append(prompt)
        return prompts

    @abstractmethod
    def get_prompt(self, dim, row):
        """Generate prompt for a single row"""
        pass
    
    @abstractmethod
    def add_results(self, stories_df, dim_outputs_dict):
        """Add results to DataFrame"""
        pass

    def postprocess_results(self, stories_df):
        """Load results from files and merge into DataFrame"""
        dim_outputs_dict = {}
        for dim in self.taxonomy.get_dims():
            dim_outputs_path = f'{self.out_dir}/outputs/{self.disambiguator}/{dim}.jsonl'
            with open(dim_outputs_path, 'r') as f:
                dim_outputs_dict[dim] = [InferenceUtils.parse_json(line) for line in f.readlines()]
        return self.add_results(stories_df, dim_outputs_dict)