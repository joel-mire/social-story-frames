from collections import defaultdict
import pandas as pd
from ssf.prompt_builders.InferenceClassificationPromptBuilder import InferenceClassificationPromptBuilder
from ssf.Constants import NO_OP_MSG
from tqdm import tqdm

def get_taxonomy_dimensions_dict(taxonomy_df):
    taxonomy_dimensions_dict = defaultdict(lambda: defaultdict(dict))  # Correct inner default type to dict
    for _, row in taxonomy_df.iterrows():
        taxonomy_dimensions_dict[row['dimension']][row['category']] = {
            "definition": row['definition'],
            "example": row['example']
        }
    return taxonomy_dimensions_dict

def get_taxonomy_summaries_dict(taxonomy_summaries_df):
    taxonomy_summaries_dict = defaultdict(str)
    for _, row in taxonomy_summaries_df.iterrows():
        taxonomy_summaries_df[row['dimension']] = row['summary']
    return taxonomy_summaries_dict

def get_taxonomy_templates_dict(taxonomy_templates_df):
    return dict(zip(taxonomy_templates_df['dimension'], taxonomy_templates_df['template']))

def parse_labels(labels_str):
    """Parse comma-separated label string into list of labels."""
    if pd.isna(labels_str) or labels_str.strip() == '':
        return []
    return [label.strip() for label in labels_str.split(',') if label.strip()]

def generate_prompts(texts, 
                     dims, 
                     taxonomy, 
                     few_shot_data=None, 
                     similarity_index=None,
                     diversity_strategy=None,
                     dim_k_dict=None, 
                     exclude_ids=None,
                     verbose=False):
    """
    Generate classification prompts with few-shot examples.
    
    Args:
        texts: List of texts to classify
        dims: List of dimensions (same length as texts)
        taxonomy: Taxonomy object
        few_shot_data: DataFrame with few-shot examples
        similarity_index: SemanticSimilarityIndex for example selection
        diversity_strategy: Strategy for example selection
        dim_k_dict: Dictionary mapping dimensions to k values
        exclude_ids: List of IDs to exclude from few-shot selection (for LOOCV)
        verbose: Whether to show progress bar
        
    Returns:
        List of prompts
    """
    prompts = []
    
    # Conditionally wrap with tqdm
    iterator = zip(texts, dims)
    if verbose:
        iterator = tqdm(iterator, total=len(texts), desc="Generating prompts")
    
    for text, dim in iterator:
        # Handle empty/None texts
        if pd.isna(text) or text == '' or str(text).strip() == '':
            prompts.append(NO_OP_MSG)
            continue

        dim_k = dim_k_dict[dim] if dim_k_dict else 0
        prompt = InferenceClassificationPromptBuilder(
            taxonomy=taxonomy,
            text=text,
            dim=dim,
            k=dim_k,
            few_shot_df=few_shot_data,
            similarity_index=similarity_index,
            diversity_strategy=diversity_strategy,
            diversity_weight=0.5,
            exclude_ids=exclude_ids
        ).build()
        prompts.append(prompt)
    return prompts

def reshape_for_llamafactory(df, dims):
    """Reshape data for LlamaFactory format."""
    reshaped_data = []
    for _, row in df.iterrows():
        if row['split'] == 'train':
            instruction = row['prompt']
            output = row['labels']
            reshaped_data.append({
                'instruction': instruction,
                'input': '',
                'output': output,
                'dim': row['dim']
            })
    return pd.DataFrame(reshaped_data)