# SSF-Sim: Story Sharing Framework Similarity

SSF-Sim is a similarity metric for comparing stories and communities based on the Story Sharing Framework taxonomy. It combines classification-based and generation-based approaches to compute comprehensive similarity scores.

## Features

- **Community-level similarity**: Compare communities based on aggregated story distributions
- **Story-pair similarity**: Compare individual stories directly
- **Dual approach**: Combines classification (sublabels) and generation (text embeddings)
- **Flexible data input**: Dict-based API with optional DataFrame helpers

## Installation

The package is part of the SSF framework:

```python
from ssf.ssf_sim import CommunitySsfSim, StorySsfSim
```

## Quick Start

### Community Similarity

```python
from ssf.ssf_sim import CommunitySsfSim
from ssf.ssf_sim.data_utils import (
    build_community_sublabel_counts,
    build_community_varvals_list
)
from ssf.Taxonomy import Taxonomy

# Initialize
taxonomy = Taxonomy(TAXONOMY_DIR)
sim_calculator = CommunitySsfSim(taxonomy, lambda_param=0.667)

# Prepare data from DataFrame
response_cols = {dim: f'prompt_default${dim}' for dim in taxonomy.get_dims()}

sublabel_counts = build_community_sublabel_counts(
    df=tc_analysis_df,
    taxonomy=taxonomy,
    groupby_col='subreddit',
    sublabels_to_ignore=['other']
)

varvals_list = build_community_varvals_list(
    df=tc_analysis_df,
    taxonomy=taxonomy,
    groupby_col='subreddit',
    response_cols=response_cols,
    response_col_suffixes=['_gen0']
)

# Compute similarity
sim_matrix = sim_calculator.compute_similarity(
    item_dim_sublabel_counts=sublabel_counts,
    item_dim_varvals_list=varvals_list
)

# Save results
sim_matrix.to_csv("ssf_sim.csv")
```

### Story-Pair Similarity

```python
from ssf.ssf_sim import StorySsfSim
from ssf.Taxonomy import Taxonomy

# Initialize
taxonomy = Taxonomy(TAXONOMY_DIR)
sim_calculator = StorySsfSim(taxonomy)

# Define stories (manually or from DataFrame)
story1_sublabels = {
    'overall_goal': ['inform', 'persuade'],
    'narrative_intent': ['share_experience'],
    'moral': ['self_transcendence'],
    # ... all 10 dimensions
}

story1_varvals = {
    'overall_goal': ['to inform the reader about climate change', 'to persuade them to act'],
    'narrative_intent': ['to share my experience with activism'],
    'moral': ['to promote caring for others'],
    # ... all 10 dimensions
}

story2_sublabels = {...}
story2_varvals = {...}

# Compute similarity
similarity = sim_calculator.compute_similarity(
    story1_dim_sublabels=story1_sublabels,
    story2_dim_sublabels=story2_sublabels,
    story1_dim_varvals=story1_varvals,
    story2_dim_varvals=story2_varvals
)

print(f"SSF-Sim: {similarity:.3f}")
```

## API Reference

### CommunitySsfSim

Compute similarity between communities (aggregated from multiple stories).

**Constructor:**
```python
CommunitySsfSim(
    taxonomy,
    sbert_model_name="sentence-transformers/all-MiniLM-L6-v2",
    lambda_param=0.667
)
```

**Methods:**
- `compute_similarity()`: Combined classification + generation similarity
- `compute_class_similarity()`: Classification-based only (Jensen-Shannon)
- `compute_gen_similarity()`: Generation-based only (cosine similarity)

### StorySsfSim

Compute similarity between individual story pairs.

**Constructor:**
```python
StorySsfSim(
    taxonomy,
    sbert_model_name="sentence-transformers/all-MiniLM-L6-v2",
    lambda_param=0.667
)
```

**Methods:**
- `compute_similarity()`: Combined classification + generation similarity
- `compute_class_similarity()`: Classification-based only
- `compute_gen_similarity()`: Generation-based only

### Data Utilities

Optional helper functions in `ssf.ssf_sim.data_utils`:

- `build_community_sublabel_counts()`: Extract sublabel counts from DataFrame
- `build_community_varvals_list()`: Extract variable values from DataFrame
- `extract_story_sublabels()`: Extract sublabels for a single story
- `extract_story_varvals()`: Extract variable values for a single story

## How SSF-Sim Works

### Classification-Based Similarity

1. For each dimension, compute Jensen-Shannon distance between sublabel distributions
2. Average JS distances across all dimensions
3. Convert to similarity: `sim = 1 - mean(JS distances)`

### Generation-Based Similarity

1. Embed generated variable values using SentenceTransformer
2. For communities: compute mean embeddings across stories
3. Compute cosine similarity between embeddings
4. Average across variables and dimensions

### Combined Similarity

For communities:
- Normalize both similarities to ranks (Borda count)
- Combine with weighted average: `λ * class_sim + (1-λ) * gen_sim`
- Default λ=0.667 favors classification

For story pairs:
- Simple weighted average (no rank normalization needed)

## Parameters

- **taxonomy**: Taxonomy instance with dimension metadata
- **sbert_model_name**: SentenceTransformer model for embeddings (default: `all-MiniLM-L6-v2`)
- **lambda_param**: Weight for classification vs generation in [0, 1] (default: 0.667)
  - Higher values favor classification-based similarity
  - Lower values favor generation-based similarity

## Examples

See the analysis notebooks for complete examples:
- `analysis/main.ipynb`: Community-level similarity computation
- Usage in various analysis workflows

## Notes

- The core SSF-Sim classes are data-agnostic (work with dicts)
- DataFrame helpers in `data_utils` are optional convenience functions
- Users can prepare data manually or adapt helpers for their data format
- Quality checks filter out invalid/unmodified responses during data preparation
