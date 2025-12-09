"""
Story-pair SSF-Sim computation.

This module provides the StorySsfSim class for computing similarity between
individual story pairs.
"""

from typing import Dict, List
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from ssf.helpers import map_moral_values

from ssf.ssf_sim.core import (
    compute_js_distance,
    compute_cosine_similarity
)


class StorySsfSim:
    """
    Compute SSF-Sim between two individual stories.

    SSF-Sim combines two approaches:
    1. Classification-based: Jensen-Shannon distance on sublabel distributions
    2. Generation-based: Cosine similarity of embeddings of generated text

    For story pairs, we use simple weighted averaging rather than rank fusion.
    """

    def __init__(
        self,
        taxonomy,
        sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        lambda_param: float = 0.667,
        exclude_dims: List[str] = []
    ):
        self.taxonomy = taxonomy
        self.sbert_model_name = sbert_model_name
        self.lambda_param = lambda_param
        self.exclude_dims = exclude_dims
        self.dims = [dim for dim in self.taxonomy.get_dims() if dim not in self.exclude_dims]
        self.st_encoder = SentenceTransformer(sbert_model_name)

    def compute_similarity(
        self,
        story1_dim_sublabels: Dict[str, List[str]],
        story2_dim_sublabels: Dict[str, List[str]],
        story1_dim_varvals: Dict[str, List[str]],
        story2_dim_varvals: Dict[str, List[str]]
    ) -> float:
        class_sim = self.compute_class_similarity(
            story1_dim_sublabels, story2_dim_sublabels
        )
        gen_sim = self.compute_gen_similarity(
            story1_dim_varvals, story2_dim_varvals
        )

        # Weighted average (not rank fusion for single pair)
        return self.lambda_param * class_sim + (1 - self.lambda_param) * gen_sim

    def compute_class_similarity(
        self,
        story1_dim_sublabels: Dict[str, List[str]],
        story2_dim_sublabels: Dict[str, List[str]]
    ) -> float:
        # Helper to get valid sublabels for a dimension
        def get_sublabel_support(dim, allow_other=False):
            sublabels = [
                s for s in self.taxonomy.dim_data_dict[dim].keys()
                if s not in self.taxonomy.get_excluded_categories(dim)
            ]
            sublabels = list(set(sublabels))
            if not allow_other and 'other' in sublabels:
                sublabels.remove('other')
            if dim == 'moral':
                sublabels = map_moral_values(sublabels)
            return sublabels

        js_distances = []

        for dim in self.dims:
            sublabel_support = get_sublabel_support(dim, allow_other=False)

            # Convert sublabel lists to Counters
            count_dict_1 = Counter(story1_dim_sublabels.get(dim, []))
            count_dict_2 = Counter(story2_dim_sublabels.get(dim, []))

            js_distance = compute_js_distance(
                sublabel_support=sublabel_support,
                count_dict_1=count_dict_1,
                count_dict_2=count_dict_2
            )
            js_distances.append(js_distance)

        # Convert mean JS distance to similarity
        return 1 - np.mean(js_distances)

    def compute_gen_similarity(
        self,
        story1_dim_varvals: Dict[str, List[str]],
        story2_dim_varvals: Dict[str, List[str]]
    ) -> float:
        dim_similarities = []

        for dim in self.dims:
            varvals1 = story1_dim_varvals.get(dim, [])
            varvals2 = story2_dim_varvals.get(dim, [])

            if not varvals1 or not varvals2:
                # If either story has no variable values for this dimension, skip
                continue

            # Embed variable values
            embeddings1 = self.st_encoder.encode(
                varvals1,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalize for cosine similarity
                show_progress_bar=False
            )

            embeddings2 = self.st_encoder.encode(
                varvals2,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            # Compute cosine similarity for each variable pair
            var_similarities = []
            for emb1, emb2 in zip(embeddings1, embeddings2):
                cos_sim = compute_cosine_similarity(emb1, emb2)
                var_similarities.append(cos_sim)

            # Average across variables for this dimension
            dim_similarities.append(np.mean(var_similarities))

        if not dim_similarities:
            # No valid dimensions to compare
            return 0.0

        # Average across all dimensions
        return np.mean(dim_similarities)
