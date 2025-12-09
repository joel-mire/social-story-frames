from typing import Dict, List, Counter
from collections import Counter as CounterCls
from itertools import combinations_with_replacement
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from ssf.helpers import map_moral_values

from ssf.ssf_sim.core import (
    compute_js_distance,
    compute_cosine_similarity,
    weighted_borda_fusion
)


class CommunitySsfSim:
    """
    Compute SSF-Sim between communities using aggregated distributions.

    SSF-Sim combines two approaches:
    1. Classification-based: Jensen-Shannon distance on sublabel distributions
    2. Generation-based: Cosine similarity of mean embeddings of generated text

    The final score uses weighted Borda rank fusion to combine both approaches.
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
        item_dim_sublabel_counts: Dict[str, Dict[str, Counter]],
        item_dim_varvals_list: Dict[str, Dict[str, List[List[str]]]]
    ) -> pd.DataFrame:
        class_sim = self.compute_class_similarity(item_dim_sublabel_counts)
        gen_sim = self.compute_gen_similarity(item_dim_varvals_list)

        # Weighted Borda rank fusion
        return weighted_borda_fusion(class_sim, gen_sim, self.lambda_param)

    def compute_class_similarity(
        self,
        item_dim_sublabel_counts: Dict[str, Dict[str, Counter]]
    ) -> pd.DataFrame:
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

        items = sorted(item_dim_sublabel_counts.keys())
        sim_df = pd.DataFrame(index=items, columns=items, dtype=float)

        for item1, item2 in combinations_with_replacement(items, 2):
            js_distances = []

            for dim in self.dims:
                sublabel_support = get_sublabel_support(dim, allow_other=False)

                count_dict_1 = item_dim_sublabel_counts[item1][dim]
                count_dict_2 = item_dim_sublabel_counts[item2][dim]

                js_distance = compute_js_distance(
                    sublabel_support=sublabel_support,
                    count_dict_1=count_dict_1,
                    count_dict_2=count_dict_2
                )
                js_distances.append(js_distance)

            # Convert mean JS distance to similarity
            similarity = 1 - np.mean(js_distances)
            sim_df.loc[item1, item2] = similarity

        return sim_df

    def compute_gen_similarity(
        self,
        item_dim_varvals_list: Dict[str, Dict[str, List[List[str]]]]
    ) -> pd.DataFrame:
        # Precompute mean embeddings for all items
        item_dim_mean_embeddings = {}

        for item, dim_varvals_list in item_dim_varvals_list.items():
            dim_mean_embeddings = self._compute_dim_mean_embeddings(dim_varvals_list)
            item_dim_mean_embeddings[item] = dim_mean_embeddings

        # Compute pairwise similarities
        items = sorted(item_dim_varvals_list.keys())
        sim_df = pd.DataFrame(index=items, columns=items, dtype=float)

        for item1, item2 in combinations_with_replacement(items, 2):
            item1_dim_embeddings = item_dim_mean_embeddings[item1]
            item2_dim_embeddings = item_dim_mean_embeddings[item2]

            dim_similarities = []

            for dim in self.dims:
                item1_embeddings = item1_dim_embeddings[dim]
                item2_embeddings = item2_dim_embeddings[dim]

                # Compute cosine similarity for each variable
                var_similarities = []
                for emb1, emb2 in zip(item1_embeddings, item2_embeddings):
                    cos_sim = compute_cosine_similarity(emb1, emb2)
                    var_similarities.append(cos_sim)

                # Average across variables for this dimension
                dim_similarities.append(np.mean(var_similarities))

            # Average across all dimensions
            overall_similarity = np.mean(dim_similarities)
            sim_df.loc[item1, item2] = overall_similarity

        return sim_df

    def _compute_dim_mean_embeddings(
        self,
        dim_varvals_list: Dict[str, List[List[str]]]
    ) -> Dict[str, np.ndarray]:
        dim_mean_embeddings = {}

        for dim, varvals_list in dim_varvals_list.items():
            # Skip excluded dimensions
            if dim not in self.dims:
                continue

            if not varvals_list:
                # No stories for this dimension, create zero embeddings
                num_vars = len(self.taxonomy.get_dim_vars_dict()[dim])
                embedding_dim = self.st_encoder.get_sentence_embedding_dimension()
                dim_mean_embeddings[dim] = np.zeros((num_vars, embedding_dim))
                continue

            var_mean_embeddings = []
            num_vars = len(varvals_list[0])

            for i in range(num_vars):
                # Collect all values for variable i across all stories
                texts = [varvals[i] for varvals in varvals_list]

                # Embed (unnormalized)
                unnormalized_embeddings = self.st_encoder.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )

                # Mean pool
                mean_embedding = np.mean(unnormalized_embeddings, axis=0)

                # Normalize
                normed_embedding = mean_embedding / np.linalg.norm(mean_embedding)

                var_mean_embeddings.append(normed_embedding)

            dim_mean_embeddings[dim] = np.array(var_mean_embeddings)

        return dim_mean_embeddings
