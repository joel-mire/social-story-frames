import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import pandas as pd
import torch

class SemanticSimilarityIndex:
    """
    FAISS-based semantic similarity index for efficient few-shot example retrieval.
    Initialize once and reuse across multiple prompt building calls.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', random_seed: int = 42):
        # Ensure deterministic behavior for sentence transformer
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self.encoder = SentenceTransformer(model_name)
        
        # Create dedicated RandomState for deterministic randomness
        self.rng = np.random.RandomState(random_seed)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.indices: Dict[str, faiss.IndexFlatIP] = {}  # Per-dimension indices
        self.texts: Dict[str, List[str]] = {}  # Store original texts per dimension
        self.metadata: Dict[str, pd.DataFrame] = {}  # Store full row data per dimension
        self.embeddings: Dict[str, np.ndarray] = {}  # Store embeddings per dimension
        
    def build_index(self, few_shot_df: pd.DataFrame, text_col: str):
        """
        Build FAISS indices for each dimension in the few-shot dataset.
        
        Args:
            few_shot_df: DataFrame with 'dim', text_col
        """
        dims = few_shot_df['dim'].unique()
        for i, dim in enumerate(dims):
            dim_df = few_shot_df[few_shot_df['dim'] == dim].copy()
            # Extract texts for embedding
            texts = dim_df[text_col].tolist()
            # Generate embeddings
            embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings.astype('float32')
            # Normalize for cosine similarity (using inner product on normalized vectors)
            faiss.normalize_L2(embeddings)
            # Create and populate FAISS index
            index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
            index.add(embeddings)
            # Store everything
            self.indices[dim] = index
            self.texts[dim] = texts
            self.metadata[dim] = dim_df.reset_index(drop=True)
            self.embeddings[dim] = embeddings
            
    def get_similar_examples(self, 
                           query_text: str, 
                           dim: str, 
                           k: int,
                           diversity_strategy: str = 'mmr',
                           diversity_weight: float = 0.5,
                           exclude_ids: list = None) -> pd.DataFrame:
        """
        Retrieve k diverse yet similar examples using various diversity strategies.
        
        Args:
            query_text: Text to find similar examples for
            dim: Dimension to search within
            k: Number of examples to retrieve
            diversity_strategy: 'mmr', 'pure_similarity', or 'random_pure_similarity_mmr'
            diversity_weight: Balance between similarity and diversity (0=pure similarity, 1=pure diversity)
            exclude_ids: List of IDs to exclude from results (for LOOCV)
            
        Returns:
            DataFrame with k diverse examples
        """
        k = min(k, len(self.texts[dim])) 
            
        # Simple exclusion: get more results and filter afterwards
        search_k = k * 3 if exclude_ids else k  # Get extra to account for filtering
        
        if diversity_strategy == 'pure_similarity':
            initial_results = self._get_pure_similarity(query_text, dim, search_k)
        elif diversity_strategy == 'mmr':
            initial_results = self._get_mmr_examples(query_text, dim, search_k, diversity_weight)
        elif diversity_strategy == 'random_pure_similarity_mmr':
            initial_results = self._get_random_pure_similarity_mmr_examples(query_text, dim, search_k)
        else:
            raise ValueError(f"Unknown diversity strategy: {diversity_strategy}")
        
        # Apply exclusion filter if provided
        if exclude_ids and len(initial_results) > 0:
            initial_results = initial_results[~initial_results['id'].isin(exclude_ids)]
        
        # Take only k results
        return initial_results.head(k)
    
    def _get_pure_similarity(self, query_text: str, dim: str, k: int) -> pd.DataFrame:
        """Original pure similarity approach."""
        query_embedding = self.encoder.encode([query_text], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        similarities, indices = self.indices[dim].search(query_embedding, k)
        return self.metadata[dim].iloc[indices[0]].copy()
    
    def _get_mmr_examples(self, 
                         query_text: str, 
                         dim: str, 
                         k: int, 
                         lambda_param: float = 0.3) -> pd.DataFrame:
        """
        Maximal Marginal Relevance: balance similarity to query with diversity among selected examples.
        """
        candidate_k = min(k * 3, len(self.texts[dim]))
        
        query_emb = self.encoder.encode([query_text], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_emb)

        scores, cand_ids = self.indices[dim].search(query_emb, candidate_k)
        cand_ids = cand_ids[0]
        sim_to_query = scores[0]

        cand_embs = self.embeddings[dim][cand_ids]

        selected = []
        remaining = list(range(len(cand_ids)))

        first = int(np.argmax(sim_to_query))
        selected.append(first)
        remaining.remove(first)

        while len(selected) < k and remaining:
            mmr_scores = []
            for idx in remaining:
                cq = sim_to_query[idx]
                max_sim_sel = max(np.dot(cand_embs[idx], cand_embs[s]) for s in selected)
                mmr_scores.append(lambda_param * cq - (1 - lambda_param) * max_sim_sel)

            best_local = int(np.argmax(mmr_scores))
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining.remove(best_global)

        final_rows = self.metadata[dim].iloc[cand_ids[selected]].copy()
        return final_rows
 
    def _get_random_pure_similarity_mmr_examples(self, 
                                                query_text: str, 
                                                dim: str, 
                                                k: int) -> pd.DataFrame:
        """
        Random Pure Similarity MMR strategy: 25% random + 50% MMR (diversity penalty 0.3) + 25% pure similarity.
        
        Args:
            query_text: Text to find similar examples for
            dim: Dimension to search within
            k: Number of examples to retrieve
            
        Returns:
            DataFrame with k diverse examples using the tri-part strategy
        """
        # Calculate splits: 25% random, 50% MMR, 25% pure similarity
        random_k = max(1, int(k * 0.25))
        mmr_k = max(1, int(k * 0.5))
        similarity_k = k - random_k - mmr_k  # Remaining goes to pure similarity
        
        all_examples = []
        used_indices = set()
        
        # Phase 1: Get random examples (25%)
        if random_k > 0:
            available_indices = list(range(len(self.metadata[dim])))
            if len(available_indices) >= random_k:
                random_indices = self.rng.choice(available_indices, size=random_k, replace=False)
                random_examples = self.metadata[dim].iloc[random_indices].copy()
                all_examples.append(random_examples)
                used_indices.update(random_indices)
        
        # Phase 2: Get MMR examples with diversity penalty 0.2 (50%)
        if mmr_k > 0:
            # Create subset excluding already selected examples
            remaining_metadata = self.metadata[dim][~self.metadata[dim].index.isin(used_indices)].copy()
            remaining_indices = remaining_metadata.index.tolist()
            
            if len(remaining_indices) >= mmr_k:
                # Create temporary index for MMR on remaining examples
                remaining_embeddings = self.embeddings[dim][remaining_indices]
                
                # Encode query
                query_emb = self.encoder.encode([query_text], convert_to_numpy=True).astype('float32')
                faiss.normalize_L2(query_emb)
                
                # Calculate similarities to query for remaining examples
                sim_to_query = np.dot(remaining_embeddings, query_emb.T).flatten()
                
                # MMR algorithm with lambda=0.3 (diversity penalty 0.3)
                lambda_param = 0.3
                selected_local_indices = []
                remaining_local_indices = list(range(len(remaining_indices)))
                
                # Select first example (most similar to query)
                if remaining_local_indices:
                    first_idx = np.argmax(sim_to_query)
                    selected_local_indices.append(first_idx)
                    remaining_local_indices.remove(first_idx)
                
                # Select remaining examples using MMR
                while len(selected_local_indices) < mmr_k and remaining_local_indices:
                    mmr_scores = []
                    for local_idx in remaining_local_indices:
                        # Similarity to query
                        cq = sim_to_query[local_idx]
                        
                        # Maximum similarity to already selected examples
                        max_sim_selected = max(
                            np.dot(remaining_embeddings[local_idx], remaining_embeddings[sel_idx])
                            for sel_idx in selected_local_indices
                        )
                        
                        # MMR score with diversity penalty 0.3
                        mmr_score = lambda_param * cq - (1 - lambda_param) * max_sim_selected
                        mmr_scores.append(mmr_score)
                    
                    # Select example with highest MMR score
                    best_local_idx = np.argmax(mmr_scores)
                    best_idx = remaining_local_indices[best_local_idx]
                    selected_local_indices.append(best_idx)
                    remaining_local_indices.remove(best_idx)
                
                # Convert local indices back to original DataFrame indices
                mmr_global_indices = [remaining_indices[local_idx] for local_idx in selected_local_indices]
                mmr_examples = self.metadata[dim].iloc[mmr_global_indices].copy()
                all_examples.append(mmr_examples)
                used_indices.update(mmr_global_indices)
            elif len(remaining_indices) > 0:
                # Not enough remaining examples for full MMR, use what's available
                mmr_examples = remaining_metadata.copy()
                all_examples.append(mmr_examples)
                used_indices.update(remaining_indices)
        
        # Phase 3: Get pure similarity examples (25%)
        if similarity_k > 0:
            # Create subset excluding already selected examples
            remaining_metadata = self.metadata[dim][~self.metadata[dim].index.isin(used_indices)].copy()
            remaining_indices = remaining_metadata.index.tolist()
            
            if len(remaining_indices) >= similarity_k:
                # Create temporary FAISS index for remaining examples
                remaining_embeddings = self.embeddings[dim][remaining_indices]
                temp_index = faiss.IndexFlatIP(self.dimension)
                temp_index.add(remaining_embeddings)
                
                # Encode query
                query_emb = self.encoder.encode([query_text], convert_to_numpy=True).astype('float32')
                faiss.normalize_L2(query_emb)
                
                # Get most similar examples
                similarities, temp_indices = temp_index.search(query_emb, similarity_k)
                similarity_global_indices = [remaining_indices[i] for i in temp_indices[0]]
                similarity_examples = self.metadata[dim].iloc[similarity_global_indices].copy()
                all_examples.append(similarity_examples)
            elif len(remaining_indices) > 0:
                # Not enough remaining examples, use what's available
                similarity_examples = remaining_metadata.copy()
                all_examples.append(similarity_examples)
        
        # Combine all examples
        if all_examples:
            combined = pd.concat(all_examples, ignore_index=True)
            # Shuffle the final result to mix the different strategy types
            return combined.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            # Fallback: return empty DataFrame with correct structure
            return pd.DataFrame(columns=self.metadata[dim].columns)