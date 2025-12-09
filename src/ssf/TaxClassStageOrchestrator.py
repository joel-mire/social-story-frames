import pandas as pd
import os
from sklearn.model_selection import GroupKFold
from ssf.Constants import DIM_TEST_SET_COUNT
from ssf.classifiers.BatchTaxonomyClassifier import BatchTaxonomyClassifier
from ssf.classifiers.BatchProcessor import BatchNotReadyError
from ssf.classifiers.GptTaxonomyClassifier import GptTaxonomyClassifier
import json
from tqdm import tqdm
from ssf.SemanticSimilarityIndex import SemanticSimilarityIndex
import json
from ssf.helpers import TaxonomyEvaluator
    
class TaxClassStageOrchestrator:
    def __init__(self, ann_dir, taxonomy, model_name, ssf_gen_base_model, random_seed, show_progress=False):
        self.id_col = 'id'
        self.ann_dir = ann_dir
        self.taxonomy = taxonomy
        self.model_name = model_name
        self.ssf_gen_base_model = ssf_gen_base_model
        self.taxonomy_evaluator = TaxonomyEvaluator(taxonomy=self.taxonomy) 
        self.show_progress = show_progress

        # Load both validation and test annotation data during initialization
        self.val_ann_df = self._load_annotation_data(self.ann_dir, 'val')
        self.test_ann_df = self._load_annotation_data(self.ann_dir, 'test')

        self.similarity_index = SemanticSimilarityIndex(random_seed=random_seed)
        self.similarity_index.build_index(few_shot_df=self.val_ann_df, text_col='response')

    def _parse_labels(self, labels_str):
        """Parse comma-separated label string into list of labels."""
        if pd.isna(labels_str) or labels_str.strip() == '':
            return []
        return [label.strip() for label in labels_str.split(',') if label.strip()]

    def _load_annotation_data(self, ann_dir, split_name):
        """Load and combine annotation data from all dimensions for a specific split.

        Parameters
        ----------
        ann_dir : str
            Directory containing annotation files
        split_name : str
            Either 'val' or 'test' - determines file naming pattern and row count
            
        Returns
        -------
        combined_df : pd.DataFrame
            Combined annotation data across all dimensions
        """
        combined_dfs = []
        for dim in self.taxonomy.get_dims():
            path = f"{ann_dir}/{split_name}_{dim}_hum_ann.csv"
            
            # Determine how many rows to load based on split type
            if split_name == 'test':
                # Use dimension-specific test set count
                row_count = DIM_TEST_SET_COUNT[dim]
            else:
                # For validation, always use 100
                row_count = 100
                
            df = pd.read_csv(path).head(row_count)
            df['labels'] = df['labels'].apply(self._parse_labels)
            dim_prefix = self.taxonomy.get_template_prefix(dim)
            df['response'] = df['response'].apply(lambda x: f"{dim_prefix}{x}")
            df['dim'] = dim
            combined_dfs.append(df)
            
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        return combined_df

    def get_fold_dfs(self, eval_split):
        if eval_split == 'val':
            return self.get_cv_dfs(self.val_ann_df, 
                                   k_folds=100)
        elif eval_split == 'test':
            # we will eval the whole test df
            test_ann_df_copy = self.test_ann_df.copy()
            test_ann_df_copy['split'] = 'eval'  # All input data is eval data

            # and use the val df as training data (for few-show examples)
            val_ann_train_df = self.val_ann_df.copy()
            val_ann_train_df['split'] = 'train'

            joint_df = pd.concat([val_ann_train_df, test_ann_df_copy], ignore_index=True)
            return [joint_df]
        else:
            raise ValueError(f"eval_split must be 'val' or 'test', got: {eval_split}")


    def get_cv_dfs(self, dataset_df, k_folds):
        """
        Split data into train/validation using sklearn's GroupKFold to ensure entire IDs stay together.

        Parameters
        ----------
        dataset_df : pd.DataFrame
            Dataset with self.id_col identifying groups
        k_folds : int
            Number of folds
            
        Returns
        -------
        cv_splits : list[pd.DataFrame]
            Each element is a copy of dataset_df with a new column 'split' 
            where 'split' == 'train' for training data and 'split' == 'val' for validation data.
        """
        unique_ids = dataset_df[self.id_col].unique()
        assert k_folds <= len(unique_ids), f"Need at least {k_folds} unique ids, have only {len(unique_ids)}."

        # Use GroupKFold to split by ID groups for k-fold cross-validation
        gkf = GroupKFold(n_splits=k_folds)
        groups = dataset_df[self.id_col]

        cv_splits = []
        for _, val_idx in gkf.split(dataset_df, groups=groups):
            fold_split = dataset_df.copy()
            # Start with all data as training
            fold_split['split'] = 'train'
            # Mark validation indices
            fold_split.iloc[val_idx, fold_split.columns.get_loc('split')] = 'eval'
            cv_splits.append(fold_split)

        return cv_splits


    def run_classification_with_evaluation(self, 
                                            eval_split, 
                                            dim_k_dict, 
                                            diversity_strategy,
                                            force_redo, 
                                            output_dir, 
                                            use_batch_api, 
                                            random_seed):
        """
        Run complete classification with evaluation.

        Returns
        -------
        final_results : dict
            Classification results by dimension
        all_predictions : dict
            Raw predictions for further analysis
        """
        # Check for cached results if force_redo is False
        results_cache_path = f"{output_dir}/final_results.json"
        predictions_cache_path = f"{output_dir}/all_predictions.json"
        
        # Check for batch jobs first if using batch API
        batch_jobs_exist = False
        if use_batch_api and not force_redo:
            # Check if any batch jobs exist for this stage
            batch_job_path = f"{output_dir}/fold_0/batch_jobs/batch_job.json"
            if os.path.exists(batch_job_path):
                batch_jobs_exist = True
            if batch_jobs_exist:
                print("[TaxClassStageOrchestrator] Found batch jobs, skipping cached predictions...")
        
        if not force_redo and os.path.exists(predictions_cache_path) and not (use_batch_api and batch_jobs_exist):
            with open(predictions_cache_path, 'r') as f:
                all_predictions = json.load(f)
            # Re-running evaluation on cached predictions...

            # Evaluate all predictions
            print(f"\nResults ({eval_split}):")
            final_results = self.taxonomy_evaluator.evaluate_all_predictions(all_predictions)
            
            # Save results to cache
            with open(results_cache_path, 'w') as f:
                json.dump(final_results, f, indent=2)
                
            return final_results, all_predictions
        
        # Get eval dfs with 'train' and 'eval' splits
        fold_dfs = self.get_fold_dfs(eval_split=eval_split)

        # Collect predictions across all folds
        all_predictions = {}  # dim -> {'refs': [], 'preds': [], 'texts': []}

        # Create unified progress bar for all folds
        total_examples = sum(sum(fold_df['split'] == 'eval') for fold_df in fold_dfs)
        pbar = tqdm(total=total_examples, desc=f"{eval_split.capitalize()} classification", unit="examples")
        
        for fold_idx, fold_df in enumerate(fold_dfs):
            # Create output directory for this fold
            fold_out_dir = f"{output_dir}/fold_{fold_idx}"
            os.makedirs(fold_out_dir, exist_ok=True)
            
            # Create classifier for this fold
            train_data = fold_df[fold_df['split'] == 'train']
            eval_data = fold_df[fold_df['split'] == 'eval']
            
            if use_batch_api:
                assert eval_split == 'test', "Batch API can only be used for test set evaluation, not validation set evaluation."
                gpt_tax_classifier = BatchTaxonomyClassifier(
                    taxonomy=self.taxonomy,
                    few_shot_data=train_data,
                    model_name=self.model_name, 
                    ssf_gen_base_model=self.ssf_gen_base_model,
                    dim_k_dict=dim_k_dict,
                    diversity_strategy=diversity_strategy,
                    show_progress=self.show_progress,
                    similarity_index=self.similarity_index)
            else:
                gpt_tax_classifier = GptTaxonomyClassifier(
                    taxonomy=self.taxonomy,
                    model_name=self.model_name,
                    few_shot_data=train_data,
                    dim_k_dict=dim_k_dict,
                    diversity_strategy=diversity_strategy,
                    show_progress=self.show_progress,
                    similarity_index=self.similarity_index)
            
            # Classify evaluation data (no training needed for GPT)
            texts = eval_data['response'].tolist()
            dims = eval_data['dim'].tolist()
            instance_ids = eval_data['id'].tolist() if eval_split == 'val' or use_batch_api else None
            outputs_path = f"{fold_out_dir}/outputs.jsonl"
            
            try:
                gpt_tax_classifier.classify_texts(texts, dims, instance_ids, outputs_path, force_redo)
            except BatchNotReadyError as e:
                if use_batch_api:
                    print(f"[TaxClassStageOrchestrator] {str(e)}")
                    print("[TaxClassStageOrchestrator] Waiting for existing batch job to complete. Please run again later to collect results.")
                    return None, None  # Return early, caller should handle this
                else:
                    raise e
            
            # Update progress bar
            eval_examples = sum(fold_df['split'] == 'eval')
            pbar.update(eval_examples)
            
            # Collect predictions using evaluator
            # dim_dict = self.taxonomy_evaluator.get_dim_eval_inputs_from_dir(pred_dir=outputs_path, ground_truth_data=eval_data)
            dim_dict = self.taxonomy_evaluator.get_dim_eval_inputs_from_file(predictions_file=outputs_path, 
                                                                             ground_truth_data=eval_data)
            
            for dim, data in dim_dict.items():
                if dim not in all_predictions:
                    all_predictions[dim] = {'refs': [], 'preds': [], 'texts': []}
                all_predictions[dim]['refs'].extend(data['refs'])
                all_predictions[dim]['preds'].extend(data['preds'])
                all_predictions[dim]['texts'].extend(data['texts'])
        
        # Close progress bar
        pbar.close()

        # Evaluate all predictions
        print("Evaluating results...")
        final_results = self.taxonomy_evaluator.evaluate_all_predictions(all_predictions)
    
        # Cache results for future use
        with open(results_cache_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        with open(predictions_cache_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        return final_results, all_predictions
