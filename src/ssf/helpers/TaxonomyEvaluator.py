from ssf.utils import InferenceUtils, TaxonomyUtils
from ssf.Constants import NO_OP_MSG
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import os

def map_moral_values(labels):
    """Map detailed moral values to simplified higher-level categories.
    
    Args:
        labels: List of moral value labels
        
    Returns:
        List of mapped labels to simplified categories
    """
    # Mapping from detailed moral values to simplified categories
    moral_mapping = {
        'achievement': 'self_enhancement',
        'power': 'self_enhancement',
        'stimulation': 'openness_to_change',
        'self-direction': 'openness_to_change',
        'security': 'conservation',
        'conformity': 'conservation',
        'tradition': 'conservation',
        'universalism': 'self_transcendence',
        'benevolence': 'self_transcendence',
        'hedonism': 'hedonism',
        'other': 'other'
    }
    
    mapped_labels = []
    for label in labels:
        mapped_label = moral_mapping.get(label, label)  # Keep original if not in mapping
        if mapped_label not in mapped_labels:  # Avoid duplicates
            mapped_labels.append(mapped_label)
    
    return mapped_labels

class TaxonomyEvaluator:
    """
    Evaluates classification results against ground truth.
    """
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy

    def _parse_prediction_output(self, output):
        """Shared helper to parse a single prediction output."""
        try:
            output_text = output.get('output', '')
            parsed = InferenceUtils.parse_json(
                output_text.lstrip("```json").rstrip("```").strip()
            )
            categories = parsed.get('response', [])
            if not isinstance(categories, list):
                categories = [categories] if categories else []
            return categories
        except Exception as e:
            print(f"Error parsing prediction: {e}")
            return []

    def get_dim_eval_inputs_from_dir(self, pred_dir, ground_truth_data):
        """Same as original get_dim_eval_inputsXXX but with better name."""
        # Organize ground truth by dimension (same as original)
        dims = ground_truth_data['dim'].tolist()
        texts = ground_truth_data['response'].tolist()
        refs = ground_truth_data['labels'].tolist()

        dim_dict = defaultdict(lambda: defaultdict(list))
        for ref, dim, text in zip(refs, dims, texts):
            dim_dict[dim]['refs'].append(ref)
            dim_dict[dim]['texts'].append(text)

        # Load predictions for each dimension (same logic, using helper)
        outputs_dir = os.path.join(pred_dir, 'outputs')
        for dim in dim_dict.keys():
            outputs_file = os.path.join(outputs_dir, f'{dim}.jsonl')

            if not os.path.exists(outputs_file):
                print(f"Warning: No outputs file found for dimension {dim}")
                dim_dict[dim]['preds'] = [[] for _ in range(len(dim_dict[dim]['refs']))]
                continue

            predictions = InferenceUtils.read_jsonl(outputs_file)
            dim_dict[dim]['preds'] = [self._parse_prediction_output(output) for output in predictions]

        return dim_dict

    def get_dim_eval_inputs_from_file(self, predictions_file, ground_truth_data):
        """Same as original commented function but using shared parsing helper."""
        predictions = InferenceUtils.read_jsonl(predictions_file)

        # Parse predictions (using helper but preserving NO_OP_MSG check)
        preds = []
        for output in predictions:
            if output['output'] == NO_OP_MSG:
                print("THIS SHOULD NEVER HAPPEN!!!!!!")
                preds.append([])  # Preserve original NO_OP handling
            else:
                preds.append(self._parse_prediction_output(output))

        # Organize by dimension (same as original)
        dims = ground_truth_data['dim'].tolist()
        texts = ground_truth_data['response'].tolist()
        refs = ground_truth_data['labels'].tolist()

        from collections import defaultdict
        dim_dict = defaultdict(lambda: defaultdict(list))
        for ref, pred, dim, text in zip(refs, preds, dims, texts):
            dim_dict[dim]['preds'].append(pred)
            dim_dict[dim]['refs'].append(ref)
            dim_dict[dim]['texts'].append(text)

        return dim_dict
    
    def evaluate_all_predictions(self, all_predictions, use_simplified_moral=True, verbose=False):
        """Evaluate predictions using the complete dataset across all folds.

        Args:
            dim_eval_inputs_dict: Dict with dim -> {'refs': [], 'preds': [], 'texts': []}
            use_simplified_moral: Whether to apply moral dimension mapping to simplified categories

        Returns:
            Dict with strategy -> k -> dimension -> metrics
        """
        print("Computing final classification metrics...")
        dim_eval_outputs_dict = {}

        for dim, eval_inputs in all_predictions.items():
            refs, preds = self._filter_excluded_labels(dim, eval_inputs['refs'], eval_inputs['preds'])

            if dim == 'moral' and use_simplified_moral:
                refs = [map_moral_values(ref_list) for ref_list in refs]
                preds = [map_moral_values(pred_list) for pred_list in preds]

            # Get labels that appear in either references or predictions
            appearing_labels = set()
            for ref_list in refs:
                appearing_labels.update(ref_list)
            # for pred_list in preds:
            #     appearing_labels.update(pred_list)
            all_labels = sorted(list(appearing_labels))
            
            # Convert to binary format
            mlb = MultiLabelBinarizer(classes=all_labels)
            refs_bin = mlb.fit_transform(refs)
            preds_bin = mlb.transform(preds)
            
            # Compute classification report
            report = classification_report(y_true=refs_bin,
                                           y_pred=preds_bin,
                                           zero_division=0, 
                                           output_dict=True)
            
            dim_eval_outputs_dict[dim] = {
                'micro_precision': report['micro avg']['precision'],
                'micro_recall': report['micro avg']['recall'],
                'micro_f1': report['micro avg']['f1-score'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'full_report': report,
                'num_examples': len(refs)
            }

            if verbose:
                print(f"\n  {dim} - Label-specific metrics:")
                print(f"    Overall: Micro F1={report['micro avg']['f1-score']:.3f}, Macro F1={report['macro avg']['f1-score']:.3f}")
                    
                # Extract and sort label-specific metrics
                label_metrics = []
                for i, label in enumerate(all_labels):
                    # Try different ways to access the label in the report
                    label_key = None
                    possible_keys = [str(label), label, str(i), i]
                    
                    for key in possible_keys:
                        if key in report and isinstance(report[key], dict):
                            label_key = key
                            break
                    
                    if label_key is not None:
                        label_info = report[label_key]
                        label_metrics.append({
                            'label': label,
                            'precision': label_info['precision'],
                            'recall': label_info['recall'],
                            'f1-score': label_info['f1-score'],
                            'support': int(label_info['support'])
                        })
                    else:
                        print(f"    Warning: Could not find metrics for label '{label}'")
                
                if label_metrics:
                    # Sort by support (descending) to show most common labels first
                        label_metrics.sort(key=lambda x: x['support'], reverse=True)
                        
                        # Print label metrics in a formatted table
                        print(f"    {'Label':<20} {'Precision':<9} {'Recall':<9} {'F1-Score':<9} {'Support':<7}")
                        print(f"    {'-'*20} {'-'*9} {'-'*9} {'-'*9} {'-'*7}")
                        for metric in label_metrics:
                            print(f"    {metric['label']:<20} {metric['precision']:<9.3f} {metric['recall']:<9.3f} "
                                    f"{metric['f1-score']:<9.3f} {metric['support']:<7}")
                        
                        # Identify problematic labels (low F1 scores)
                        low_f1_labels = [m for m in label_metrics if m['f1-score'] < 0.3 and m['support'] > 0]
                        if low_f1_labels:
                            print(f"    Low-performing labels (F1 < 0.3): {[l['label'] for l in low_f1_labels]}")
                        
                        # Calculate support-weighted metrics to understand macro vs micro difference
                        total_support = sum(m['support'] for m in label_metrics)
                        if total_support > 0:
                            weighted_f1 = sum(m['f1-score'] * m['support'] for m in label_metrics) / total_support
                            print(f"    Support-weighted F1: {weighted_f1:.3f} (should ≈ micro F1)")
                            
                            # Show distribution of support across labels
                            support_values = [m['support'] for m in label_metrics]
                            if len(support_values) > 1:
                                support_std = np.std(support_values)
                                support_mean = np.mean(support_values)
                                print(f"    Support distribution: mean={support_mean:.1f}, std={support_std:.1f}")
                                if support_std / support_mean > 0.5:  # High relative variance
                                    print(f"    ⚠️  Imbalanced label distribution may explain macro vs micro F1 gap")
                else:
                    print("    Could not extract label-specific metrics from classification report")

        # Print overall results
        for dim, metrics in dim_eval_outputs_dict.items():
            print(f"  {dim:<25} | Micro F1: {metrics['micro_f1']:.2f} | Macro F1: {metrics['macro_f1']:.2f} | Examples: {metrics['num_examples']}")

        return dim_eval_outputs_dict

    def _filter_excluded_labels(self, dim, refs, preds):
        """Filter out predictions/references for categories marked as include_in_tax_class=False."""
        excluded_categories = self.taxonomy.get_excluded_categories(dim)
        if not excluded_categories:
            return refs, preds
        filtered_refs = []
        filtered_preds = []
        for ref, pred in zip(refs, preds):
            # Filter reference labels
            if isinstance(ref, str):
                ref_labels = TaxonomyUtils.parse_labels(ref)
            else:
                ref_labels = ref if isinstance(ref, list) else [ref] if ref else []
            # Filter prediction labels  
            pred_labels = pred if isinstance(pred, list) else [pred] if pred else []
            
            # Remove excluded categories from both refs and preds
            filtered_ref_labels = [label for label in ref_labels if label not in excluded_categories]
            filtered_pred_labels = [label for label in pred_labels if label not in excluded_categories]
            
            filtered_refs.append(filtered_ref_labels)
            filtered_preds.append(filtered_pred_labels)
        return filtered_refs, filtered_preds