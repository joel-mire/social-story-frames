"""
Pure taxonomy classification interface - no training, no evaluation.
"""
from abc import ABC, abstractmethod

class TaxonomyClassifier(ABC):
    
    def __init__(self, taxonomy):
        self.taxonomy = taxonomy
    
    @abstractmethod
    def classify_texts(self, texts, dims, output_path, force_redo=False):
        """
        Classify a list of texts for given dimensions.
        
        Args:
            texts: List of texts to classify
            dims: List of dimensions (same length as texts)
            output_path: Path to save classification results
            force_redo: Whether to regenerate if output exists
            
        Returns:
            Path to output file
        """
        pass