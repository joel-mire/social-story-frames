import evaluate
from sentence_transformers import SentenceTransformer
import string
import re
from scipy.spatial.distance import cosine

def aggregate(vals: list[float],
               agg_strategy):
  if agg_strategy == "avg":
    return sum(vals) / len(vals) if vals else 0.0
  elif agg_strategy == "max":
    return max(vals) if vals else 0.0
  elif agg_strategy == "min":
    return min(vals) if vals else 0.0
  
def _normalize_text(text: str) -> str:
  text = text.lower().strip()
  text = text.strip(string.punctuation)
  text = re.sub(r"\s+", " ", text)
  return text

def _normalize_pred_and_refs(pred_varVals: list[str], 
                             refs_varVals: list[list[str]]):
  pred_varVals = [_normalize_text(pred_varVal) for pred_varVal in pred_varVals]
  refs_varVals = [[_normalize_text(ref) for ref in refs_varVal] for refs_varVal in refs_varVals]
  return pred_varVals, refs_varVals

def _should_fast_fail(pred_varVals: list[str],
                       refs_varVals: list[list[str]]):
  return any(not pred_varVal for pred_varVal in pred_varVals) or all(all(not ref_varVal for ref_varVal in refs_varVal)  for refs_varVal in  refs_varVals)

class MetricExecutor:
  def __init__(self, sbert_model: str):
    self.encoder = SentenceTransformer(sbert_model)
    self._bert = evaluate.load("bertscore")
    self._bleu = evaluate.load("bleu")
    self._meteor = evaluate.load("meteor")

  def cosine_similarity(self, 
                        pred_varVals: list[str], 
                        refs_varVals: list[list[str]],
                        refs_agg_strategy: str = "avg"):
    pred_varVals, refs_varVals = _normalize_pred_and_refs(pred_varVals, refs_varVals)
    if _should_fast_fail(pred_varVals, refs_varVals):
      print("Fast fail condition met, returning 0.0")
      return 0.0
    vals = []
    for ref_varVal in refs_varVals:
      metric_vals = []
      for pred_varVal, ref_varVal in zip(pred_varVals, ref_varVal):
        emb = self.encoder.encode([pred_varVal, ref_varVal])
        cosine_sim = 1 - cosine(emb[0], emb[1])
        metric_vals.append(cosine_sim)
      vals.append(aggregate(metric_vals, agg_strategy="avg"))
    return aggregate(vals=vals, 
                                  agg_strategy=refs_agg_strategy)

  def bert_score(self, 
                        pred_varVals: list[str], 
                        refs_varVals: list[list[str]],
                        refs_agg_strategy: str = "avg"):
    pred_varVals, refs_varVals = _normalize_pred_and_refs(pred_varVals, refs_varVals)
    if _should_fast_fail(pred_varVals, refs_varVals):
      print("Fast fail condition met, returning 0.0")
      return 0.0
    vals = []
    for ref_varVal in refs_varVals:
      metric_vals = []
      for pred_varVal, ref_varVal in zip(pred_varVals, ref_varVal):
        result = self._bert.compute(predictions=[pred_varVal], 
                                    references=[ref_varVal],
                                    lang="en")
        metric_vals.append(result["f1"][0])
      vals.append(aggregate(metric_vals, agg_strategy="avg"))
    return aggregate(vals=vals, 
                                  agg_strategy=refs_agg_strategy)
  
  def bleu(self, 
                        pred_varVals: list[str], 
                        refs_varVals: list[list[str]],
                        refs_agg_strategy: str = "max"):
    if refs_agg_strategy != "max":
      raise ValueError(f"Unsupported refs_agg_strategy: {refs_agg_strategy}. Only 'max' is supported for BLEU.")
    pred_varVals, refs_varVals = _normalize_pred_and_refs(pred_varVals, refs_varVals)
    if _should_fast_fail(pred_varVals, refs_varVals):
      print("Fast fail condition met, returning 0.0")
      return 0.0
    vals = []
    for ref_varVal in refs_varVals:
      metric_vals = []
      for pred_varVal, ref_varVal in zip(pred_varVals, ref_varVal):
        result = self._bleu.compute(predictions=[pred_varVal],
                                  references=[ref_varVal],
                                  max_order=3,
                                  smooth=True)
        metric_vals.append(result["bleu"])
      vals.append(aggregate(metric_vals, agg_strategy="avg"))
    return aggregate(vals=vals, 
                                  agg_strategy=refs_agg_strategy)

  def meteor(self, 
              pred_varVals: list[str], 
              refs_varVals: list[list[str]],
              refs_agg_strategy: str = "max"):
    if refs_agg_strategy != "max":
      raise ValueError(f"Unsupported refs_agg_strategy: {refs_agg_strategy}. Only 'max' is supported for METEOR.")
    pred_varVals, refs_varVals = _normalize_pred_and_refs(pred_varVals, refs_varVals)
    if _should_fast_fail(pred_varVals, refs_varVals):
      print("Fast fail condition met, returning 0.0")
      return 0.0
    vals = []
    for ref_varVal in refs_varVals:
      metric_vals = []
      for pred_varVal, ref_varVal in zip(pred_varVals, ref_varVal):
        result = self._meteor.compute(predictions=[pred_varVal],
                                         references=[ref_varVal])
        metric_vals.append(result["meteor"])
      vals.append(aggregate(metric_vals, agg_strategy="avg"))
    return aggregate(vals=vals, 
                                  agg_strategy=refs_agg_strategy)
