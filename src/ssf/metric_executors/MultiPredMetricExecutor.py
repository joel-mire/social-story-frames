
from ssf.metric_executors.MetricExecutor import aggregate
from ssf.metric_executors.MetricExecutor import MetricExecutor

class MultiPredMetricExecutor:

  def __init__(self, sbert_model: str):
    self.metric_executor = MetricExecutor(sbert_model=sbert_model)

  def execute(self, 
              metric_executor_fn, 
              preds_varVals: list[list[str]], 
              refs_varVals: list[list[str]],
              preds_agg_strategy,
              refs_agg_strategy):
    vals = [metric_executor_fn(pred_varVals=pred_varVals,
                               refs_varVals=refs_varVals,
                                refs_agg_strategy=refs_agg_strategy)
            for pred_varVals in preds_varVals]
    return aggregate(vals=vals, 
                      agg_strategy=preds_agg_strategy)

  def cosine_similarity(self, 
                        preds_varVals: list[list[str]], 
                        refs_varVals: list[list[str]],
                        preds_agg_strategy: str = "avg",
                        refs_agg_strategy: str = "avg"):
    return self.execute(self.metric_executor.cosine_similarity,
                        preds_varVals=preds_varVals, 
                        refs_varVals=refs_varVals,
                        preds_agg_strategy=preds_agg_strategy,
                        refs_agg_strategy=refs_agg_strategy)

  def bert_score(self, 
                 preds_varVals: list[list[str]], 
                 refs_varVals: list[list[str]],
                 preds_agg_strategy: str = "avg",
                 refs_agg_strategy: str = "avg"):
    return self.execute(self.metric_executor.bert_score,
                        preds_varVals=preds_varVals, 
                        refs_varVals=refs_varVals,
                        preds_agg_strategy=preds_agg_strategy,
                        refs_agg_strategy=refs_agg_strategy)

  def bleu(self, 
           preds_varVals: list[list[str]], 
           refs_varVals: list[list[str]],
           preds_agg_strategy: str = "avg"):
    return self.execute(self.metric_executor.bleu,
                        preds_varVals=preds_varVals, 
                        refs_varVals=refs_varVals,
                        preds_agg_strategy=preds_agg_strategy,
                        refs_agg_strategy="max")  # BLEU only supports max aggregation for refs
  
  def meteor(self,
             preds_varVals: list[list[str]], 
             refs_varVals: list[list[str]],
             preds_agg_strategy: str = "avg"):
    return self.execute(self.metric_executor.meteor,
                        preds_varVals=preds_varVals, 
                        refs_varVals=refs_varVals,
                        preds_agg_strategy=preds_agg_strategy,
                        refs_agg_strategy="max")  # METEOR only supports max aggregation for refs
  