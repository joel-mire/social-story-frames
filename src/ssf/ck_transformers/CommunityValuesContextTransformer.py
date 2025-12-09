from convokit import Transformer, Corpus
from ssf.Constants import *
from tqdm import tqdm

class CommunityValuesContextTransformer(Transformer):
  def __init__(self,
               community_data_dict,
               meta_key=COMMUNITY_VALUES_META_KEY):
    self.community_data_dict = community_data_dict
    self.meta_key = meta_key

  def fit(self, corpus: Corpus, selector=None):
    return
    
  def transform(self, corpus: Corpus, selector=None):
    for utt in tqdm(corpus.iter_utterances(selector=selector), total=len(corpus.utterances)):
      community = utt.get_meta()[COMMUNITY_META_KEY]

      community_values = self.community_data_dict[community]['values']
      utt.add_meta(key=self.meta_key,
                   value=community_values)
    return corpus