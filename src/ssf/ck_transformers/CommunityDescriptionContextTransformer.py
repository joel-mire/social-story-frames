from convokit import Transformer, Corpus
from ssf.Constants import *
from tqdm import tqdm

class CommunityDescriptionContextTransformer(Transformer):
  def __init__(self,
               community_data_dict,
               meta_key=COMMUNITY_DESCRIPTION_META_KEY):
    self.community_data_dict = community_data_dict
    self.meta_key = meta_key

  def fit(self, corpus: Corpus, selector=None):
    return
    
  def transform(self, corpus: Corpus, selector=None):
    for utt in tqdm(corpus.iter_utterances(selector=selector), total=len(corpus.utterances)):
      community = utt.get_meta()[COMMUNITY_META_KEY]

      community_description = self.community_data_dict[community]['description']
      utt.add_meta(key=self.meta_key, 
                   value=community_description)
    return corpus