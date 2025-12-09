from convokit import Transformer, Corpus, Utterance
from ssf.Constants import *
from tqdm import tqdm
from ssf.Exceptions import *
from ssf.helpers import get_parent_utt

class AncestralUttIdsTransformer(Transformer):

  def __init__(self, 
               meta_key=ANCESTRAL_UTT_IDS_META_KEY):
    self.meta_key = meta_key

  def fit(self, corpus: Corpus):
    return
  
  def transform(self, corpus: Corpus):
    for utt in tqdm(corpus.iter_utterances(), total=len(corpus.utterances)):
      utt.add_meta(key=self.meta_key, 
                   value=self._get_ancestral_utt_ids(utt, corpus))
    return corpus

  def _get_ancestral_utt_ids(self, utt: Utterance, corpus: Corpus):
    ancestor_utt_ids = []
    cur_utt = utt
    while True:
      try:
        parent_utt = get_parent_utt(cur_utt, corpus)
        ancestor_utt_ids.append(parent_utt.get_id())
        cur_utt = parent_utt
      except ResourceDoesNotExistException:
        return list(reversed(ancestor_utt_ids))