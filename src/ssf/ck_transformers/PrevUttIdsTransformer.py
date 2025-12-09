from convokit import Transformer, Corpus, Utterance
from ssf.helpers import get_parent_utt
from ssf.Constants import *
from tqdm import tqdm
from ssf.Exceptions import *

class PrevUttIdsTransformer(Transformer):

  def __init__(self, 
               meta_key=PREV_UTT_IDS_META_KEY):
    self.meta_key = meta_key
    self.conv_chron_utts_dict = {}

  def fit(self, corpus: Corpus):
    for conv in corpus.iter_conversations():
      self.conv_chron_utts_dict[conv.get_id()] = conv.get_chronological_utterance_list()
    return
  
  def transform(self, corpus: Corpus):
    for utt in tqdm(corpus.iter_utterances(), total=len(corpus.utterances)):
      utt.add_meta(key=self.meta_key, 
                   value=self._get_prev_utt_ids(utt, corpus))
    return corpus

  def _get_prev_utt_ids(self, utt: Utterance, corpus: Corpus):
    try:
      parent_utt = get_parent_utt(utt, corpus)
      conv_chron_utts = self.conv_chron_utts_dict[utt._get_conversation_id()]
      parent_scoped_chron_utt_ids = [utt.get_id() for utt in conv_chron_utts if utt.reply_to == parent_utt.get_id()]
      utt_pos = parent_scoped_chron_utt_ids.index(utt.get_id())
      return parent_scoped_chron_utt_ids[:utt_pos]
    except ResourceDoesNotExistException:
      return []