from convokit import Transformer, Corpus
from ssf.Constants import *
from transformers import pipeline
from tqdm import tqdm

class StoryProbTransformer(Transformer):

  def __init__(self, 
               meta_key=STORY_SEEKER_META_KEY,
               hugging_face_model_name=STORY_SEEKER_MODEL_NAME):
    self.meta_key = meta_key
    self.pipe = pipeline(model=hugging_face_model_name, 
                         device_map='auto')

  def fit(self, corpus: Corpus):
    return
  
  def transform(self, corpus: Corpus):
    for utt in tqdm(corpus.iter_utterances(), total=len(corpus.utterances)):
      utt.add_meta(key=self.meta_key, 
                   value=self._get_story_prob(utt.text))
    return corpus

  def _get_story_prob(self, text):
    if not text:
      return 0.0
    pred_result = self.pipe(text, truncation=True)[0]
    label = pred_result['label']
    score = pred_result['score']
    if label == 'LABEL_1':
      return score
    elif label == 'LABEL_0':
      return 1 - score