from convokit import Transformer, Corpus
from ssf.Constants import *
from tqdm import tqdm
from convokit import Corpus
import time
from googleapiclient.errors import HttpError
from googleapiclient import discovery

def clip_string_to_max_bytes(string, max_bytes):
    encoded = string.encode('utf-8')
    if len(encoded) <= max_bytes:
        return string
    truncated_bytes = encoded[:max_bytes]
    truncated_string = truncated_bytes.decode('utf-8', 'ignore')
    return truncated_string

class PerspectiveTransformer(Transformer):
  
  def __init__(self, 
               gcloud_api_key,
               meta_key=PERSPECTIVE_META_KEY,
               requested_attributes=PERSPECTIVE_ATTRIBUTES):
    self.client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=gcloud_api_key,
      discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False)
    self.meta_key = meta_key
    self.requested_attributes = requested_attributes
    self.byte_limit = 20480
    
  def fit(self, corpus: Corpus):
    return
  
  def transform(self, corpus: Corpus):
    count = 0
    for utt in tqdm(corpus.iter_utterances(), total=len(corpus.utterances)):
      if self.meta_key in utt.get_meta():
         continue
      utt.add_meta(key=self.meta_key,
                   value=self._get_attribute_scores(utt.text))
      count += 1
    return corpus

  def _get_attribute_scores(self, text):
      if not text:
          return {att_key: 0.0 for att_key in self.requested_attributes}
      if len(text.encode("utf-8")) >= self.byte_limit:
          text = clip_string_to_max_bytes(text, self.byte_limit)

      analyze_request = {
      'comment': { 'text': text },
      'languages': ['en'],
      'requestedAttributes': {att_key: {} for att_key in self.requested_attributes},
      'doNotStore': True
      }
      
      retries = 10
      for attempt in range(retries):
          try:
              response = self.client.comments().analyze(body=analyze_request).execute()
              time.sleep(1 / 24)      # based on quota
              return {att_key: response['attributeScores'][att_key]['summaryScore']['value'] for att_key in self.requested_attributes}
          except HttpError as e:
              if e.resp.status in [429]:
                  if attempt < retries - 1:
                      wait_time = (2 ** attempt) * 5  # Exponential backoff
                      time.sleep(wait_time)
                  else:
                      print("Max retries reached. Exiting.")
                      raise
              else:
                  raise