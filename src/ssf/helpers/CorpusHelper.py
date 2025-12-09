from convokit import Corpus, Utterance, Conversation, download
from ssf.Constants import *
from ssf.Exceptions import *
import nltk
from tqdm import tqdm
nltk.download('stopwords')
from nltk.corpus import stopwords
import shutil
import os
import random
from ssf.Configs import Corpus as CorpusConfig
from ssf.SelectorType import SelectorType

def get_parent_utt(utt: Utterance, corpus: Corpus):
    if utt.reply_to == None:
        raise ResourceDoesNotExistException()
    return corpus.get_utterance(utt.reply_to)

def get_progenitor_utt(utt: Utterance, corpus: Corpus):
    return corpus.get_utterance(utt.conversation_id)

def get_utt_texts(corpus: Corpus, selector=None):
    return [utt.text for utt in corpus.iter_utterances(selector=selector)]
    
class CorpusHelper:
    def __init__(self, 
                 config: CorpusConfig, 
                 corpus_dir):
        self.config = config
        self.corpus_dir = corpus_dir
        self.disqualified_set = {stopword for stopword in stopwords.words('english')}
        self.disqualified_set.update(self.config.disqualification_strings)
        self._cached_substantial_context_utt_ids = None

    def load_corpus(self,
                    force_rebuild):
        ext_corpus_path = f"{self.corpus_dir}/{self.config.extension_name}"
        if os.path.exists(ext_corpus_path):
            if force_rebuild:
                shutil.rmtree(ext_corpus_path)
            else:
                return Corpus(filename=ext_corpus_path)
        source_corpus = Corpus(download(self.config.original_name))
        all_conversation_ids = source_corpus.get_conversation_ids()
        num_convos = int(len(all_conversation_ids) * self.config.source_conversations_coverage)
        sampled_ids = random.sample(all_conversation_ids, k=num_convos)
        ext_corpus = source_corpus.filter_conversations_by(lambda c: c.id in sampled_ids)
        ext_corpus.add_meta(CORPUS_NAME_META_KEY, self.config.extension_name)
        self.save_corpus(ext_corpus)
        return ext_corpus

    def save_corpus(self, 
                    corpus, 
                    base_path=None):
        if base_path is None:
            base_path = self.corpus_dir
        os.makedirs(base_path, exist_ok=True)
        corpus.dump(name=self.get_corpus_name(corpus),
                    base_path=base_path)

    def get_utterances(self, conv: Conversation):
        return [conv.get_utterance(utt_id) for utt_id in conv.get_utterance_ids()]

    def get_communities(self, corpus: Corpus):
        return {utt.get_meta()[self.config.community_meta_key] for utt in corpus.iter_utterances()}
     
    def get_corpus_name(self, corpus: Corpus):
        return corpus.get_meta()[CORPUS_NAME_META_KEY]

    def save_corpus(self, 
                    corpus: Corpus,
                    base_path,
                    overwrite_existing_corpus=False):
        os.makedirs(base_path, exist_ok=True)
        corpus.dump(name=self.get_corpus_name(corpus),
                    base_path=base_path,
                    overwrite_existing_corpus=overwrite_existing_corpus)

    def get_substantial_context_utts(self, utt_ids, corpus):
        context_utts = [corpus.get_utterance(utt_id) for utt_id in utt_ids]
        substantial_context_utts = [utt for utt in context_utts if self.is_substantial_text(utt.text, min_char_length=self.config.substantial_text_min_chars.context_utt)]
        return substantial_context_utts

    def get_utt_ids(self, utts):
        return [utt.id for utt in utts]

    def get_substantial_context_utt_ids(self, utt_ids, corpus):
        return self.get_utt_ids(self.get_substantial_context_utts(utt_ids, corpus))

    def get_k_closest_substantial_context_utt_ids(self, utt: Utterance,
                                                  corpus: Corpus,
                                                  context_utt_meta_key: str,
                                                  k):
        context_utt_ids = utt.get_meta()[context_utt_meta_key]
        substantial_context_utts = self.get_substantial_context_utts(context_utt_ids, corpus)
        k_closest_substantial_context_utts = substantial_context_utts[-k:]
        return [utt.id for utt in k_closest_substantial_context_utts]

    def get_k_closest_substantial_context_utt_summaries_list(self, 
                                                             corpus: Corpus,
                                                            context_utt_meta_key,
                                                            summary_meta_key,
                                                            k,
                                                            selector=None):
        context_utt_summaries = []
        for utt in tqdm(corpus.iter_utterances(selector=selector), total=len(corpus.utterances)):
            context_utt_summaries.append(self.get_k_closest_substantial_context_utt_summaries(utt=utt,
                                                                                         corpus=corpus,
                                                                                         context_utt_meta_key=context_utt_meta_key,
                                                                                         summary_meta_key=summary_meta_key,
                                                                                         k=k))
        return context_utt_summaries

    def get_k_closest_substantial_context_utt_summaries(self, utt: Utterance,
                                                        corpus: Corpus,
                                                        context_utt_meta_key: str,
                                                        summary_meta_key: str,
                                                        k):
        context_utt_ids = utt.get_meta()[context_utt_meta_key]
        substantial_context_utts = self.get_substantial_context_utts(context_utt_ids, corpus)
        k_closest_substantial_context_utts = substantial_context_utts[-k:]
        k_closest_substantial_context_summaries = [utt.get_meta()[summary_meta_key] for utt in k_closest_substantial_context_utts]
        return k_closest_substantial_context_summaries

    def is_substantial_text(self, text, min_char_length):
        if text.strip() in self.disqualified_set:
            return False
        elif len(text) < min_char_length:
            return False
        return True

    def is_story(self, utt):
        return utt.meta[STORY_SEEKER_META_KEY] >= self.config.story_seeker_threshold

    def is_toxic(self, utt):
        return utt.meta[PERSPECTIVE_META_KEY][PERSPECTIVE_ATT_TOXICITY] >= self.config.perspective_toxicity_threshold

    def is_sexually_explicit(self, utt):
        return utt.meta[PERSPECTIVE_META_KEY][PERSPECTIVE_ATT_SEXUALLY_EXPLICIT] >= self.config.perspective_sexually_explicit_threshold

    def is_safe_for_annotators(self, utt):
        return not(self.is_toxic(utt) or self.is_sexually_explicit(utt))

    def get_all_substantial_context_utt_ids(self, corpus, selector):
        if self._cached_substantial_context_utt_ids is not None:
            return self._cached_substantial_context_utt_ids  # Return cached result

        substantial_context_utt_ids = set()
        i = 0
        for utt in corpus.iter_utterances(selector=selector):
            i += 1
            substantial_context_utt_ids.add(get_progenitor_utt(utt, corpus).id)
            # Use configured context chain count
            ancestor_k = self.config.convo_context.context_chain_utts
            context_utts = self.get_k_closest_substantial_context_utt_ids(utt=utt,
                                                                     corpus=corpus,
                                                                     context_utt_meta_key=ANCESTRAL_UTT_IDS_META_KEY,
                                                                     k=ancestor_k)
            substantial_context_utt_ids.update(context_utts)

            previous_k = self.config.convo_context.context_chain_utts
            context_utts = self.get_k_closest_substantial_context_utt_ids(utt=utt,
                                                                     corpus=corpus,
                                                                     context_utt_meta_key=PREV_UTT_IDS_META_KEY,
                                                                     k=previous_k)
            substantial_context_utt_ids.update(context_utts)

        self._cached_substantial_context_utt_ids = substantial_context_utt_ids
        return substantial_context_utt_ids
    
    def get_selector(self, selector_type, corpus, community_allowlist):
        if selector_type == SelectorType.ALL:
            return lambda utt: utt.get_meta()[self.config.community_meta_key] in community_allowlist
        elif selector_type == SelectorType.STORIES_SUPERSET:
            return lambda utt: utt.get_meta()[self.config.community_meta_key] in community_allowlist and self.is_story(utt) and self.is_substantial_text(utt.text, min_char_length=self.config.substantial_text_min_chars.story)
        elif selector_type == SelectorType.STORIES:
            return lambda utt: utt.get_meta()[self.config.community_meta_key] in community_allowlist and self.is_story(utt) and self.is_safe_for_annotators(utt) and self.is_substantial_text(utt.text, min_char_length=self.config.substantial_text_min_chars.story)
        elif selector_type == SelectorType.STORIES_AND_CONTEXTS:
            # Precompute the STORIES selector once
            story_selector = self.get_selector(SelectorType.STORIES, corpus=corpus, community_allowlist=community_allowlist)
            # Precompute the context utterance IDs once
            context_utt_ids = self.get_all_substantial_context_utt_ids(
                corpus, selector=story_selector
            )
            return lambda utt: (story_selector(utt) or utt.id in context_utt_ids)
        
    def get_ssf_df(self, corpus, community_allowlist):
        stories_selector = self.get_selector(SelectorType.STORIES, corpus=corpus, community_allowlist=community_allowlist)
        ssf_df = corpus.get_utterances_dataframe(selector=stories_selector)
        ssf_df['id'] = ssf_df.index
        return ssf_df