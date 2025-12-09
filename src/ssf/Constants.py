DATA_DIR = '../data'
TAXONOMY_DIR = f'{DATA_DIR}/taxonomy'
TAXONOMY_TEX_PATH = f'{TAXONOMY_DIR}/taxonomy_table.tex'
SUBREDDITS_METADATA_DIR = f'{DATA_DIR}/subreddits_metadata'
SUBREDDITS_PATH = f'{SUBREDDITS_METADATA_DIR}/subreddits.csv'
SUBREDDIT_DESC_RULES_PATH = f'{SUBREDDITS_METADATA_DIR}/rules_subreddit_set.csv'
SSF_DF_PATH = 'ssf.csv'

REPLICATION_CONFIG_PATH = "../configs/replication.yaml"

SCRIPTS_DIR = "../scripts"

RESULTS_DIR = "../results"
COUNTS_RESULTS_DIR = f"{RESULTS_DIR}/counts"
ENTROPY_RESULTS_DIR = f"{RESULTS_DIR}/entropy"
SIMILARITY_RESULTS_DIR = f"{RESULTS_DIR}/similarity"
TAX_CLASS_RESULTS_DIR = f"{RESULTS_DIR}/tax_class"
DEMOGRAPHIC_RESULTS_DIR = f"{RESULTS_DIR}/demographics"
CONTEXT_SUMMARIZATION_RESULTS_DIR = f"{RESULTS_DIR}/context_summarization"
PLAUSIBILITY_RESULTS_DIR = f"{RESULTS_DIR}/plausibility"

NO_OP_MSG = "<<<NO_OP>>>"
SKIP_INFERENCE_PREFIX = "$$SKIP_INFERENCE$$"

CORPUS_NAME_META_KEY = 'corpusName'
ANCESTRAL_UTT_IDS_META_KEY = 'ancestralUttIds'
PREV_UTT_IDS_META_KEY = 'prevUttIds'
SUMMARY_META_KEY = 'summary'
PERSPECTIVE_META_KEY = 'perspective'
STORY_SEEKER_META_KEY = 'storySeeker'
COMMUNITY_META_KEY = 'subreddit'
ORIGINAL_UTT_TEXT_META_KEY = 'original'
ANCESTRAL_CONTEXT_META_KEY = 'ancestralContext'
PREVIOUS_CONTEXT_META_KEY = 'previousContext'
CONSOLIDATED_CONTEXT_META_KEY = 'consolidatedContext'
PROGENITOR_CONTEXT_META_KEY = 'progenitorContext'
CONVERSATION_CONTEXT_META_KEY = 'conversationContext'
COMMUNITY_DESCRIPTION_META_KEY = 'communityDescription'
COMMUNITY_VALUES_META_KEY = 'communityValues'
COMMUNITY_META_COL = f"meta.{COMMUNITY_META_KEY}"
COMMUNITY_DESCRIPTION_META_COL = f"meta.{COMMUNITY_DESCRIPTION_META_KEY}"
COMMUNITY_VALUES_META_COL = f"meta.{COMMUNITY_VALUES_META_KEY}"
PROGENITOR_SUMMARY_META_COL = f"meta.{PROGENITOR_CONTEXT_META_KEY}"
CONVERSATION_SUMMARY_META_COL = f"meta.{CONVERSATION_CONTEXT_META_KEY}"

GCLOUD_API_KEY_ENV_VAR_NAME = 'GCLOUD_API_KEY'
PERSPECTIVE_ATT_TOXICITY = 'TOXICITY'
PERSPECTIVE_ATT_SEXUALLY_EXPLICIT = 'SEXUALLY_EXPLICIT'
PERSPECTIVE_ATTRIBUTES=[PERSPECTIVE_ATT_TOXICITY, PERSPECTIVE_ATT_SEXUALLY_EXPLICIT]
STORY_SEEKER_MODEL_NAME = "mariaantoniak/storyseeker"

TAXONOMY_TASK_SINGULAR_DICT = {
  "overall_goal": "overall goal an author had in posting a particular comment",
  "narrative_intent": "narrative intent an author had in telling a particular story",
  "author_emotional_response": "emotion the author would feel after telling a particular story",
  "causal_explanation": "inferences many readers would make to explain aspects of a particular story",
  "prediction": "prediction many readers would make about what will happen next in a particular story",
  "moral": "moral or theme many readers would take away after reading a particular story",
  "narrative_feeling": "emotion many readers would feel in response to the narrative content of a particular story",
  "aesthetic_feeling": "aesthetic feeling many readers would experience in response to the form, techniques, or style of a particular story",
  "character_appraisal": "appraisal or judgment many readers would make about the character in a particular story",
  "stance": "the viewpoint or overall opinion many readers would adopt in response to the author's main argument or stance in a particular story",
}
TAXONOMY_TASK_PLURAL_DICT = {
  "overall_goal": "overall goal(s) an author had in posting a particular comment",
  "narrative_intent": "narrative intent(s) an author had in telling a particular story",
  "author_emotional_response": "emotion(s) the author would feel after telling a particular story",
  "causal_explanation": "inferences(s) many readers would make to explain aspects of a particular story",
  "prediction": "prediction(s) many readers would make about what will happen next in a particular story",
  "moral": "moral(s) or theme(s) many readers would take away after reading a particular story",
  "narrative_feeling": "emotion(s) many readers would feel in response to the narrative content of a particular story",
  "aesthetic_feeling": "aesthetic feeling(s) many readers would experience in response to the form, techniques, or style of a particular story",
  "character_appraisal": "appraisal(s) or judgment(s) many readers would make about the character(s) in a particular story",
  "stance": "the viewpoint(s) or overall opinion(s) many readers would adopt in response to the author's main argument or stance in a particular story",
}
PROMPT_COL_SUFFIX_FULL_CONTEXT = "prompt_default"
PROMPT_COL_SUFFIX_NO_CONTEXT = "prompt_noSubredditName_noSubredditDescription_noSubredditValues_noProgenitorSummary_noConversationSummary"
PROMPT_COL_SUFFIX_NO_COMMUNITY_CONTEXT = "prompt_noSubredditName_noSubredditDescription_noSubredditValues"
PROMPT_COL_SUFFIX_NO_CONVERSATION_CONTEXT = "prompt_noProgenitorSummary_noConversationSummary"
ALL_PROMPT_COL_SUFFIXES = [
  PROMPT_COL_SUFFIX_FULL_CONTEXT,
  PROMPT_COL_SUFFIX_NO_CONTEXT,
  PROMPT_COL_SUFFIX_NO_COMMUNITY_CONTEXT,
  PROMPT_COL_SUFFIX_NO_CONVERSATION_CONTEXT
]

DIM_TEST_SET_COUNT = {
  "overall_goal": 100,
  "narrative_intent": 101,
  "author_emotional_response": 297,
  "narrative_feeling": 257,
  "aesthetic_feeling": 156,
  "stance": 242,
  "prediction": 117,
  "causal_explanation": 135,
  "moral": 100,
  "character_appraisal": 259
}

# Taxonomy Classification Tips
OVERALL_GOAL_TIPS = f"""- sharing an experience to “highlight” a normative or controversial perspective => “persuade_debate”
- sharing a perspective/advice in a casual/helpful manner, without the sense that the narrator is personally invested in getting readers to take their advice  =>  “provide_info_support”
- sharing an experience to express, explain, or justify one’s identity, core beliefs, values, or emotions; or highlight why one feels a certain way; nostalgia => “affirm_identity_self”
- trying to “connect” or to “relate” or be “relatable” to other people; apologizing => both “provide_emotional_support” and “request_emotional_support”
  - if instead of relating to people, the focus in on sharing a “related” (i.e. topically relevant) story, just label “provide_experiential_support”
- trying to “empathize with” => “provide_emotional_support”
- references to humor; lighthearted or memorable stories => “entertain”
- if expressing disappointment and it is not obvious that they are just venting/expressing themself without concern for how people respond => implicit “request_emotional_support”
- sharing/highlighting a personal perspective doesn’t necessarily imply "provide_experiential_accounts"."""

NARRATIVE_INTENT_TIPS = """- explaining events, providing context, updates, or describing what happened **to correct misunderstandings or fill information gaps** (informational intent) => "clarify_what_transpired"
  - **do not use "clarify_what_transpired" simply because events are referenced; only when the primary intent is corrective or to add new facts/info or a personal account to the topic**
- revealing core values, identifications, or moments of personal growth or self-awareness => "show_identity"
- drawing explicit broader conclusions or offering advice/lessons that others could apply beyond the specific situation described => "justify_challenge_offer_belief_norm"
- advocating for an idea, belief, or opinion (e.g., by providing evidence to defend a new or existing claim); defending/supporting/explaining one's interpretation; contradicting misconceptions, challenging established narratives, or disputing commonly held beliefs (persuasive/argumentative intent) => "justify_challenge_offer_belief_norm"
- humor, lighthearted => "entertain"
- venting or intentionally expressing an intense emotion (e.g., anger, sadness, relief, etc) => "release_pent_up_emotions"
- seeking reassurance, validation, or emotional support (often through sharing struggles, asking for advice, expressing uncertainty, describing awkward/difficult situations, or revealing vulnerability) => "convey_emotional_support_need"
  - "release_pent_up_emotions" is for cathartic expression; "convey_emotional_support_need" is for seeking comfort/help
  - merely seeking informational advice does NOT qualify as "convey_emotional_support_need"
- creating connection by sharing relatable experiences to bond with others" => "convey_similar_experience" if there is a clear signal in the response suggesting this intent
  - do not assume "convey_similar_experience" based on outside knowledge
- seeking informational (as opposed to emotional) support/advice, speculation => out of scope, so return empty list if no other labels apply
- **focus on prominent intent(s); secondary purposes should only be included if relatively substantial**
- when correcting misinformation or explaining events, consider whether the primary intent is neutral explanation (clarify_what_transpired) or taking a stance to persuade/challenge (justify_challenge_offer_belief_norm). Both can apply when someone explains facts AND argues a position."""

EMOTIONAL_RESPONSE_TIPS = f"""- amusement => “joy”, “appreciation”
- validation and agreement => “relief”, “pride”, “connection”
- nostalgia => “sadness”, “joy”, “appreciation”, “connection”
- satisfaction => “pride” (could sometimes also point to “joy”, “relief”, and/or “appreciation” depending on context)
- hopelessness => “fear”, “sadness”
- self-consciousness about one’s appearance => “fear”, “disgust”
- frustration or exasperation => “anger”
- if the anger/exasperation/disbelief is visceral or especially intense AND directed toward a misbehaving third party whose behavior is explicitly/implicitly considered very offensive => also add "disgust"
- concerned, anxious, defensive => “fear”
- cautious => usually “fear” but sometimes *nothing* (e.g., “cautious skepticism” is nothing)
- regret => “sadness”
- conflicted, embarrassed => depends on context, but often “guilt”
- confidence => usually “hope” but could be “pride” in some contexts
- empathy => “connection”, often “compassion”, and whatever target emotion is empathized with (e.g., “sadness” for “empathy for their loss”)
- awe => “appreciation” (could also point to “fear” in some contexts)
- curiosity or skepticism => *nothing* (these are just cognitive states, not emotions)
- concern for others or duty to care for others => "compassion"
  - compassion alone does not imply connection
- appreciating someone else (who is proud) does not imply that oneself is proud
- excitement => often “hope”, “joy”."""

AESTHETIC_FEELING_TIPS = f"""- empathy, skepticism, concern, admiration, frustration, relatability, exasperation, satisfaction, reassurance, validation, disappointment, compassion, discomfort => “other” if there is not a strong signal for one of our provided labels (these are other kinds of feelings not covered by our set of aesthetic feeling labels)
- interest in finding connections between events / piecing them together => “curiosity”
- if something grabs or holds focus => “attention_engagement”
- nostalgia, vivid => “evocation”
- tension, anxiety, fear about future event => “suspense”
- being pulled into, drawn into, absorbed, immersed in a story; visceral, secondhand feelings (e.g. secondhand embarrassment) => “transportation”
- if not accompanied by another label, put ** (we don’t count empathy as an aesthetic feeling in and of itself).
- if you’ve already found one label, don’t feel the need to put ‘other’ to cover other aspects of the same response"""

PREDICTION_TIPS = """- subject identification
  - prediction about narrator => “narr_*”
  - prediction about character besides narrator => “other_char_*”
  - prediction about non-character entity => “non_char_thing_*”
- action/event vs. state
  - active behavior / doing something => action/event
  - passive condition / being in a situation / feeling a certain way => state
  - there is a spectrum between states and actions, with many predictions falling somewhere in the middle. When in doubt, apply both labels (e.g., “narr_future_state” and “narr_future_action”)
  - if the prediction is about someone “continuing”, your label should be determined by the expected length of the continuation
    - continuing to argue/justify/make a point in this specific conversation => action
    - continuing a behavior indefinitely => action and state
    - continuing to feel a certain way or maintain a condition => state
- focus on the main point — if the prediction is about a future action, say action (even if some state must implicitly motivate that action)
- if a character is described as passively receiving something / something happening to them, focus on the action of the other character offering / doing something."""

CAUSAL_EXPLANATION_TIPS = """distinguishing characters from things:
- *character* (narr or other_char): intentional/conscious actions, emotional reactions, behaviors, and mental states are associated with characters.
  - in addition to individuals, any group (e.g., family), animal, or company whose agency is foregrounded counts as a characters
- *thing*: cultural/institutional force (e.g., religious doctrine), systems, non-conscious processes, non-conscious body parts, objects.
  - underlying somatic or environmental factors, such as an undiagnosed medical conditon or instincts, that influence characters’ mental states or actions
  - cultural artifacts (e.g., books, films, video games): treat are considered things UNLESS creator agency is explicitly foregrounded
- special case: if a character’s behavior is described as enacting or being influenced by a social norm or cultural institution (e.g. religious doctrine), we consider that BOTH a character's action and a thing.

distinguishing narrator from either other character or things:
- narr: Text explicitly presents narrator's reasoning/beliefs/mental processes
- other_char_or_thing: Reasoning attributed to non-narrator character or systemic factors
- KEY: Don't use narrator labels when narrator simply reports facts about others

general tips:
- check for multiple explanation types. Use multiple labels when more than one character or thing is being explained, or when multiple characters/things are doing the explaining.
- if someone or something is explained (partially or fully) by a character’s belief, perception, or opinion, use:
  - “*_explained_by_narr” if the narrator holds the belief
  - “*_explained_by_other_char_or_thing” if another character holds the belief
  - if the belief explicitly stems from a cultural or institutional source (e.g., religious doctrine) or social norm, also include “*_explained_by_other_char_or_thing”
  - if the belief is a proposition (e.g., “narrator believes President is angry”), label based on the belief’s content as well, if that content isn’t already captured. For example: “_explained_by_narr” + “_explained_by_other_char_or_thing”
- when the narrator makes a neutral observation, comment, or mention, label based on what is described, not the act of commenting UNLESS the explanation itself is about emphasizes why/how the narrator made the comment or took the action.
- if the narrator emphasizes, argues, or believes something, include labels for both the action (e.g., arguing) and the content or subject of that action.
- other_char_or_things (e.g., other char actions, non-conscious bodily processes in the narrator like illnesses, aesthetic qualities of comments) attributed (at least partly) to conscious actions, perceived beliefs, or expected reactions of the narrator => "other_char_or_thing_explained_by_narr"
- edge case: If it's unclear whether to label based on the character experiencing something or the one causing it, label the more active party. Focus on explaining their behavior, not the recipient’s experience.
- edge case: If it's ambiguous whether a character is the narrator or another character, assume they are not the narrator. Label as another character."""

MORAL_TIPS = """general tips:
- select labels based on thematic relevance, even if the text is not fully endorsing the value (e.g., if the text highlights a tradeoff between multiple values)

label-specific tips:
- independent thinking, critical evaluation to form one's own opinion, effortful researching to make informed personal choices, creative problem-solving
- embracing challenges and change FOR THEIR OWN SAKE, risk-taking, excitement and adventure, unpredictability/adaptability, valuing novelty and difficulty as inherently rewarding => “stimulation”
- pleasure, enjoyment, fun, humor, entertainment, appearance => “hedonism”
- demonstrated individual competence, measurable goal attainment and success, skill development leading to improved performance, professional or competitive success with clear outcomes (focus on results and competence, not just effort or persistence) => “achievement”
- authority, institutional influence/control, status and dominance dynamics, organizational hierarchy issues, obedience, submission => “power”
- safety and risk management for SIGNIFICANT threats, relationship survival, safeguarding financial or health status from substantial harm, maintaining equanimity against serious risks => “security”
- following communication and social interaction norms, anticipating and trying to avoid misunderstandings => “conformity”
- respecting the specific customs and practices according to cultural consensus or a large religious or state institution, respecting or learning from the past, trusting conventional media/institutions (e.g. news, libraries) => “tradition”
- caring for family, friends, teammates; loyalty and commitment to specific groups, helping those in close proximity => “benevolence”
- fairness and equality, appreciating/celebrating differences or variation across individuals or groups, avoiding discrimination or prejudice based on assumptions about unknown others, broad social welfare concerns => “universalism”

edge cases:
- if a text highlights awareness of variation across traditions, label “universalism”, not “tradition”—which should be used when a text focuses on a single tradition (either positively or critically).
- referring to systemic things or to the fact that individuals are explained by social factors does not automatically imply “universalism”
- posts about pleasure vs. practicality (e.g., choosing practical/reliable option over flashier option) can still merit “hedonism”, even if pleasure-seeking isn’t explicitly endorsed.
- minor miscommunications, everyday foibles, or small issues with tools/resources do not warrant a “security” label unless there’s a clear threat to well-being.
- critiques/descriptions of large institutions: 
  - use “power” to reflect dynamics of control or abuse.
  - add “security” if societal risks are emphasized.
  - add “achievement” if poor competence, mismanagement, or flawed strategy is central.
- strategy, planning, and persistence are usually "achievement" (if goal-oriented) or "self-direction" (if analytical), NOT "stimulation" unless the challenge itself is valued as rewarding
- research and verification activities are "self-direction" when about critical thinking; "universalism" when about avoiding assumptions about unknown others
- KEY: if in doubt about whether to include a label that is not clearly implied or covered by the definitions or guidelines, leave it out."""

STANCE_TIPS = """answer based on the keywords in the beginning of the statement:
  - support => “support_belief_norm”
  - counter => “counter_belief_norm”
  - be neutral to => “neutral_belief_norm”"""

CHARACTER_APPRAISAL_TIPS = """- determine the sentiment of appraisal based on the keywords at the beginning of the statement
  - positively => “positive_appraisal_*”
  - negatively => “negative_appraisal_*”
  - neutrally => “neutral_appraisal_*”
- general tips
  - if it is ambiguous whether the character being judged is the narrator or another character, assume it is another character"""

TAXONOMY_CLASSIFICATION_TIPS_DICT = {
  "overall_goal": OVERALL_GOAL_TIPS,
  "narrative_intent": NARRATIVE_INTENT_TIPS,
  "author_emotional_response": EMOTIONAL_RESPONSE_TIPS,
  "narrative_feeling": EMOTIONAL_RESPONSE_TIPS,
  "aesthetic_feeling": AESTHETIC_FEELING_TIPS,
  "stance": STANCE_TIPS,
  "prediction": PREDICTION_TIPS,
  "causal_explanation": CAUSAL_EXPLANATION_TIPS,
  "moral": MORAL_TIPS,
  "character_appraisal": CHARACTER_APPRAISAL_TIPS,
}

GPT4O_INF_PLAUSIBILITY_RATINGS_PATH = "../data/replication/annotations/prolific/gpt4o_inf_plausibility_ratings.tsv"
SSF_GEN_INF_PLAUSIBILITY_RATINGS_PATH = "../data/replication/annotations/prolific/ssf_generator_inf_plausibility_ratings.tsv"
HUMAN_WRITTEN_INFS_PATH = "../data/replication/annotations/prolific/human_written_infs.tsv"
CONSENT_ID = 'Contextual-Reasoning-about-Narrative-Intents-and-Reactions-consent.html'

SSF_CORPUS_HF = "joelmire/ssf-corpus"