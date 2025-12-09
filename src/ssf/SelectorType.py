from enum import Enum

class SelectorType(Enum):
    ALL = 'ALL'
    STORIES_SUPERSET = 'STORIES_SUPERSET'
    STORIES = 'STORIES'
    STORIES_AND_CONTEXTS = 'STORIES_AND_CONTEXTS'