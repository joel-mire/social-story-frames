from abc import ABC, abstractmethod

class CommunityPreprocessor(ABC):
  def __init__(self):
    pass
  
  @abstractmethod
  def get_community_allowlist(self):
    pass

  @abstractmethod
  def get_community_data_dict(self):
    pass