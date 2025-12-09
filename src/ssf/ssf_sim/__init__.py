"""
SSF-Sim: Similarity computation for Story Sharing Framework.

This package provides tools for computing SSF-Sim (Story Sharing Framework Similarity)
between stories and communities using both classification-based and generation-based
approaches.

Main classes:
    - CommunitySsfSim: Compute similarity between communities (aggregated from multiple stories)
    - StorySsfSim: Compute similarity between individual story pairs

Helper utilities:
    - data_utils: Functions to prepare data from DataFrames
"""

from ssf.ssf_sim.community import CommunitySsfSim
from ssf.ssf_sim.story import StorySsfSim

__all__ = [
    'CommunitySsfSim',
    'StorySsfSim',
]
