"""Utility functions for Hugging Face auto-models."""

from coral.models.pretrained import load_pretrained_models
from coral.models.oa_model import AutoModelForOA, OAModelOutput


__all__ = ['load_pretrained_models', 'AutoModelForOA', 'OAModelOutput']
