"""
Fake News Detection - Source Package

This package contains all core modules for the AI-Powered Fake News Detection project.

Modules:
    - utils: Dataset loading and helper functions
    - preprocessing: Text cleaning pipeline
    - feature_engineering: BoW and TF-IDF vectorization
    - models: Machine learning model training
    - evaluation: Model evaluation and visualization

Usage:
    from src.preprocessing import TextPreprocessor
    from src.feature_engineering import FeatureEngineer
    from src.models import ModelTrainer
    from src.evaluation import ModelEvaluator
"""

from .utils import load_dataset, create_synthetic_dataset, ensure_dir
from .preprocessing import TextPreprocessor
from .feature_engineering import FeatureEngineer
from .models import ModelTrainer
from .evaluation import ModelEvaluator

__version__ = "1.0.0"
__author__ = "AI & ML Internship Project"
__all__ = [
    "load_dataset",
    "create_synthetic_dataset",
    "ensure_dir",
    "TextPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator"
]