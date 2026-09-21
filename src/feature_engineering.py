"""
Feature Engineering Module for Fake News Detection.
Implements Bag of Words and TF-IDF.
"""

import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature extraction supporting BoW and TF-IDF."""
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.feature_names = None
        self.method = None
        logger.info(f"FeatureEngineer initialized: max_features={max_features}")
    
    def fit_tfidf(self, texts: pd.Series) -> np.ndarray:
        """Fit TF-IDF vectorizer."""
        logger.info("Fitting TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.method = 'tfidf'
        logger.info(f"TF-IDF vocabulary size: {len(self.feature_names)}")
        return X.toarray()
    
    def fit_bow(self, texts: pd.Series) -> np.ndarray:
        """Fit Bag of Words vectorizer."""
        logger.info("Fitting Bag of Words vectorizer...")
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.method = 'bow'
        logger.info(f"BoW vocabulary size: {len(self.feature_names)}")
        return X.toarray()
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Return feature information."""
        if self.vectorizer is None:
            return {}
        return {
            'method': self.method,
            'vocabulary_size': len(self.feature_names),
            'feature_dimensions': self.max_features,
            'ngram_range': self.ngram_range
        }