"""
Machine Learning Models Module for Fake News Detection.
"""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Model training class with 4 algorithms."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, BaseEstimator] = {}
        self.trained_models: Dict[str, BaseEstimator] = {}
        self._initialize_models()
        logger.info("ModelTrainer initialized with 4 models")
    
    def _initialize_models(self) -> None:
        """Initialize models with chosen hyperparameters."""
        self.models = {
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='cosine',
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                solver='lbfgs',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'NeuralNet': MLPClassifier(
                hidden_layer_sizes=(100,),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                max_iter=300,
                random_state=self.random_state,
                early_stopping=True
            )
        }
    
    def get_model_descriptions(self) -> Dict[str, str]:
        """Return model descriptions."""
        return {
            'KNN': 'K-Nearest Neighbors: Instance-based learning using similarity.',
            'LogisticRegression': 'Logistic Regression: Linear probabilistic classifier.',
            'RandomForest': 'Random Forest: Ensemble of decision trees.',
            'NeuralNet': 'Multi-Layer Perceptron: Simple neural network.'
        }
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, BaseEstimator]:
        """Train all models."""
        logger.info(f"Training {len(self.models)} models...")
        self.trained_models = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            logger.info(f"{name} training complete")
        return self.trained_models
    
    def predict(self, name: str, X_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if name not in self.trained_models:
            raise ValueError(f"Model {name} not trained.")
        return self.trained_models[name].predict(X_test)