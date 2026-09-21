"""
Model Evaluation Module for Fake News Detection.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8-whitegrid')


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['Fake', 'Real']
        self.results: Dict[str, Dict] = {}
    
    def evaluate_model(self, name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate single model."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0)
        }
        self.results[name] = metrics
        return metrics
    
    def evaluate_all(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Evaluate all models."""
        comparison_data = []
        for name, y_pred in predictions.items():
            metrics = self.evaluate_model(name, y_true, y_pred)
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        df = pd.DataFrame(comparison_data).sort_values('F1-Score', ascending=False).reset_index(drop=True)
        return df
    
    def plot_class_distribution(self, y: np.ndarray, save_path: str = None):
        """Plot class distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))
        unique, counts = np.unique(y, return_counts=True)
        labels = [self.class_names[i] for i in unique]
        bars = ax.bar(labels, counts, color=['#e74c3c', '#2ecc71'], edgecolor='black')
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(count), ha='center', fontsize=12, fontweight='bold')
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Samples')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, metric: str = 'F1-Score', save_path: str = None):
        """Plot model comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors[:len(comparison_df)], edgecolor='black')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(comparison_df[metric]) * 1.15)
        ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric)
        plt.xticks(rotation=15)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, name: str, save_path: str = None):
        """Plot confusion matrix."""
        if name not in self.results:
            return
        cm = self.results[name]['confusion_matrix']
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()