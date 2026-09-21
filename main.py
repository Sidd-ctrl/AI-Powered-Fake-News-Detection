"""
Main execution script - Complete Fake News Detection Pipeline
"""
import os
import logging
import warnings
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('outputs/images', exist_ok=True)
os.makedirs('outputs/models', exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

def create_synthetic_dataset(n_samples=2000):
    np.random.seed(42)
    real_templates = [
        "Scientists have discovered a new species of butterfly in the Amazon rainforest.",
        "The government announced new economic policies aimed at reducing inflation.",
        "A major breakthrough in cancer research was announced by Stanford University.",
        "The United Nations held its annual climate summit this week.",
        "Local elections will be held next month across multiple states.",
        "Tech companies reported strong quarterly earnings driven by cloud computing.",
        "A new vaccine for malaria has been approved by health authorities.",
        "Archaeologists uncovered ancient artifacts dating back 5000 years.",
        "The stock market reached an all-time high today amid positive indicators.",
        "Researchers developed a new method for water purification."
    ]
    fake_templates = [
        "Aliens have landed in New York City according to eyewitness reports.",
        "Miracle cure discovered that can treat all diseases including cancer.",
        "The moon landing was faked according to new evidence leaked by NASA.",
        "Bill Gates is planning to microchip the entire population through vaccines.",
        "A secret society controls all world governments and media.",
        "Drinking bleach can cure COVID-19 according to a viral video.",
        "The earth is flat and all space agencies are lying to the public.",
        "Celebrity X was arrested for serious crimes but media is covering it up.",
        "5G technology is causing COVID-19 according to suppressed studies.",
        "The election was stolen using advanced technology."
    ]
    
    data = []
    for i in range(n_samples):
        if np.random.rand() > 0.5:
            text = np.random.choice(real_templates) + " " + " ".join(np.random.choice(real_templates, 2))
            label = 1
        else:
            text = np.random.choice(fake_templates) + " " + " ".join(np.random.choice(fake_templates, 1))
            label = 0
        data.append({'text': text, 'label': label})
    return pd.DataFrame(data)

def main():
    logger.info("="*70)
    logger.info("AI-POWERED FAKE NEWS DETECTION PIPELINE")
    logger.info("="*70)
    
    # 1. Load Data
    logger.info("[1/8] Loading dataset...")
    df = create_synthetic_dataset(2000)
    logger.info(f"Dataset loaded: {len(df)} samples")
    
    # 2. Preprocessing
    logger.info("[2/8] Preprocessing text...")
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        tokens = text.split()
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        tokens = [stemmer.stem(w) for w in tokens]
        return ' '.join(tokens)
    
    df['clean_text'] = df['text'].apply(clean_text)
    logger.info("Preprocessing complete")
    
    # 3. Feature Engineering
    logger.info("[3/8] Feature Engineering (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(df['clean_text']).toarray()
    y = df['label'].values
    logger.info(f"Feature matrix shape: {X.shape}")
    
    # 4. Train-Test Split
    logger.info("[4/8] Train-Test Split (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 5. Models
    logger.info("[5/8] Training models...")
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42),
        'NeuralNet': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, early_stopping=True)
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"  Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1, 'CM': confusion_matrix(y_test, preds)}
        logger.info(f"    {name}: Acc={acc:.4f}, F1={f1:.4f}")
    
    # 6. Comparison
    comparison_df = pd.DataFrame([{'Model': name, **{k: v for k, v in res.items() if k != 'CM'}} for name, res in results.items()]).sort_values('F1-Score', ascending=False)
    print("\n" + "="*55)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*55)
    print(comparison_df.round(4).to_string(index=False))
    
    best_model = comparison_df.iloc[0]['Model']
    
    # 7. Visualizations
    logger.info("[7/8] Creating visualizations...")
    img_dir = 'outputs/images'
    
    # Class Distribution
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Fake', 'Real']
    counts = [sum(y == 0), sum(y == 1)]
    bars = ax.bar(labels, counts, color=['#e74c3c', '#27ae60'], edgecolor='black')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(c), ha='center', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{img_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # F1 Comparison
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    bars = ax.bar(comparison_df['Model'], comparison_df['F1-Score'], color=colors, edgecolor='black')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{img_dir}/f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrices
    for name in results:
        fig, ax = plt.subplots(figsize=(5, 4))
        cm = results[name]['CM']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake','Real'], yticklabels=['Fake','Real'], ax=ax)
        ax.set_title(f'Confusion Matrix - {name}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{img_dir}/cm_{name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 8. Save
    logger.info("[8/8] Saving best model...")
    joblib.dump(models[best_model], 'outputs/models/best_model.pkl')
    joblib.dump(vectorizer, 'outputs/models/vectorizer.pkl')
    comparison_df.to_csv('outputs/model_comparison.csv', index=False)
    
    logger.info("="*70)
    logger.info(f"PIPELINE COMPLETED | Best Model: {best_model} | F1: {comparison_df.iloc[0]['F1-Score']:.4f}")
    logger.info("="*70)

if __name__ == "__main__":
    main()