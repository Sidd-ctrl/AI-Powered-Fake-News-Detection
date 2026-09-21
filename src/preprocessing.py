"""
Text Preprocessing Module for Fake News Detection.
Implements complete NLP preprocessing pipeline from scratch.
"""

import re
import string
import logging
from typing import List
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Complete text preprocessing pipeline for NLP tasks.
    
    Steps implemented:
    1. Lowercasing
    2. Punctuation removal
    3. Special character removal
    4. Number removal
    5. Stopword removal
    6. Tokenization
    7. Stemming
    """
    
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        logger.info(f"TextPreprocessor initialized (stemming={use_stemming})")
    
    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_special_characters(self, text: str) -> str:
        """Remove special characters."""
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    def remove_numbers(self, text: str) -> str:
        """Remove numeric characters."""
        return re.sub(r'\d+', '', text)
    
    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        return ' '.join(text.split())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove common English stopwords."""
        return [word for word in tokens if word not in self.stop_words and len(word) > 2]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming."""
        return [self.stemmer.stem(word) for word in tokens]
    
    def clean_text(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline.
        
        Steps:
        1. Lowercase
        2. Remove punctuation
        3. Remove special characters
        4. Remove numbers
        5. Clean whitespace
        6. Tokenize
        7. Remove stopwords
        8. Apply stemming
        9. Join tokens
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Step 1: Lowercase
        text = self.to_lowercase(text)
        
        # Step 2: Remove punctuation
        text = self.remove_punctuation(text)
        
        # Step 3: Remove special characters
        text = self.remove_special_characters(text)
        
        # Step 4: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 5: Clean whitespace
        text = self.remove_extra_whitespace(text)
        
        # Step 6: Tokenize
        tokens = self.tokenize(text)
        
        # Step 7: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 8: Stemming
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        
        # Step 9: Join tokens
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Apply preprocessing to entire DataFrame."""
        logger.info(f"Preprocessing {len(df)} texts...")
        df['clean_text'] = df[text_column].apply(self.clean_text)
        df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
        logger.info(f"Preprocessing complete. Final size: {len(df)}")
        return df