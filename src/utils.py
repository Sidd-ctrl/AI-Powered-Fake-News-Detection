"""
Utility functions for Fake News Detection Project.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Create realistic synthetic fake news dataset."""
    np.random.seed(random_state)
    
    real_templates = [
        "Scientists have discovered a new species of butterfly in the Amazon rainforest. The findings were published in Nature journal today.",
        "The government announced new economic policies aimed at reducing inflation. The measures include tax reforms and infrastructure investments.",
        "A major breakthrough in cancer research was announced by researchers at Stanford University. Clinical trials show promising results.",
        "The United Nations held its annual climate summit this week. World leaders agreed on new emission reduction targets.",
        "Local elections will be held next month across multiple states. Candidates from various parties are campaigning actively.",
        "Tech companies reported strong quarterly earnings driven by cloud computing growth. Stock prices rose significantly.",
        "A new vaccine for malaria has been approved by health authorities following successful trials in Africa.",
        "Archaeologists uncovered ancient artifacts dating back 5000 years in a remote excavation site.",
        "The stock market reached an all-time high today amid positive economic indicators and strong corporate performance.",
        "Researchers developed a new method for water purification that could help millions in developing countries.",
    ]
    
    fake_templates = [
        "Aliens have landed in New York City according to multiple eyewitness reports. Government officials are denying everything.",
        "Miracle cure discovered that can treat all diseases including cancer and diabetes. Big pharma is trying to suppress it.",
        "The moon landing was faked according to new evidence leaked by a former NASA employee. Photos show obvious studio lighting.",
        "Bill Gates is planning to microchip the entire population through vaccines. This is part of a global conspiracy.",
        "A secret society controls all world governments and media. They have been hiding this fact for decades.",
        "Drinking bleach can cure COVID-19 according to a viral video from a self-proclaimed health expert.",
        "The earth is flat and all space agencies are lying to the public. Multiple whistleblowers have come forward.",
        "Celebrity X was arrested for serious crimes but mainstream media is covering it up. Share this before it gets deleted.",
        "5G technology is causing COVID-19 and other illnesses according to scientific studies that were suppressed.",
        "The election was stolen using advanced technology that changed millions of votes. Proof is available if you look carefully.",
    ]
    
    data = []
    for i in range(n_samples):
        if np.random.rand() > 0.5:
            template = np.random.choice(real_templates)
            title = template.split('.')[0] + "."
            text = template + " " + " ".join(np.random.choice(real_templates, 2))
            label = 1
        else:
            template = np.random.choice(fake_templates)
            title = template.split('.')[0] + "."
            text = template + " " + " ".join(np.random.choice(fake_templates, 1))
            label = 0
        
        data.append({
            'id': i,
            'title': title,
            'author': f'Author_{np.random.randint(1, 100)}',
            'text': text,
            'label': label
        })
    
    df = pd.DataFrame(data)
    logger.info(f"Created synthetic dataset with {len(df)} samples")
    return df


def load_dataset(dataset_path: str = None, synthetic: bool = True, n_samples: int = 2000) -> pd.DataFrame:
    """Load dataset from file or generate synthetic data."""
    try:
        if dataset_path and os.path.exists(dataset_path):
            logger.info(f"Loading dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            if 'text' not in df.columns and 'title' in df.columns:
                df['text'] = df['title']
            if 'label' not in df.columns:
                raise ValueError("Dataset must contain 'label' column")
        else:
            if synthetic:
                logger.info("Generating synthetic dataset")
                df = create_synthetic_dataset(n_samples=n_samples)
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)