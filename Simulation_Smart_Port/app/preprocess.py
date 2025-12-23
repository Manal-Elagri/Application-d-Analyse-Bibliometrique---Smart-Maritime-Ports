"""
Module de préprocessing et nettoyage des données textuelles
"""

import re
import string
import pandas as pd
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from unidecode import unidecode
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Téléchargement des ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Stopwords en anglais
STOPWORDS_EN = set(stopwords.words('english'))
# Stopwords supplémentaires spécifiques aux publications scientifiques
SCIENTIFIC_STOPWORDS = {
    'abstract', 'article', 'paper', 'study', 'research', 'method', 'methods',
    'result', 'results', 'conclusion', 'conclusions', 'introduction',
    'background', 'objective', 'objectives', 'purpose', 'aim', 'aims',
    'however', 'therefore', 'furthermore', 'moreover', 'additionally',
    'et al', 'et', 'al', 'fig', 'figure', 'table', 'ref', 'reference'
}
STOPWORDS = STOPWORDS_EN.union(SCIENTIFIC_STOPWORDS)


def normalize_text(text: str) -> str:
    """
    Normalise un texte : lowercase, suppression caractères spéciaux, etc.
    
    Args:
        text: Texte à normaliser
    
    Returns:
        Texte normalisé
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Conversion en minuscules
    text = text.lower()
    
    # Normalisation Unicode (suppression des accents)
    text = unidecode(text)
    
    # Suppression des caractères spéciaux sauf espaces et tirets
    text = re.sub(r'[^\w\s-]', ' ', text)
    
    # Normalisation des espaces multiples
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def remove_stopwords(text: str, custom_stopwords: Optional[set] = None) -> str:
    """
    Supprime les stopwords d'un texte.
    
    Args:
        text: Texte à traiter
        custom_stopwords: Ensemble de stopwords personnalisés (optionnel)
    
    Returns:
        Texte sans stopwords
    """
    if not text:
        return ""
    
    stopwords_set = custom_stopwords if custom_stopwords else STOPWORDS
    
    try:
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords_set]
        return ' '.join(filtered_tokens)
    except Exception as e:
        logger.warning(f"Erreur lors de la suppression des stopwords: {e}")
        return text


def clean_abstract(text: str) -> str:
    """
    Nettoie un abstract : normalisation + suppression stopwords.
    
    Args:
        text: Abstract à nettoyer
    
    Returns:
        Abstract nettoyé
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Normalisation
    text = normalize_text(text)
    
    # Suppression des stopwords
    text = remove_stopwords(text)
    
    return text


def clean_keywords(keywords: str) -> List[str]:
    """
    Nettoie et tokenise les mots-clés.
    
    Args:
        keywords: Chaîne de mots-clés séparés par ';' ou ','
    
    Returns:
        Liste de mots-clés nettoyés
    """
    if pd.isna(keywords) or not isinstance(keywords, str):
        return []
    
    # Séparation par ';' ou ','
    keyword_list = re.split(r'[;,]', keywords)
    
    # Nettoyage de chaque mot-clé
    cleaned_keywords = []
    for kw in keyword_list:
        kw = normalize_text(kw)
        if kw and len(kw) > 2:  # Ignorer les mots trop courts
            cleaned_keywords.append(kw)
    
    return cleaned_keywords


def clean_authors(authors: str) -> List[str]:
    """
    Nettoie et parse la liste des auteurs.
    
    Args:
        authors: Chaîne d'auteurs séparés par ';'
    
    Returns:
        Liste d'auteurs nettoyés
    """
    if pd.isna(authors) or not isinstance(authors, str):
        return []
    
    # Séparation par ';'
    author_list = re.split(r'[;]', authors)
    
    # Nettoyage
    cleaned_authors = []
    for author in author_list:
        author = author.strip()
        if author:
            cleaned_authors.append(author)
    
    return cleaned_authors


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prétraite un DataFrame complet : nettoyage de toutes les colonnes textuelles.
    
    Args:
        df: DataFrame à prétraiter
    
    Returns:
        DataFrame prétraité
    """
    logger.info("Début du préprocessing...")
    df_processed = df.copy()
    
    # Nettoyage des abstracts
    if 'Abstract' in df_processed.columns:
        logger.info("Nettoyage des abstracts...")
        df_processed['Abstract_cleaned'] = df_processed['Abstract'].apply(clean_abstract)
    
    # Nettoyage des titres
    if 'Title' in df_processed.columns:
        logger.info("Nettoyage des titres...")
        df_processed['Title_cleaned'] = df_processed['Title'].apply(normalize_text)
    
    # Nettoyage des mots-clés
    if 'Author Keywords' in df_processed.columns:
        logger.info("Nettoyage des mots-clés...")
        df_processed['Keywords_list'] = df_processed['Author Keywords'].apply(clean_keywords)
    
    # Nettoyage des auteurs
    if 'Authors' in df_processed.columns:
        logger.info("Nettoyage des auteurs...")
        df_processed['Authors_list'] = df_processed['Authors'].apply(clean_authors)
    
    # Conversion de l'année en numérique
    if 'Year' in df_processed.columns:
        logger.info("Conversion des années...")
        df_processed['Year'] = pd.to_numeric(df_processed['Year'], errors='coerce')
        df_processed = df_processed.dropna(subset=['Year'])  # Supprime les lignes sans année
    
    logger.info("Préprocessing terminé")
    
    return df_processed

