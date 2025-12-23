"""
Module d'analyse bibliométrique
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def annual_evolution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'évolution annuelle des publications.
    
    Args:
        df: DataFrame avec colonne 'Year'
    
    Returns:
        DataFrame avec colonnes 'Year' et 'Count'
    """
    if 'Year' not in df.columns:
        logger.warning("Colonne 'Year' absente")
        return pd.DataFrame()
    
    annual_counts = df['Year'].value_counts().sort_index()
    result = pd.DataFrame({
        'Year': annual_counts.index,
        'Count': annual_counts.values
    })
    
    logger.info(f"Évolution annuelle calculée: {len(result)} années")
    return result


def top_authors(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Identifie les top auteurs.
    
    Args:
        df: DataFrame avec colonne 'Authors_list' (liste d'auteurs)
        top_n: Nombre d'auteurs à retourner
    
    Returns:
        DataFrame avec colonnes 'Author' et 'Count'
    """
    if 'Authors_list' not in df.columns:
        logger.warning("Colonne 'Authors_list' absente")
        return pd.DataFrame()
    
    # Aplatir la liste des auteurs
    all_authors = []
    for authors_list in df['Authors_list']:
        if isinstance(authors_list, list):
            all_authors.extend(authors_list)
    
    if not all_authors:
        return pd.DataFrame()
    
    # Compter les occurrences
    author_counts = Counter(all_authors)
    
    # Créer le DataFrame
    top_authors_df = pd.DataFrame(
        author_counts.most_common(top_n),
        columns=['Author', 'Count']
    )
    
    logger.info(f"Top {len(top_authors_df)} auteurs identifiés")
    return top_authors_df


def top_journals(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Identifie les top journaux.
    
    Args:
        df: DataFrame avec colonne 'Source title'
        top_n: Nombre de journaux à retourner
    
    Returns:
        DataFrame avec colonnes 'Journal' et 'Count'
    """
    if 'Source title' not in df.columns:
        logger.warning("Colonne 'Source title' absente")
        return pd.DataFrame()
    
    journal_counts = df['Source title'].value_counts().head(top_n)
    result = pd.DataFrame({
        'Journal': journal_counts.index,
        'Count': journal_counts.values
    })
    
    logger.info(f"Top {len(result)} journaux identifiés")
    return result


def top_keywords(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Identifie les top mots-clés.
    
    Args:
        df: DataFrame avec colonne 'Keywords_list' (liste de mots-clés)
        top_n: Nombre de mots-clés à retourner
    
    Returns:
        DataFrame avec colonnes 'Keyword' et 'Count'
    """
    if 'Keywords_list' not in df.columns:
        logger.warning("Colonne 'Keywords_list' absente")
        return pd.DataFrame()
    
    # Aplatir la liste des mots-clés
    all_keywords = []
    for keywords_list in df['Keywords_list']:
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)
    
    if not all_keywords:
        return pd.DataFrame()
    
    # Compter les occurrences
    keyword_counts = Counter(all_keywords)
    
    # Créer le DataFrame
    top_keywords_df = pd.DataFrame(
        keyword_counts.most_common(top_n),
        columns=['Keyword', 'Count']
    )
    
    logger.info(f"Top {len(top_keywords_df)} mots-clés identifiés")
    return top_keywords_df


def co_occurrence_matrix(df: pd.DataFrame, min_freq: int = 2) -> pd.DataFrame:
    """
    Calcule la matrice de co-occurrence des mots-clés.
    
    Args:
        df: DataFrame avec colonne 'Keywords_list'
        min_freq: Fréquence minimale d'un mot-clé pour être inclus
    
    Returns:
        DataFrame avec matrice de co-occurrence
    """
    if 'Keywords_list' not in df.columns:
        logger.warning("Colonne 'Keywords_list' absente")
        return pd.DataFrame()
    
    # Compter les fréquences des mots-clés
    all_keywords = []
    for keywords_list in df['Keywords_list']:
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)
    
    keyword_counts = Counter(all_keywords)
    
    # Filtrer par fréquence minimale
    frequent_keywords = {kw: count for kw, count in keyword_counts.items() if count >= min_freq}
    
    if not frequent_keywords:
        return pd.DataFrame()
    
    # Créer la matrice de co-occurrence
    co_occurrence = {}
    
    for keywords_list in df['Keywords_list']:
        if isinstance(keywords_list, list):
            # Filtrer les mots-clés fréquents
            filtered_kw = [kw for kw in keywords_list if kw in frequent_keywords]
            
            # Calculer les co-occurrences
            for i, kw1 in enumerate(filtered_kw):
                for kw2 in filtered_kw[i+1:]:
                    pair = tuple(sorted([kw1, kw2]))
                    co_occurrence[pair] = co_occurrence.get(pair, 0) + 1
    
    if not co_occurrence:
        return pd.DataFrame()
    
    # Convertir en DataFrame
    co_occurrence_list = [
        {'Keyword1': pair[0], 'Keyword2': pair[1], 'Co_occurrence': count}
        for pair, count in co_occurrence.items()
    ]
    
    result = pd.DataFrame(co_occurrence_list)
    result = result.sort_values('Co_occurrence', ascending=False)
    
    logger.info(f"Matrice de co-occurrence créée: {len(result)} paires")
    return result


def get_statistics(df: pd.DataFrame) -> Dict:
    """
    Calcule des statistiques générales sur le dataset.
    
    Args:
        df: DataFrame à analyser
    
    Returns:
        Dictionnaire avec les statistiques
    """
    stats = {
        'total_publications': len(df),
        'years_span': None,
        'unique_authors': 0,
        'unique_journals': 0,
        'unique_keywords': 0
    }
    
    if 'Year' in df.columns:
        years = df['Year'].dropna()
        if len(years) > 0:
            stats['years_span'] = f"{int(years.min())} - {int(years.max())}"
    
    if 'Authors_list' in df.columns:
        all_authors = set()
        for authors_list in df['Authors_list']:
            if isinstance(authors_list, list):
                all_authors.update(authors_list)
        stats['unique_authors'] = len(all_authors)
    
    if 'Source title' in df.columns:
        stats['unique_journals'] = df['Source title'].nunique()
    
    if 'Keywords_list' in df.columns:
        all_keywords = set()
        for keywords_list in df['Keywords_list']:
            if isinstance(keywords_list, list):
                all_keywords.update(keywords_list)
        stats['unique_keywords'] = len(all_keywords)
    
    return stats

