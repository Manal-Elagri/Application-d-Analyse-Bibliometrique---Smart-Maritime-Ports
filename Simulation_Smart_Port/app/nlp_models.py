"""
Module pour le clustering thématique et LDA Topic Modeling
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
from gensim.models import LdaModel
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def tfidf_clustering(
    texts: List[str],
    n_clusters: int = 5,
    max_features: int = 1000,
    min_df: int = 2,
    max_df: float = 0.95
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Effectue un clustering KMeans sur des textes avec TF-IDF.
    
    Args:
        texts: Liste de textes à clusteriser
        n_clusters: Nombre de clusters
        max_features: Nombre maximum de features TF-IDF
        min_df: Fréquence minimale d'un terme
        max_df: Fréquence maximale d'un terme
    
    Returns:
        Tuple (labels, vectorizer)
    """
    if not texts:
        logger.warning("Liste de textes vide")
        return np.array([]), None
    
    logger.info(f"Clustering TF-IDF avec {n_clusters} clusters...")
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words='english',
        ngram_range=(1, 2)  # Unigrammes et bigrammes
    )
    
    try:
        X = vectorizer.fit_transform(texts)
        
        # Clustering KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        logger.info(f"Clustering terminé: {len(set(labels))} clusters")
        return labels, vectorizer
        
    except Exception as e:
        logger.error(f"Erreur lors du clustering: {e}")
        return np.array([]), None


def get_cluster_keywords(
    texts: List[str],
    labels: np.ndarray,
    vectorizer: TfidfVectorizer,
    top_n: int = 10
) -> Dict[int, List[str]]:
    """
    Extrait les mots-clés les plus représentatifs de chaque cluster.
    
    Args:
        texts: Liste de textes
        labels: Labels de cluster
        vectorizer: Vectorizer TF-IDF utilisé
        top_n: Nombre de mots-clés par cluster
    
    Returns:
        Dictionnaire {cluster_id: [mots-clés]}
    """
    if vectorizer is None or len(texts) == 0:
        return {}
    
    cluster_keywords = {}
    
    try:
        # Vectoriser les textes
        X = vectorizer.transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Pour chaque cluster
        for cluster_id in set(labels):
            # Indices des textes du cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Moyenne des vecteurs TF-IDF du cluster
            cluster_vectors = X[cluster_indices]
            cluster_mean = np.mean(cluster_vectors.toarray(), axis=0)
            
            # Top mots-clés
            top_indices = np.argsort(cluster_mean)[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            cluster_keywords[cluster_id] = keywords
        
        return cluster_keywords
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des mots-clés: {e}")
        return {}


def lda_topic_modeling(
    texts: List[str],
    num_topics: int = 5,
    passes: int = 10,
    alpha: float = 0.1
) -> Tuple[LdaModel, List[List[Tuple[int, float]]]]:
    """
    Effectue une modélisation LDA avec Gensim.
    
    Args:
        texts: Liste de textes (chaque texte est une liste de tokens)
        num_topics: Nombre de topics
        passes: Nombre de passes
        alpha: Paramètre alpha de LDA
    
    Returns:
        Tuple (modèle LDA, topics par document)
    """
    if not texts:
        logger.warning("Liste de textes vide")
        return None, []
    
    logger.info(f"Modélisation LDA avec {num_topics} topics...")
    
    try:
        # Tokenisation si nécessaire
        if isinstance(texts[0], str):
            # Tokeniser les textes
            tokenized_texts = []
            for text in texts:
                tokens = text.lower().split()
                tokenized_texts.append(tokens)
        else:
            tokenized_texts = texts
        
        # Créer le dictionnaire et le corpus
        dictionary = corpora.Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=2, no_above=0.95)
        
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Entraîner le modèle LDA
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha=alpha,
            random_state=42
        )
        
        # Topics par document
        doc_topics = [lda_model[doc] for doc in corpus]
        
        logger.info(f"Modélisation LDA terminée")
        return lda_model, doc_topics
        
    except Exception as e:
        logger.error(f"Erreur lors de la modélisation LDA: {e}")
        return None, []


def get_lda_topics(lda_model: LdaModel, num_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extrait les topics du modèle LDA.
    
    Args:
        lda_model: Modèle LDA entraîné
        num_words: Nombre de mots par topic
    
    Returns:
        Dictionnaire {topic_id: [(mot, probabilité), ...]}
    """
    if lda_model is None:
        return {}
    
    topics = {}
    
    try:
        for topic_id in range(lda_model.num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=num_words)
            topics[topic_id] = topic_words
        
        return topics
        
    except Exception as e:
        logger.error(f"Erreur lors de l'extraction des topics: {e}")
        return {}

