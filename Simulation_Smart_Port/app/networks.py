"""
Module pour la création de réseaux (co-occurrence, co-auteurs)
"""

import networkx as nx
import pandas as pd
from typing import List, Tuple, Optional
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_cooccurrence_network(
    cooccurrence_df: pd.DataFrame,
    min_weight: int = 2,
    top_n: Optional[int] = None
) -> nx.Graph:
    """
    Crée un graphe de co-occurrence des mots-clés.
    
    Args:
        cooccurrence_df: DataFrame avec colonnes 'Keyword1', 'Keyword2', 'Co_occurrence'
        min_weight: Poids minimum pour inclure une arête
        top_n: Nombre maximum d'arêtes à inclure (optionnel)
    
    Returns:
        Graphe NetworkX
    """
    if cooccurrence_df.empty:
        logger.warning("DataFrame de co-occurrence vide")
        return nx.Graph()
    
    G = nx.Graph()
    
    # Filtrer par poids minimum
    filtered_df = cooccurrence_df[cooccurrence_df['Co_occurrence'] >= min_weight]
    
    # Limiter le nombre d'arêtes si nécessaire
    if top_n:
        filtered_df = filtered_df.head(top_n)
    
    # Ajouter les arêtes
    for _, row in filtered_df.iterrows():
        kw1 = row['Keyword1']
        kw2 = row['Keyword2']
        weight = row['Co_occurrence']
        
        G.add_edge(kw1, kw2, weight=weight)
    
    logger.info(f"Réseau de co-occurrence créé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    return G


def create_coauthors_network(
    df: pd.DataFrame,
    min_collaborations: int = 2,
    max_authors: int = 100
) -> nx.Graph:
    """
    Crée un réseau de co-auteurs.
    
    Args:
        df: DataFrame avec colonne 'Authors_list'
        min_collaborations: Nombre minimum de collaborations pour inclure une arête
        max_authors: Nombre maximum d'auteurs à inclure (pour éviter la saturation)
    
    Returns:
        Graphe NetworkX
    """
    if 'Authors_list' not in df.columns:
        logger.warning("Colonne 'Authors_list' absente")
        return nx.Graph()
    
    # Compter les fréquences des auteurs
    all_authors = []
    for authors_list in df['Authors_list']:
        if isinstance(authors_list, list):
            all_authors.extend(authors_list)
    
    author_counts = Counter(all_authors)
    
    # Sélectionner les auteurs les plus fréquents
    top_authors = {author for author, _ in author_counts.most_common(max_authors)}
    
    # Créer le graphe
    G = nx.Graph()
    
    # Parcourir les publications
    coauthor_pairs = Counter()
    
    for authors_list in df['Authors_list']:
        if isinstance(authors_list, list):
            # Filtrer les auteurs fréquents
            filtered_authors = [auth for auth in authors_list if auth in top_authors]
            
            # Créer toutes les paires de co-auteurs
            for i, auth1 in enumerate(filtered_authors):
                for auth2 in filtered_authors[i+1:]:
                    pair = tuple(sorted([auth1, auth2]))
                    coauthor_pairs[pair] += 1
    
    # Ajouter les arêtes avec poids suffisant
    for (auth1, auth2), count in coauthor_pairs.items():
        if count >= min_collaborations:
            G.add_edge(auth1, auth2, weight=count)
    
    logger.info(f"Réseau de co-auteurs créé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
    return G


def get_network_metrics(G: nx.Graph) -> dict:
    """
    Calcule des métriques sur un graphe.
    
    Args:
        G: Graphe NetworkX
    
    Returns:
        Dictionnaire avec les métriques
    """
    if G.number_of_nodes() == 0:
        return {}
    
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
    }
    
    if G.number_of_nodes() > 0:
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
        
        # Composantes connexes
        components = list(nx.connected_components(G))
        metrics['num_components'] = len(components)
        if components:
            metrics['largest_component_size'] = len(max(components, key=len))
    
    return metrics


def filter_network_by_degree(G: nx.Graph, min_degree: int = 2) -> nx.Graph:
    """
    Filtre un graphe en gardant uniquement les nœuds avec un degré minimum.
    
    Args:
        G: Graphe NetworkX
        min_degree: Degré minimum
    
    Returns:
        Nouveau graphe filtré
    """
    nodes_to_keep = [node for node, degree in G.degree() if degree >= min_degree]
    G_filtered = G.subgraph(nodes_to_keep).copy()
    
    logger.info(f"Réseau filtré: {G_filtered.number_of_nodes()} nœuds (degré >= {min_degree})")
    return G_filtered

