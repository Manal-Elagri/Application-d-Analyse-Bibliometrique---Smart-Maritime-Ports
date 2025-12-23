"""
Module pour la génération de cartes stratégiques (Strategic Diagram)
basées sur la théorie des co-mots de Callon et al.
"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def classify_cluster_theme(centrality: float, density: float) -> str:
    """Classifie un cluster dans l'un des 4 quadrants (Seuil à 0.5)."""
    threshold = 0.5
    if centrality >= threshold and density >= threshold:
        return 'Motor'
    elif centrality >= threshold and density < threshold:
        return 'Basic'
    elif centrality < threshold and density >= threshold:
        return 'Niche'
    else:
        return 'Emerging'

def calculate_cluster_centrality_density(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    cooccurrence_df: pd.DataFrame
) -> pd.DataFrame:
    if cooccurrence_df.empty:
        logger.warning("DataFrame de co-occurrence vide")
        return pd.DataFrame()
    
    G = nx.Graph()
    c1, c2, cw = cooccurrence_df.columns[0], cooccurrence_df.columns[1], cooccurrence_df.columns[2]
    
    for _, row in cooccurrence_df.iterrows():
        G.add_edge(row[c1], row[c2], weight=row[cw])
    
    df_temp = df.copy().reset_index(drop=True)
    cluster_keywords = {}
    for i, keywords_list in enumerate(df_temp['Keywords_list']):
        if isinstance(keywords_list, list) and i < len(cluster_labels):
            cluster_id = cluster_labels[i]
            if cluster_id not in cluster_keywords:
                cluster_keywords[cluster_id] = []
            cluster_keywords[cluster_id].extend(keywords_list)
    
    cluster_keyword_counts = {cid: Counter(kws) for cid, kws in cluster_keywords.items()}
    results = []
    
    for cluster_id, keyword_counts in cluster_keyword_counts.items():
        cluster_kw_set = set(keyword_counts.keys())
        if len(cluster_kw_set) == 0: continue
        
        total_internal_weight = 0
        for kw1 in cluster_kw_set:
            if kw1 not in G: continue
            for kw2 in cluster_kw_set:
                if kw1 != kw2 and kw2 in G and G.has_edge(kw1, kw2):
                    total_internal_weight += G[kw1][kw2].get('weight', 0)
        
        max_internal_links = len(cluster_kw_set) * (len(cluster_kw_set) - 1) / 2
        density = (total_internal_weight / 2) / max_internal_links if max_internal_links > 0 else 0
        
        total_external_weight = 0
        for kw in cluster_kw_set:
            if kw in G:
                neighbors = list(G.neighbors(kw))
                for neighbor in neighbors:
                    if neighbor not in cluster_kw_set:
                        total_external_weight += G[kw][neighbor].get('weight', 0)
        
        centrality = total_external_weight / len(cluster_kw_set) if len(cluster_kw_set) > 0 else 0
        top_keywords = [kw for kw, _ in keyword_counts.most_common(5)]
        
        results.append({
            'Cluster': cluster_id,
            'Centrality': centrality,
            'Density': density,
            'Keywords': ', '.join(top_keywords),
            'NumKeywords': len(cluster_kw_set)
        })
    
    result_df = pd.DataFrame(results)
    
    if not result_df.empty:
        # 1. Normalisation
        for col in ['Centrality', 'Density']:
            c_min, c_max = result_df[col].min(), result_df[col].max()
            if c_max > c_min:
                result_df[f'{col}_norm'] = (result_df[col] - c_min) / (c_max - c_min)
            else:
                result_df[f'{col}_norm'] = 0.5
        
        # 2. AJOUT DE LA COLONNE THEME (CORRIGE L'ERREUR)
        result_df['Theme'] = result_df.apply(
            lambda row: classify_cluster_theme(row['Centrality_norm'], row['Density_norm']), 
            axis=1
        )
                
    return result_df

def plot_strategic_diagram(strategic_df, output_path, format='png'):
    if strategic_df.empty or 'Theme' not in strategic_df.columns:
        logger.error("DataFrame vide ou colonne 'Theme' manquante")
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {
        'Motor': '#2ecc71', 'Basic': '#3498db',
        'Niche': '#f1c40f', 'Emerging': '#e74c3c'
    }
    
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.3)
    
    for theme_type, color in colors.items():
        theme_df = strategic_df[strategic_df['Theme'] == theme_type]
        if not theme_df.empty:
            bubble_sizes = 300 + (theme_df['NumKeywords'] * 5) 
            ax.scatter(
                theme_df['Centrality_norm'], theme_df['Density_norm'],
                s=bubble_sizes, c=color, alpha=0.6, edgecolors='white',
                linewidths=1.5, label=f'Thèmes {theme_type}'
            )
            
            for _, row in theme_df.iterrows():
                ax.text(
                    row['Centrality_norm'], row['Density_norm'],
                    str(int(row['Cluster'])),
                    fontsize=9, fontweight='bold', ha='center', va='center'
                )

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlabel('Centralité (Interaction externe) →')
    ax.set_ylabel('Densité (Cohésion interne) →')
    
    # Labels des quadrants
    ax.text(0.95, 0.95, 'MOTEURS', alpha=0.3, fontsize=12, ha='right', transform=ax.transAxes, fontweight='bold')
    ax.text(0.05, 0.95, 'NICHES', alpha=0.3, fontsize=12, ha='left', transform=ax.transAxes, fontweight='bold')
    ax.text(0.05, 0.05, 'ÉMERGENTS', alpha=0.3, fontsize=12, ha='left', transform=ax.transAxes, fontweight='bold')
    ax.text(0.95, 0.05, 'TRANSVERSAUX', alpha=0.3, fontsize=12, ha='right', transform=ax.transAxes, fontweight='bold')

    ax.set_title('Carte Stratégique des Thèmes', fontsize=15, pad=20)
    plt.tight_layout()
    
    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, dpi=150)
    plt.close()
    return output_file

def get_strategic_analysis_data(strategic_df: pd.DataFrame) -> Dict:
    """Prépare les données pour l'IA."""
    if strategic_df.empty:
        return {}
    
    analysis_data = {'motor_themes': [], 'basic_themes': [], 'niche_themes': [], 'emerging_themes': []}
    
    for _, row in strategic_df.iterrows():
        theme_info = {
            'cluster_id': int(row['Cluster']),
            'keywords': row['Keywords'],
            'centrality': float(row['Centrality_norm']),
            'density': float(row['Density_norm']),
            'num_keywords': int(row['NumKeywords'])
        }
        
        theme_type = row['Theme']
        key = f"{theme_type.lower()}_themes"
        if key in analysis_data:
            analysis_data[key].append(theme_info)
    
    return analysis_data