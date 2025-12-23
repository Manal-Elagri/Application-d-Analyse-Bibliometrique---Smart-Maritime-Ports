"""
Module pour la génération de visualisations
"""

import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
import numpy as np
from typing import Optional, List, Dict
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration matplotlib avec style moderne
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#1a237e',
    'text.color': '#212121',
    'xtick.color': '#616161',
    'ytick.color': '#616161',
    'grid.color': '#e0e0e0',
    'grid.alpha': 0.5,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
})

# Palette de couleurs moderne
MODERN_COLORS = ['#1a237e', '#3f51b5', '#5c6bc0', '#7986cb', '#9fa8da', 
                 '#c5cae9', '#e8eaf6', '#283593', '#3949ab', '#303f9f']
sns.set_palette(MODERN_COLORS)


def plot_annual_evolution(df: pd.DataFrame, output_path: str, format: str = 'png') -> str:
    """
    Trace l'évolution annuelle des publications avec design moderne.
    
    Args:
        df: DataFrame avec colonnes 'Year' et 'Count'
        output_path: Chemin de sortie (sans extension)
        format: Format de sortie ('png', 'svg', 'pdf')
    
    Returns:
        Chemin du fichier généré
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('white')
    
    # Gradient de couleur moderne
    color1 = '#3f51b5'
    color2 = '#7986cb'
    
    # Ligne principale avec gradient
    line = ax.plot(df['Year'], df['Count'], marker='o', linewidth=3, 
                   markersize=10, color=color1, markerfacecolor='white',
                   markeredgewidth=2, markeredgecolor=color1, zorder=3)
    
    # Zone remplie avec gradient
    ax.fill_between(df['Year'], df['Count'], alpha=0.2, color=color1)
    
    # Ajouter une ligne de tendance si possible
    if len(df) > 1:
        z = np.polyfit(df['Year'], df['Count'], 1)
        p = np.poly1d(z)
        ax.plot(df['Year'], p(df['Year']), "--", alpha=0.5, color=color2, 
                linewidth=2, label='Tendance', zorder=2)
    
    # Style amélioré
    ax.set_xlabel('Année', fontsize=13, fontweight='bold', color='#1a237e')
    ax.set_ylabel('Nombre de publications', fontsize=13, fontweight='bold', color='#1a237e')
    ax.set_title('Évolution annuelle des publications\nSmart Maritime Ports', 
                 fontsize=16, fontweight='bold', color='#1a237e', pad=20)
    
    # Grille améliorée
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Style des axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdbdbd')
    ax.spines['bottom'].set_color('#bdbdbd')
    
    # Format des nombres sur l'axe Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Légende si tendance
    if len(df) > 1:
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    
    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format, 
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Graphique d'évolution annuelle sauvegardé: {output_file}")
    return output_file


def plot_top_items(df: pd.DataFrame, column: str, title: str, 
                   output_path: str, top_n: int = 20, format: str = 'png') -> str:
    """
    Trace un graphique en barres moderne pour les top items.
    
    Args:
        df: DataFrame avec colonnes 'Item' et 'Count'
        column: Nom de la colonne contenant les items
        title: Titre du graphique
        output_path: Chemin de sortie (sans extension)
        top_n: Nombre d'items à afficher
        format: Format de sortie
    
    Returns:
        Chemin du fichier généré
    """
    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))
    fig.patch.set_facecolor('white')
    
    top_df = df.head(top_n).copy()
    top_df = top_df.sort_values('Count', ascending=True)  # Tri pour meilleure visualisation
    
    # Palette de couleurs avec gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_df)))
    
    bars = ax.barh(range(len(top_df)), top_df['Count'], color=colors, 
                   edgecolor='white', linewidth=1.5, height=0.7)
    
    # Labels améliorés
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df[column], fontsize=11, color='#212121')
    ax.set_xlabel('Nombre de publications', fontsize=13, fontweight='bold', color='#1a237e')
    ax.set_title(title, fontsize=16, fontweight='bold', color='#1a237e', pad=20)
    
    # Ajouter les valeurs sur les barres avec style amélioré
    for i, (idx, row) in enumerate(top_df.iterrows()):
        value = int(row['Count'])
        ax.text(value + max(top_df['Count']) * 0.02, i, f'{value:,}', 
                va='center', fontsize=10, fontweight='bold', color='#1a237e')
    
    # Style des axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bdbdbd')
    ax.spines['bottom'].set_color('#bdbdbd')
    
    # Grille améliorée
    ax.grid(True, axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Format des nombres sur l'axe X
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format,
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Graphique {title} sauvegardé: {output_file}")
    return output_file


def plot_network_graph(G: nx.Graph, output_path: str, 
                       layout: str = 'spring', format: str = 'png',
                       node_size_factor: int = 100, figsize: tuple = (18, 14)) -> str:
    """
    Trace un graphe de réseau avec design moderne.
    
    Args:
        G: Graphe NetworkX
        output_path: Chemin de sortie (sans extension)
        layout: Type de layout ('spring', 'circular', 'kamada_kawai')
        format: Format de sortie
        node_size_factor: Facteur de taille des nœuds
        figsize: Taille de la figure
    
    Returns:
        Chemin du fichier généré
    """
    if G.number_of_nodes() == 0:
        logger.warning("Graphe vide, impossible de générer la visualisation")
        return ""
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Calculer les positions avec meilleurs paramètres
    if layout == 'spring':
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G)
        except:
            pos = nx.spring_layout(G, k=2, iterations=100)
    else:
        pos = nx.spring_layout(G, k=2, iterations=100)
    
    # Calculer les degrés pour la taille et couleur des nœuds
    degrees = dict(G.degree())
    node_sizes = [max(300, degrees[node] * node_size_factor) for node in G.nodes()]
    
    # Calculer les poids des arêtes
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [w / max_weight for w in edge_weights]
    
    # Dessiner les arêtes avec gradient de couleur
    edges = nx.draw_networkx_edges(
        G, pos, 
        width=[w * 2 + 0.5 for w in normalized_weights],
        alpha=0.4, 
        edge_color='#90caf9',
        style='solid',
        ax=ax
    )
    
    # Dessiner les nœuds avec gradient de couleur basé sur le degré
    node_colors = [degrees[node] for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes, 
        node_color=node_colors,
        cmap=plt.cm.Blues,
        alpha=0.9,
        edgecolors='white',
        linewidths=2,
        ax=ax
    )
    
    # Dessiner les labels pour les nœuds importants
    percentile_75 = np.percentile(list(degrees.values()), 75)
    important_nodes = [node for node, degree in degrees.items() 
                      if degree >= percentile_75 or degree >= 3]
    labels = {node: node[:20] + '...' if len(node) > 20 else node 
              for node in important_nodes}
    
    nx.draw_networkx_labels(
        G, pos, labels, 
        font_size=9, 
        font_weight='bold',
        font_color='#1a237e',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor='none', alpha=0.8),
        ax=ax
    )
    
    # Titre amélioré
    ax.set_title('Réseau de co-occurrence des mots-clés\nSmart Maritime Ports', 
                 fontsize=18, fontweight='bold', color='#1a237e', pad=20)
    ax.axis('off')
    
    # Ajouter une légende pour les tailles de nœuds
    if nodes:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                   norm=plt.Normalize(vmin=min(node_colors), 
                                                      vmax=max(node_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label('Degré du nœud', rotation=270, labelpad=15, 
                       fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format,
                facecolor='white', edgecolor='none')
    plt.close()
    
    logger.info(f"Graphe de réseau sauvegardé: {output_file}")
    return output_file


def create_wordcloud(text: str, output_path: str, format: str = 'png',
                    width: int = 1200, height: int = 600) -> str:
    """
    Crée un nuage de mots.
    
    Args:
        text: Texte à visualiser
        output_path: Chemin de sortie (sans extension)
        format: Format de sortie
        width: Largeur de l'image
        height: Hauteur de l'image
    
    Returns:
        Chemin du fichier généré
    """
    if not text:
        logger.warning("Texte vide, impossible de générer le WordCloud")
        return ""
    
    try:
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=200,
            colormap='viridis',
            relative_scaling=0.5
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Nuage de mots-clés', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = f"{output_path}.{format}"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format)
        plt.close()
        
        logger.info(f"WordCloud sauvegardé: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du WordCloud: {e}")
        return ""


def create_interactive_network(G: nx.Graph, output_path: str) -> str:
    """
    Crée un réseau interactif avec PyVis.
    
    Args:
        G: Graphe NetworkX
        output_path: Chemin de sortie HTML
    
    Returns:
        Chemin du fichier HTML généré
    """
    try:
        from pyvis.network import Network
        
        net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white')
        
        # Ajouter les nœuds et arêtes
        for node in G.nodes():
            degree = G.degree(node)
            net.add_node(node, size=degree * 5, title=f"{node}\nDegré: {degree}")
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            net.add_edge(u, v, value=weight, title=f"Poids: {weight}")
        
        # Options de physique
        net.set_options("""
        {
          "physics": {
            "enabled": true,
            "stabilization": {"iterations": 100}
          }
        }
        """)
        
        net.save_graph(output_path)
        
        logger.info(f"Réseau interactif sauvegardé: {output_path}")
        return output_path
        
    except ImportError:
        logger.warning("PyVis non disponible, création du réseau interactif ignorée")
        return ""
    except Exception as e:
        logger.error(f"Erreur lors de la création du réseau interactif: {e}")
        return ""


def plot_cluster_distribution(labels: np.ndarray, output_path: str, format: str = 'png') -> str:
    """
    Trace la distribution des clusters.
    
    Args:
        labels: Labels de cluster
        output_path: Chemin de sortie (sans extension)
        format: Format de sortie
    
    Returns:
        Chemin du fichier généré
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique, counts = np.unique(labels, return_counts=True)
    
    bars = ax.bar(unique, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique))))
    ax.set_xlabel('Cluster ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de documents', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des clusters thématiques', fontsize=14, fontweight='bold')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format=format)
    plt.close()
    
    logger.info(f"Graphique de distribution des clusters sauvegardé: {output_file}")
    return output_file

