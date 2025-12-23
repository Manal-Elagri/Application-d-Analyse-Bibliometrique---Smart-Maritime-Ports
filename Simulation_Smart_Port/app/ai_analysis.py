"""
Module d'intégration avec Ollama pour l'analyse IA
"""

import requests
import json
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Ollama
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:1.5b"


def ask_ai(prompt: str, max_retries: int = 3) -> str:
    """
    Envoie une requête à Ollama et retourne la réponse.
    
    Args:
        prompt: Prompt à envoyer à l'IA
        max_retries: Nombre maximum de tentatives en cas d'échec
    
    Returns:
        Réponse de l'IA ou message d'erreur
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Envoi de la requête à Ollama (tentative {attempt + 1})...")
            response = requests.post(
                OLLAMA_ENDPOINT,
                json=payload,
                timeout=120  # Timeout de 2 minutes
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result.get('response', '')
                logger.info("Réponse reçue de Ollama")
                return ai_response
            else:
                logger.warning(f"Erreur HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            error_msg = (
                "Erreur de connexion à Ollama. "
                "Vérifiez que Ollama est démarré et accessible sur http://localhost:11434"
            )
            logger.error(error_msg)
            return error_msg
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout lors de la tentative {attempt + 1}")
            if attempt < max_retries - 1:
                continue
            return "Timeout: La requête a pris trop de temps. Veuillez réessayer."
            
        except Exception as e:
            logger.error(f"Erreur lors de la requête: {e}")
            return f"Erreur: {str(e)}"
    
    return "Erreur: Impossible de contacter Ollama après plusieurs tentatives."


def analyze_cluster(cluster_texts: list, cluster_id: int, keywords: list) -> str:
    """
    Analyse un cluster d'abstracts avec l'IA.
    
    Args:
        cluster_texts: Liste des abstracts du cluster
        cluster_id: ID du cluster
        keywords: Mots-clés du cluster
    
    Returns:
        Analyse générée par l'IA
    """
    # Préparer un échantillon des abstracts
    sample_texts = cluster_texts[:5]  # Prendre les 5 premiers
    
    prompt = f"""Tu es un expert en analyse bibliométrique et en ports maritimes intelligents.

Analyse ce cluster thématique de publications scientifiques:

Cluster ID: {cluster_id}
Mots-clés principaux: {', '.join(keywords[:10])}

Abstracts représentatifs:
{chr(10).join([f"- {text[:500]}..." for text in sample_texts])}

Fournis une analyse concise (3-4 paragraphes) qui décrit:
1. La thématique principale de ce cluster
2. Les tendances et concepts clés identifiés
3. L'importance de cette thématique dans le domaine des Smart Maritime Ports
"""
    
    return ask_ai(prompt)


def analyze_trends(annual_data: dict, top_keywords: list) -> str:
    """
    Analyse les tendances scientifiques avec l'IA.
    
    Args:
        annual_data: Données d'évolution annuelle
        top_keywords: Liste des mots-clés les plus fréquents
    
    Returns:
        Analyse des tendances générée par l'IA
    """
    prompt = f"""Tu es un expert en analyse bibliométrique et en ports maritimes intelligents.

Analyse les tendances scientifiques suivantes:

Évolution des publications:
{json.dumps(annual_data, indent=2)}

Mots-clés les plus fréquents:
{', '.join(top_keywords[:20])}

Fournis une analyse approfondie (5-6 paragraphes) qui couvre:
1. L'évolution temporelle de la recherche
2. Les thématiques dominantes identifiées
3. Les tendances émergentes
4. Les perspectives futures de recherche dans le domaine des Smart Maritime Ports
"""
    
    return ask_ai(prompt)


def analyze_graph(graph_description: str, graph_type: str = "co-occurrence", 
                 top_nodes: list = None, metrics: dict = None) -> str:
    """
    Analyse un graphe avec l'IA avec des détails améliorés.
    
    Args:
        graph_description: Description textuelle du graphe (métriques, nœuds importants, etc.)
        graph_type: Type de graphe ("co-occurrence" ou "co-auteurs")
        top_nodes: Liste des nœuds les plus importants
        metrics: Dictionnaire avec les métriques du graphe
    
    Returns:
        Analyse du graphe générée par l'IA
    """
    # Construire une description enrichie
    enriched_description = graph_description
    
    if metrics:
        enriched_description += f"\n\nMétriques détaillées du réseau:\n"
        enriched_description += f"- Nombre total de nœuds: {metrics.get('nodes', 'N/A')}\n"
        enriched_description += f"- Nombre total d'arêtes: {metrics.get('edges', 'N/A')}\n"
        enriched_description += f"- Densité du réseau: {metrics.get('density', 0):.4f}\n"
        enriched_description += f"- Degré moyen: {metrics.get('avg_degree', 'N/A')}\n"
        enriched_description += f"- Nombre de composantes connexes: {metrics.get('num_components', 'N/A')}\n"
        if metrics.get('largest_component_size'):
            enriched_description += f"- Taille de la plus grande composante: {metrics.get('largest_component_size', 'N/A')}\n"
    
    if top_nodes:
        enriched_description += f"\n\nNœuds les plus importants (top 10):\n"
        for i, node in enumerate(top_nodes[:10], 1):
            enriched_description += f"{i}. {node}\n"
    
    prompt = f"""Tu es un expert en analyse de réseaux scientifiques, bibliométrie et ports maritimes intelligents.

Analyse approfondie d'un graphe de {graph_type}:

{enriched_description}

Fournis une analyse scientifique complète et structurée (6-8 paragraphes) qui couvre:

1. **Structure et topologie du réseau** (1-2 paragraphes):
   - Caractéristiques structurelles principales (densité, connectivité, modularité)
   - Type de réseau observé (centralisé, distribué, modulaire, etc.)
   - Implications de cette structure pour le domaine

2. **Nœuds centraux et hubs** (1-2 paragraphes):
   - Identification et analyse des nœuds les plus importants
   - Leur rôle dans la connectivité du réseau
   - Signification scientifique de ces concepts/auteurs dans le contexte des Smart Maritime Ports

3. **Relations et clusters thématiques** (1-2 paragraphes):
   - Groupes de nœuds fortement connectés (communautés thématiques)
   - Relations fortes identifiées et leur signification
   - Ponts entre différentes communautés thématiques

4. **Interprétation scientifique et implications** (2 paragraphes):
   - Ce que révèle ce réseau sur l'état actuel de la recherche
   - Tendances émergentes ou domaines sous-explorés identifiés
   - Recommandations pour les futures recherches basées sur la structure du réseau

Utilise un langage scientifique précis mais accessible. Structure ta réponse avec des paragraphes clairs et des transitions logiques.
"""
    
    return ask_ai(prompt)


def generate_research_recommendations(analysis_summary: str) -> str:
    """
    Génère des recommandations de recherche basées sur l'analyse.
    
    Args:
        analysis_summary: Résumé de l'analyse bibliométrique
    
    Returns:
        Recommandations générées par l'IA
    """
    prompt = f"""Tu es un expert en recherche scientifique sur les ports maritimes intelligents.

Basé sur cette analyse bibliométrique:

{analysis_summary}

Génère des recommandations de recherche (5-7 points) qui identifient:
1. Les lacunes dans la littérature actuelle
2. Les directions de recherche prometteuses
3. Les collaborations potentielles suggérées par les données
4. Les questions de recherche importantes non encore explorées
"""
    
    return ask_ai(prompt)


def analyze_strategic_map(strategic_data: dict) -> str:
    """
    Analyse une carte stratégique avec l'IA.
    
    Args:
        strategic_data: Dictionnaire avec les données de la carte stratégique
    
    Returns:
        Analyse générée par l'IA
    """
    import json
    
    prompt = f"""Tu es un expert en analyse bibliométrique stratégique et en ports maritimes intelligents.

Analyse cette carte stratégique (Strategic Diagram) basée sur la théorie des co-mots de Callon:

THÈMES MOTEURS (Haut-Droit - Centralité élevée, Densité élevée):
{json.dumps(strategic_data.get('motor_themes', []), indent=2)}

THÈMES DE BASE (Bas-Droit - Centralité élevée, Densité faible):
{json.dumps(strategic_data.get('basic_themes', []), indent=2)}

THÈMES DE NICHE (Haut-Gauche - Centralité faible, Densité élevée):
{json.dumps(strategic_data.get('niche_themes', []), indent=2)}

THÈMES ÉMERGENTS (Bas-Gauche - Centralité faible, Densité faible):
{json.dumps(strategic_data.get('emerging_themes', []), indent=2)}

Fournis une analyse stratégique complète et structurée (6-8 paragraphes) qui couvre:

1. **Thèmes Moteurs** (1-2 paragraphes):
   - Identification des thèmes moteurs actuels (X, Y, etc.)
   - Leur importance dans le domaine des Smart Maritime Ports
   - Pourquoi ils sont considérés comme moteurs (centralité ET densité élevées)

2. **Thèmes de Base** (1 paragraphe):
   - Thèmes fondamentaux mais moins développés
   - Leur rôle dans la recherche actuelle

3. **Thèmes de Niche** (1 paragraphe):
   - Thèmes spécialisés et isolés
   - Leur signification dans le domaine

4. **Thèmes Émergents** (1-2 paragraphes):
   - Identification des thèmes émergents (Z, etc.)
   - Potentiel de développement futur
   - Opportunités de recherche

5. **Recommandations stratégiques** (1-2 paragraphes):
   - Orientations de recherche prioritaires
   - Domaines à développer
   - Synergies possibles entre thèmes

Utilise un langage scientifique précis et structuré. Mentionne explicitement les noms des thèmes moteurs et émergents identifiés.
"""
    
    return ask_ai(prompt)


def generate_comprehensive_analysis(
    stats: dict,
    trends: dict,
    clusters: dict,
    graph_metrics: dict
) -> str:
    """
    Génère une analyse complète combinant toutes les informations.
    
    Args:
        stats: Statistiques générales
        trends: Données de tendances
        clusters: Informations sur les clusters
        graph_metrics: Métriques du graphe
    
    Returns:
        Analyse complète générée par l'IA
    """
    prompt = f"""Tu es un expert en analyse bibliométrique et en ports maritimes intelligents.

Génère un rapport d'analyse scientifique complet et professionnel basé sur ces données:

STATISTIQUES GÉNÉRALES:
{json.dumps(stats, indent=2)}

TENDANCES:
{json.dumps(trends, indent=2)}

CLUSTERS THÉMATIQUES:
{json.dumps(clusters, indent=2)}

MÉTRIQUES DU RÉSEAU:
{json.dumps(graph_metrics, indent=2)}

Fournis un rapport structuré (8-10 paragraphes) qui inclut:
1. Introduction et contexte
2. Vue d'ensemble des données
3. Analyse des tendances temporelles
4. Identification des thématiques principales
5. Analyse de la structure du réseau
6. Interprétation scientifique
7. Conclusions et perspectives
"""
    
    return ask_ai(prompt)

