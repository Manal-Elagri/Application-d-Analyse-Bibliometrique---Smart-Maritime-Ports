"""
Exemple d'utilisation de l'application bibliométrique
Ce script montre comment utiliser les modules directement en Python
"""

import sys
import os

# Ajouter le répertoire app au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from load_data import load_scopus_csv, remove_duplicates
from preprocess import preprocess_dataframe
from bibliometry import annual_evolution, top_authors, top_keywords, get_statistics
from networks import create_cooccurrence_network
from visualizations import plot_annual_evolution, plot_network_graph
from utils import get_export_path

def example_analysis(csv_file_path: str):
    """
    Exemple d'analyse complète d'un fichier Scopus
    
    Args:
        csv_file_path: Chemin vers le fichier CSV Scopus
    """
    print("=" * 60)
    print("EXEMPLE D'ANALYSE BIBLIOMÉTRIQUE")
    print("=" * 60)
    
    # 1. Chargement des données
    print("\n1. Chargement des données...")
    df = load_scopus_csv(csv_file_path, chunksize=5000)
    print(f"   ✓ {len(df)} publications chargées")
    
    # 2. Suppression des doublons
    print("\n2. Suppression des doublons...")
    df = remove_duplicates(df)
    print(f"   ✓ {len(df)} publications après nettoyage")
    
    # 3. Préprocessing
    print("\n3. Préprocessing des données...")
    df_processed = preprocess_dataframe(df)
    print("   ✓ Données prétraitées")
    
    # 4. Statistiques générales
    print("\n4. Statistiques générales:")
    stats = get_statistics(df_processed)
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # 5. Évolution annuelle
    print("\n5. Calcul de l'évolution annuelle...")
    annual_df = annual_evolution(df_processed)
    print(f"   ✓ {len(annual_df)} années analysées")
    
    # Visualisation
    output_path = get_export_path("evolution_annuelle")
    plot_annual_evolution(annual_df, output_path)
    print(f"   ✓ Graphique sauvegardé: {output_path}.png")
    
    # 6. Top auteurs
    print("\n6. Identification des top auteurs...")
    top_auth = top_authors(df_processed, top_n=10)
    print("   Top 5 auteurs:")
    for idx, row in top_auth.head(5).iterrows():
        print(f"   - {row['Author']}: {row['Count']} publications")
    
    # 7. Top mots-clés
    print("\n7. Identification des top mots-clés...")
    top_kw = top_keywords(df_processed, top_n=20)
    print("   Top 10 mots-clés:")
    for idx, row in top_kw.head(10).iterrows():
        print(f"   - {row['Keyword']}: {row['Count']} occurrences")
    
    # 8. Réseau de co-occurrence
    print("\n8. Création du réseau de co-occurrence...")
    from bibliometry import co_occurrence_matrix
    cooc_df = co_occurrence_matrix(df_processed, min_freq=2)
    
    if not cooc_df.empty:
        G = create_cooccurrence_network(cooc_df, min_weight=2, top_n=100)
        print(f"   ✓ Réseau créé: {G.number_of_nodes()} nœuds, {G.number_of_edges()} arêtes")
        
        # Visualisation
        output_path = get_export_path("reseau_cooccurrence")
        plot_network_graph(G, output_path)
        print(f"   ✓ Graphique sauvegardé: {output_path}.png")
    
    print("\n" + "=" * 60)
    print("ANALYSE TERMINÉE!")
    print("=" * 60)
    print(f"\nLes résultats sont disponibles dans le dossier 'exports/'")


if __name__ == "__main__":
    # Exemple d'utilisation
    # Remplacez par le chemin de votre fichier CSV
    csv_file = "data/scopus_data.csv"
    
    if os.path.exists(csv_file):
        example_analysis(csv_file)
    else:
        print(f"Fichier non trouvé: {csv_file}")
        print("\nPour utiliser cet exemple:")
        print("1. Placez votre fichier CSV Scopus dans le dossier 'data/'")
        print("2. Modifiez la variable 'csv_file' dans ce script")
        print("3. Exécutez: python example_usage.py")
        print("\nOu utilisez l'interface Streamlit:")
        print("  streamlit run app/interface.py")

