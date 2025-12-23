"""
Interface Streamlit pour l'application bibliométrique
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import tempfile
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import des modules de l'application
try:
    from app.load_data import load_scopus_csv, remove_duplicates
    from app.preprocess import preprocess_dataframe
    from app.bibliometry import (
        annual_evolution, top_authors, top_journals, top_keywords,
        co_occurrence_matrix, get_statistics
    )
    from app.networks import (
        create_cooccurrence_network, create_coauthors_network,
        get_network_metrics, filter_network_by_degree
    )
    from app.nlp_models import tfidf_clustering, get_cluster_keywords, lda_topic_modeling, get_lda_topics
    from app.visualizations import (
        plot_annual_evolution, plot_top_items, plot_network_graph,
        create_wordcloud, create_interactive_network, plot_cluster_distribution
    )
    from app.ai_analysis import (
        ask_ai, analyze_cluster, analyze_trends, analyze_graph,
        generate_research_recommendations, generate_comprehensive_analysis,
        analyze_strategic_map
    )
    from app.strategic_map import (
        calculate_cluster_centrality_density, plot_strategic_diagram,
        get_strategic_analysis_data
    )
    from app.pdf_generator import generate_pdf, generate_comprehensive_report
    from app.utils import get_export_path, ensure_dir
except ImportError as e:
    # Fallback pour les imports relatifs
    from load_data import load_scopus_csv, remove_duplicates
    from preprocess import preprocess_dataframe
    from bibliometry import (
        annual_evolution, top_authors, top_journals, top_keywords,
        co_occurrence_matrix, get_statistics
    )
    from networks import (
        create_cooccurrence_network, create_coauthors_network,
        get_network_metrics, filter_network_by_degree
    )
    from nlp_models import tfidf_clustering, get_cluster_keywords, lda_topic_modeling, get_lda_topics
    from visualizations import (
        plot_annual_evolution, plot_top_items, plot_network_graph,
        create_wordcloud, create_interactive_network, plot_cluster_distribution
    )
    from ai_analysis import (
        ask_ai, analyze_cluster, analyze_trends, analyze_graph,
        generate_research_recommendations, generate_comprehensive_analysis,
        analyze_strategic_map
    )
    from strategic_map import (
        calculate_cluster_centrality_density, plot_strategic_diagram,
        get_strategic_analysis_data
    )
    from pdf_generator import generate_pdf, generate_comprehensive_report
    from utils import get_export_path, ensure_dir

# Configuration de la page
st.set_page_config(
    page_title="Analyse Bibliométrique - Smart Maritime Ports",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
    <style>
    /* Style général */
    .main {
        padding: 2rem;
    }
    
    /* Titres */
    h1 {
        color: #1a237e;
        font-weight: 700;
        border-bottom: 3px solid #3f51b5;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #283593;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #3949ab;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f5f5f5;
    }
    
    /* Boutons */
    .stButton > button {
        background: linear-gradient(90deg, #3f51b5 0%, #5c6bc0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #5c6bc0 0%, #7986cb 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Métriques */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1a237e;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #616161;
        font-weight: 500;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #3f51b5;
        border-radius: 8px;
        padding: 1rem;
        background-color: #f5f5f5;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #283593;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation de la session
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'analyses' not in st.session_state:
    st.session_state.analyses = {}
if 'all_visualizations' not in st.session_state:
    st.session_state.all_visualizations = {}


def main():
    """Fonction principale de l'interface Streamlit"""
    
    # Sidebar
    st.sidebar.title("🚢 Analyse Bibliométrique")
    st.sidebar.markdown("### Smart Maritime Ports")
    
    st.sidebar.markdown("---")
    
    # Menu de navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Importation des données", "📈 Analyses bibliométriques", 
         "🕸️ Réseaux et visualisations", "🤖 Analyse IA", "📄 Export PDF"]
    )
    
    # Mapping des pages
    page_mapping = {
        "📊 Importation": "📊 Importation des données",
        "📈 Analyses": "📈 Analyses bibliométriques",
        "🕸️ Réseaux": "🕸️ Réseaux et visualisations",
        "🤖 IA": "🤖 Analyse IA",
        "📄 Export PDF": "📄 Export PDF"
    }
    page = page_mapping.get(page, page)
    
    # Page 1: Importation des données
    if page == "📊 Importation des données":
        st.title("📊 Importation et préparation des données")
        
        uploaded_file = st.file_uploader(
            "Téléversez un fichier CSV Scopus",
            type=['csv'],
            help="Format attendu: CSV avec colonnes Title, Abstract, Author Keywords, Authors, Year, Source title"
        )
        
        if uploaded_file is not None:
            with st.spinner("Chargement du fichier..."):
                # Sauvegarder temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Charger les données
                    df = load_scopus_csv(tmp_path, chunksize=5000)
                    
                    if df is not None and len(df) > 0:
                        st.session_state.df = df
                        
                        st.success(f"✅ Fichier chargé avec succès: {len(df)} lignes")
                        
                        # Afficher les premières lignes
                        st.subheader("Aperçu des données")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Statistiques initiales
                        st.subheader("Statistiques initiales")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Nombre de publications", len(df))
                        col2.metric("Colonnes", len(df.columns))
                        col3.metric("Années", f"{int(df['Year'].min()) if 'Year' in df.columns else 'N/A'} - {int(df['Year'].max()) if 'Year' in df.columns else 'N/A'}")
                        col4.metric("Doublons détectés", len(df) - len(df.drop_duplicates(subset=['Title', 'Authors', 'Year'] if all(c in df.columns for c in ['Title', 'Authors', 'Year']) else [])))
                        
                        # Options de nettoyage
                        st.subheader("Options de nettoyage")
                        remove_dups = st.checkbox("Supprimer les doublons", value=True)
                        
                        if st.button("🔧 Préparer les données"):
                            with st.spinner("Préprocessing en cours..."):
                                if remove_dups:
                                    df = remove_duplicates(df)
                                
                                df_processed = preprocess_dataframe(df)
                                st.session_state.df_processed = df_processed
                                
                                st.success("✅ Données préparées avec succès!")
                                st.session_state.df = df
                    else:
                        st.error("❌ Le fichier est vide ou invalide")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement: {str(e)}")
                finally:
                    # Nettoyer le fichier temporaire
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
    
    # Page 2: Analyses bibliométriques
    elif page == "📈 Analyses bibliométriques":
        st.title("📈 Analyses bibliométriques")
        
        if st.session_state.df_processed is None:
            st.warning("⚠️ Veuillez d'abord importer et préparer les données")
        else:
            df = st.session_state.df_processed
            
            # Onglets pour différentes analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📅 Évolution annuelle", "👥 Top auteurs", 
                "📚 Top journaux", "🔑 Top mots-clés", "📊 Statistiques"
            ])
            
            with tab1:
                st.subheader("Évolution annuelle des publications")
                annual_df = annual_evolution(df)
                
                if not annual_df.empty:
                    st.line_chart(annual_df.set_index('Year'))
                    
                    # Export
                    csv = annual_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger en CSV",
                        csv,
                        "evolution_annuelle.csv",
                        "text/csv"
                    )
            
            with tab2:
                st.subheader("Top auteurs")
                top_n_authors = st.slider("Nombre d'auteurs", 5, 50, 20, key="top_authors")
                authors_df = top_authors(df, top_n_authors)
                
                if not authors_df.empty:
                    st.dataframe(authors_df, use_container_width=True)
                    
                    csv = authors_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger en CSV",
                        csv,
                        "top_auteurs.csv",
                        "text/csv"
                    )
            
            with tab3:
                st.subheader("Top journaux")
                top_n_journals = st.slider("Nombre de journaux", 5, 50, 20, key="top_journals")
                journals_df = top_journals(df, top_n_journals)
                
                if not journals_df.empty:
                    st.dataframe(journals_df, use_container_width=True)
                    
                    csv = journals_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger en CSV",
                        csv,
                        "top_journaux.csv",
                        "text/csv"
                    )
            
            with tab4:
                st.subheader("Top mots-clés")
                top_n_keywords = st.slider("Nombre de mots-clés", 10, 100, 50, key="top_keywords")
                keywords_df = top_keywords(df, top_n_keywords)
                
                if not keywords_df.empty:
                    st.dataframe(keywords_df, use_container_width=True)
                    
                    csv = keywords_df.to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger en CSV",
                        csv,
                        "top_mots_cles.csv",
                        "text/csv"
                    )
            
            with tab5:
                st.subheader("Statistiques générales")
                stats = get_statistics(df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total publications", stats.get('total_publications', 0))
                    st.metric("Période", stats.get('years_span', 'N/A'))
                with col2:
                    st.metric("Auteurs uniques", stats.get('unique_authors', 0))
                    st.metric("Journaux uniques", stats.get('unique_journals', 0))
                    st.metric("Mots-clés uniques", stats.get('unique_keywords', 0))
    
    # Page 3: Réseaux et visualisations
    elif page == "🕸️ Réseaux et visualisations":
        st.title("🕸️ Réseaux et visualisations")
        
        if st.session_state.df_processed is None:
            st.warning("⚠️ Veuillez d'abord importer et préparer les données")
        else:
            df = st.session_state.df_processed
            
            tab1, tab2, tab3, tab4 = st.tabs([
                "🕸️ Réseau de co-occurrence", "👥 Réseau de co-auteurs",
                "☁️ Nuage de mots", "📊 Clustering"
            ])
            
            with tab1:
                st.subheader("Réseau de co-occurrence des mots-clés")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_freq = st.slider("Fréquence minimale", 1, 10, 2, key="cooc_min_freq")
                    min_weight = st.slider("Poids minimum des arêtes", 1, 10, 2, key="cooc_min_weight")
                with col2:
                    top_n_edges = st.slider("Nombre max d'arêtes", 50, 500, 200, key="cooc_top_n")
                    layout_type = st.selectbox("Layout", ["spring", "circular", "kamada_kawai"], key="cooc_layout")
                
                if st.button("🔄 Générer le réseau"):
                    with st.spinner("Création du réseau..."):
                        cooc_df = co_occurrence_matrix(df, min_freq=min_freq)
                        
                        if not cooc_df.empty:
                            G = create_cooccurrence_network(cooc_df, min_weight=min_weight, top_n=top_n_edges)
                            
                            if G.number_of_nodes() > 0:
                                # Visualisation matplotlib
                                output_path = get_export_path("cooccurrence_network")
                                img_path = plot_network_graph(G, output_path, layout=layout_type)
                                
                                if img_path:
                                    st.image(img_path, use_container_width=True)
                                    
                                    # Bouton de téléchargement
                                    with open(img_path, 'rb') as f:
                                        img_bytes = f.read()
                                    st.download_button(
                                        "📥 Télécharger l'image",
                                        img_bytes,
                                        "cooccurrence_network.png",
                                        "image/png"
                                    )
                                
                                # Métriques du réseau
                                metrics = get_network_metrics(G)
                                st.subheader("Métriques du réseau")
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Nœuds", metrics.get('nodes', 0))
                                col2.metric("Arêtes", metrics.get('edges', 0))
                                col3.metric("Densité", f"{metrics.get('density', 0):.4f}")
                                col4.metric("Composantes", metrics.get('num_components', 0))
                                
                                # Sauvegarder pour l'analyse IA
                                st.session_state.analyses['cooccurrence_graph'] = G
                                st.session_state.analyses['cooccurrence_metrics'] = metrics
                                
                                # Réseau interactif
                                st.subheader("Réseau interactif")
                                html_path = get_export_path("cooccurrence_interactive.html")
                                html_file = create_interactive_network(G, html_path)
                                
                                if html_file:
                                    with open(html_file, 'r', encoding='utf-8') as f:
                                        html_content = f.read()
                                    st.components.v1.html(html_content, height=800)
            
            with tab2:
                st.subheader("Réseau de co-auteurs")
                
                col1, col2 = st.columns(2)
                with col1:
                    min_collab = st.slider("Collaborations minimales", 1, 5, 2, key="coauth_min")
                    max_authors = st.slider("Auteurs max", 50, 500, 100, key="coauth_max")
                
                if st.button("🔄 Générer le réseau de co-auteurs"):
                    with st.spinner("Création du réseau..."):
                        G = create_coauthors_network(df, min_collaborations=min_collab, max_authors=max_authors)
                        
                        if G.number_of_nodes() > 0:
                            output_path = get_export_path("coauthors_network")
                            img_path = plot_network_graph(G, output_path, layout="spring")
                            
                            if img_path:
                                st.image(img_path, use_container_width=True)
                                
                                with open(img_path, 'rb') as f:
                                    img_bytes = f.read()
                                st.download_button(
                                    "📥 Télécharger l'image",
                                    img_bytes,
                                    "coauthors_network.png",
                                    "image/png"
                                )
                            
                            metrics = get_network_metrics(G)
                            st.subheader("Métriques du réseau")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Nœuds", metrics.get('nodes', 0))
                            col2.metric("Arêtes", metrics.get('edges', 0))
                            col3.metric("Densité", f"{metrics.get('density', 0):.4f}")
            
            with tab3:
                st.subheader("Nuage de mots-clés")
                
                if st.button("🔄 Générer le WordCloud"):
                    with st.spinner("Création du nuage de mots..."):
                        # Récupérer tous les mots-clés
                        all_keywords = []
                        for kw_list in df['Keywords_list']:
                            if isinstance(kw_list, list):
                                all_keywords.extend(kw_list)
                        
                        text = ' '.join(all_keywords)
                        
                        if text:
                            output_path = get_export_path("wordcloud")
                            img_path = create_wordcloud(text, output_path)
                            
                            if img_path:
                                st.image(img_path, use_container_width=True)
                                
                                with open(img_path, 'rb') as f:
                                    img_bytes = f.read()
                                st.download_button(
                                    "📥 Télécharger l'image",
                                    img_bytes,
                                    "wordcloud.png",
                                    "image/png"
                                )
            
            with tab4:
                st.subheader("Clustering thématique")
                
                n_clusters = st.slider("Nombre de clusters", 3, 10, 5, key="n_clusters")
                
                if st.button("🔄 Effectuer le clustering"):
                    with st.spinner("Clustering en cours..."):
                        # 1. Filtrer le DataFrame pour n'avoir que les lignes avec des abstracts valides
                        mask = st.session_state.df_processed['Abstract_cleaned'].notna() & \
                               (st.session_state.df_processed['Abstract_cleaned'] != "")
                        df_clustering = st.session_state.df_processed[mask].copy()
                        
                        texts = df_clustering['Abstract_cleaned'].tolist()
                        
                        if texts:
                            # Effectuer le clustering sur le DF filtré
                            labels, vectorizer = tfidf_clustering(texts, n_clusters=n_clusters)
                            
                            # Distribution des clusters
                            output_path = get_export_path("cluster_distribution")
                            img_path = plot_cluster_distribution(labels, output_path)
                            
                            if img_path:
                                st.image(img_path, use_container_width=True)
                            
                            # Mots-clés par cluster
                            cluster_keywords = get_cluster_keywords(texts, labels, vectorizer)
                            
                            st.subheader("Mots-clés par cluster")
                            for cluster_id, keywords in cluster_keywords.items():
                                with st.expander(f"Cluster {cluster_id}"):
                                    st.write(", ".join(keywords))
                            
                            st.session_state.analyses['clusters'] = {
                                'labels': labels,
                                'keywords': cluster_keywords
                            }
                            
                            # 2. Calculer la carte stratégique
                            st.subheader("🗺️ Carte Stratégique (Strategic Diagram)")
                            
                            try:
                                # Calculer la matrice de co-occurrence (sur le DF filtré)
                                cooc_df = co_occurrence_matrix(df_clustering, min_freq=2)
                                
                                if not cooc_df.empty:
                                    # Calculer centralité et densité
                                    cooc_df = cooc_df.sort_values(by=cooc_df.columns[2], ascending=False).head(150)
                                if not cooc_df.empty:
                                    strategic_df = calculate_cluster_centrality_density(df_clustering, labels, cooc_df)   
                                    
                                    if not strategic_df.empty:
                                        # Générer le graphique
                                        output_path = get_export_path("strategic_diagram")
                                        strategic_img = plot_strategic_diagram(strategic_df, output_path)
                                        
                                        if strategic_img:
                                            st.image(strategic_img, use_container_width=True)
                                            
                                            # Afficher le tableau des résultats
                                            st.subheader("Résultats de l'analyse stratégique")
                                            display_df = strategic_df[['Cluster', 'Centrality_norm', 'Density_norm', 'Theme', 'Keywords']].copy()
                                            display_df.columns = ['Cluster', 'Centralité', 'Densité', 'Type de Thème', 'Mots-clés']
                                            st.dataframe(display_df, use_container_width=True)
                                            
                                            # Sauvegarder pour l'analyse IA
                                            strategic_data = get_strategic_analysis_data(strategic_df)
                                            st.session_state.analyses['strategic_map'] = {
                                                'data': strategic_data,
                                                'dataframe': strategic_df,
                                                'image_path': strategic_img
                                            }
                                            
                                            # Bouton pour analyser avec l'IA
                                            #if st.button("🤖 Analyser la carte stratégique avec l'IA", key="analyze_strategic"):
                                               # with st.spinner("Analyse stratégique en cours..."):
                                                 #   analysis = analyze_strategic_map(strategic_data)
                                                  #  st.session_state.analyses['strategic_analysis'] = analysis
                                                   # st.info(analysis)
                                            
                                            # Télécharger l'image
                                            with open(strategic_img, 'rb') as f:
                                                img_bytes = f.read()
                                            st.download_button(
                                                "📥 Télécharger la carte stratégique",
                                                img_bytes,
                                                "carte_strategique.png",
                                                "image/png"
                                            )
                                        else:
                                            st.warning("⚠️ Erreur lors de la génération de l'image.")
                                    else:
                                        st.warning("⚠️ Impossible de calculer la carte stratégique.")
                                else:
                                    st.warning("⚠️ Matrice de co-occurrence vide.")
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la génération de la carte stratégique: {str(e)}")
                                logger.error(f"Erreur carte stratégique: {e}")
                        else:
                            st.error("❌ Aucun texte trouvé pour le clustering.")
    # Page 4: Analyse IA
    elif page == "🤖 Analyse IA":
        st.title("🤖 Analyse par Intelligence Artificielle")
        
        st.info("💡 Cette fonctionnalité nécessite Ollama avec le modèle deepseek-r1:1.5b")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Analyser un graphe", "📈 Analyser les tendances",
            "🔍 Analyser un cluster", "💡 Recommandations", "🗺️ Carte Stratégique"
        ])
        
        with tab1:
            st.subheader("Analyse d'un graphe")
            
            if 'cooccurrence_graph' in st.session_state.analyses:
                G = st.session_state.analyses['cooccurrence_graph']
                metrics = st.session_state.analyses.get('cooccurrence_metrics', {})
                
                # Description du graphe
                graph_desc = f"""
                Type: Réseau de co-occurrence
                Nœuds: {metrics.get('nodes', 0)}
                Arêtes: {metrics.get('edges', 0)}
                Densité: {metrics.get('density', 0):.4f}
                Composantes connexes: {metrics.get('num_components', 0)}
                """
                
                st.text_area("Description du graphe", graph_desc, height=150)
                
                if st.button("🤖 Analyser avec l'IA"):
                    with st.spinner("Analyse en cours..."):
                        # Récupérer les nœuds les plus importants
                        degrees = dict(G.degree())
                        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                        top_nodes = [node for node, _ in sorted_nodes[:10]]
                        
                        analysis = analyze_graph(
                            graph_desc, 
                            graph_type="co-occurrence",
                            top_nodes=top_nodes,
                            metrics=metrics
                        )
                        st.session_state.analyses['graph_analysis'] = analysis
                        
                        st.subheader("Analyse générée")
                        st.markdown(f"<div style='background-color: #e3f2fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2196f3;'>", unsafe_allow_html=True)
                        st.write(analysis)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Veuillez d'abord générer un réseau de co-occurrence")
        
        with tab2:
            st.subheader("Analyse des tendances")
            
            if st.session_state.df_processed is not None:
                df = st.session_state.df_processed
                
                annual_df = annual_evolution(df)
                keywords_df = top_keywords(df, 20)
                
                if st.button("🤖 Analyser les tendances"):
                    with st.spinner("Analyse en cours..."):
                        annual_data = annual_df.to_dict('records')
                        top_kw = keywords_df['Keyword'].tolist()
                        
                        analysis = analyze_trends(annual_data, top_kw)
                        st.session_state.analyses['trends_analysis'] = analysis
                        
                        st.subheader("Analyse générée")
                        st.write(analysis)
            else:
                st.warning("⚠️ Veuillez d'abord importer les données")
        
        with tab3:
            st.subheader("Analyse d'un cluster")
            
            if 'clusters' in st.session_state.analyses:
                clusters = st.session_state.analyses['clusters']
                cluster_id = st.selectbox("Sélectionner un cluster", list(clusters['keywords'].keys()))
                
                if st.button("🤖 Analyser le cluster"):
                    with st.spinner("Analyse en cours..."):
                        keywords = clusters['keywords'][cluster_id]
                        
                        # Récupérer quelques abstracts du cluster
                        labels = clusters['labels']
                        df = st.session_state.df_processed
                        cluster_texts = []
                        for i, label in enumerate(labels):
                            if label == cluster_id and i < len(df):
                                abstract = df.iloc[i]['Abstract']
                                if pd.notna(abstract):
                                    cluster_texts.append(str(abstract))
                        
                        analysis = analyze_cluster(cluster_texts[:5], cluster_id, keywords)
                        
                        st.subheader(f"Analyse du cluster {cluster_id}")
                        st.write(analysis)
            else:
                st.warning("⚠️ Veuillez d'abord effectuer le clustering")
        
        with tab4:
            st.subheader("Recommandations de recherche")
            
            if st.button("🤖 Générer des recommandations"):
                with st.spinner("Génération en cours..."):
                    summary = "Analyse bibliométrique des Smart Maritime Ports"
                    recommendations = generate_research_recommendations(summary)
                    
                    st.subheader("Recommandations")
                    st.write(recommendations)
        
        with tab5:
            st.subheader("Analyse de la carte stratégique")
            
            if 'strategic_map' in st.session_state.analyses:
                strategic_data = st.session_state.analyses['strategic_map']['data']
                
                # Afficher un résumé
                st.info("💡 La carte stratégique permet d'identifier les thèmes moteurs, de base, de niche et émergents dans votre domaine de recherche.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Thèmes Moteurs", len(strategic_data.get('motor_themes', [])))
                    st.metric("Thèmes de Base", len(strategic_data.get('basic_themes', [])))
                with col2:
                    st.metric("Thèmes de Niche", len(strategic_data.get('niche_themes', [])))
                    st.metric("Thèmes Émergents", len(strategic_data.get('emerging_themes', [])))
                
                # Afficher l'image si disponible
                if 'image_path' in st.session_state.analyses['strategic_map']:
                    strategic_img = st.session_state.analyses['strategic_map']['image_path']
                    if strategic_img and os.path.exists(strategic_img):
                        st.image(strategic_img, use_container_width=True)
                
                if 'strategic_analysis' in st.session_state.analyses:
                    st.subheader("Analyse stratégique")
                    st.markdown(f"<div style='background-color: #e3f2fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2196f3;'>", unsafe_allow_html=True)
                    st.write(st.session_state.analyses['strategic_analysis'])
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    if st.button("🤖 Analyser la carte stratégique", key="analyze_strategic_tab"):
                        with st.spinner("Analyse stratégique en cours..."):
                            analysis = analyze_strategic_map(strategic_data)
                            st.session_state.analyses['strategic_analysis'] = analysis
                            
                            st.subheader("Analyse stratégique générée")
                            st.markdown(f"<div style='background-color: #e3f2fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #2196f3;'>", unsafe_allow_html=True)
                            st.write(analysis)
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Veuillez d'abord générer la carte stratégique dans l'onglet 'Réseaux et visualisations' > 'Clustering thématique'")
    
    # Page 5: Export PDF
    elif page == "📄 Export PDF":
        st.title("📄 Export PDF")
        
        st.subheader("Génération de rapport PDF")
        
        report_title = st.text_input("Titre du rapport", "Analyse Bibliométrique - Smart Maritime Ports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_graph = st.checkbox("Inclure le graphe de co-occurrence", value=True)
            include_trends = st.checkbox("Inclure l'analyse des tendances", value=True)
            include_strategic_map = st.checkbox("Inclure la carte stratégique", value=True)
        
        with col2:
            include_clusters = st.checkbox("Inclure l'analyse des clusters", value=True)
            include_ai_analysis = st.checkbox("Inclure l'analyse IA", value=True)
        
        if st.button("📄 Générer le rapport PDF complet"):
            with st.spinner("Génération du PDF complet..."):
                images_dict = {}
                tables_dict = {}
                analyses_dict = {}
                
                # Collecter toutes les visualisations disponibles
                if st.session_state.df_processed is not None:
                    df = st.session_state.df_processed
                    
                    # Graphique d'évolution annuelle
                    try:
                        annual_df = annual_evolution(df)
                        if not annual_df.empty:
                            output_path = get_export_path("evolution_annuelle")
                            img_path = plot_annual_evolution(annual_df, output_path)
                            if img_path and os.path.exists(img_path):
                                images_dict['Évolution annuelle'] = img_path
                    except:
                        pass
                    
                    # Top auteurs
                    try:
                        top_auth = top_authors(df, top_n=20)
                        if not top_auth.empty:
                            tables_dict['Top auteurs'] = top_auth
                            output_path = get_export_path("top_auteurs")
                            img_path = plot_top_items(top_auth, 'Author', 'Top 20 Auteurs', output_path)
                            if img_path and os.path.exists(img_path):
                                images_dict['Top auteurs'] = img_path
                    except:
                        pass
                    
                    # Top journaux
                    try:
                        top_jour = top_journals(df, top_n=20)
                        if not top_jour.empty:
                            tables_dict['Top journaux'] = top_jour
                            output_path = get_export_path("top_journaux")
                            img_path = plot_top_items(top_jour, 'Journal', 'Top 20 Journaux', output_path)
                            if img_path and os.path.exists(img_path):
                                images_dict['Top journaux'] = img_path
                    except:
                        pass
                    
                    # Top mots-clés
                    try:
                        top_kw = top_keywords(df, top_n=30)
                        if not top_kw.empty:
                            tables_dict['Top mots-clés'] = top_kw
                            output_path = get_export_path("top_mots_cles")
                            img_path = plot_top_items(top_kw, 'Keyword', 'Top 30 Mots-clés', output_path)
                            if img_path and os.path.exists(img_path):
                                images_dict['Top mots-clés'] = img_path
                    except:
                        pass
                
                # Réseau de co-occurrence
                if include_graph and 'cooccurrence_graph' in st.session_state.analyses:
                    try:
                        output_path = get_export_path("cooccurrence_network")
                        img_path = plot_network_graph(
                            st.session_state.analyses['cooccurrence_graph'],
                            output_path
                        )
                        if img_path and os.path.exists(img_path):
                            images_dict['Réseau de co-occurrence'] = img_path
                    except:
                        pass
                
                # WordCloud
                try:
                    if st.session_state.df_processed is not None:
                        df = st.session_state.df_processed
                        all_keywords = []
                        for kw_list in df['Keywords_list']:
                            if isinstance(kw_list, list):
                                all_keywords.extend(kw_list)
                        text = ' '.join(all_keywords)
                        if text:
                            output_path = get_export_path("wordcloud")
                            img_path = create_wordcloud(text, output_path)
                            if img_path and os.path.exists(img_path):
                                images_dict['Nuage de mots'] = img_path
                except:
                    pass
                
                # Carte stratégique
                if include_strategic_map and 'strategic_map' in st.session_state.analyses:
                    try:
                        strategic_img = st.session_state.analyses['strategic_map'].get('image_path')
                        if strategic_img and os.path.exists(strategic_img):
                            images_dict['Carte stratégique'] = strategic_img
                    except:
                        pass
                
                # Collecter les analyses IA
                if include_ai_analysis:
                    if 'graph_analysis' in st.session_state.analyses:
                        analyses_dict['Analyse du réseau de co-occurrence'] = st.session_state.analyses['graph_analysis']
                    if 'trends_analysis' in st.session_state.analyses:
                        analyses_dict['Analyse des tendances'] = st.session_state.analyses['trends_analysis']
                    if 'strategic_analysis' in st.session_state.analyses:
                        analyses_dict['Analyse de la carte stratégique'] = st.session_state.analyses['strategic_analysis']
                
                # Statistiques
                stats = {}
                if st.session_state.df_processed is not None:
                    stats = get_statistics(st.session_state.df_processed)
                
                # Générer le PDF
                if images_dict or tables_dict or analyses_dict:
                    output_pdf = get_export_path("rapport_complet.pdf")
                    
                    try:
                        pdf_path = generate_comprehensive_report(
                            report_title,
                            stats,
                            images_dict,
                            tables_dict,
                            analyses_dict,
                            output_pdf
                        )
                        
                        if pdf_path and os.path.exists(pdf_path):
                            st.success("✅ PDF généré avec succès!")
                            st.info(f"📊 Le rapport contient: {len(images_dict)} graphiques, {len(tables_dict)} tableaux, {len(analyses_dict)} analyses IA")
                            
                            with open(pdf_path, 'rb') as f:
                                pdf_bytes = f.read()
                            
                            st.download_button(
                                "📥 Télécharger le PDF complet",
                                pdf_bytes,
                                "rapport_bibliometrique_complet.pdf",
                                "application/pdf"
                            )
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la génération du PDF: {str(e)}")
                else:
                    st.warning("⚠️ Aucun élément disponible pour le PDF. Veuillez d'abord générer des analyses.")


if __name__ == "__main__":
    main()

