# 🚢 Application d'Analyse Bibliométrique - Smart Maritime Ports

Application Python complète et modulaire pour l'analyse bibliométrique de fichiers Scopus volumineux (20 000 à 100 000 lignes) centrée sur les Smart Maritime Ports.

## 📋 Table des matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Architecture](#architecture)
- [Modules](#modules)
- [Dépendances](#dépendances)
- [Configuration Ollama](#configuration-ollama)

## ✨ Fonctionnalités

### 1. Importation optimisée CSV
- Lecture de fichiers CSV volumineux avec chargement progressif par chunks
- Gestion robuste des erreurs et encodages multiples
- Détection et suppression automatique des doublons
- Nettoyage avancé des données textuelles

### 2. Analyses bibliométriques
- **Évolution annuelle** des publications
- **Top auteurs** avec statistiques détaillées
- **Top journaux** et sources
- **Top mots-clés** avec fréquences
- **Matrice de co-occurrence** des mots-clés
- **Réseau de co-auteurs** avec filtrage intelligent
- **Clustering thématique** via TF-IDF + KMeans
- **LDA Topic Modeling** pour l'identification de topics

### 3. Visualisations
- Graphiques matplotlib optimisés (PNG, SVG, PDF)
- Réseaux interactifs avec PyVis
- Nuages de mots (WordCloud) thématiques
- Graphiques de distribution et évolution temporelle

### 4. Intégration IA (Ollama)
- Résumé de clusters d'abstracts
- Analyse des tendances scientifiques
- Interprétation de graphes de co-occurrence
- Génération de recommandations de recherche
- Analyse complète automatisée

### 5. Export PDF
- Génération de rapports PDF professionnels
- Intégration d'images et analyses IA
- Mise en page propre et structurée
- Export de toutes les visualisations

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- Ollama (pour les fonctionnalités IA) - optionnel mais recommandé

### Étapes d'installation

1. **Cloner ou télécharger le projet**

2. **Créer un environnement virtuel** (recommandé)
```bash
python -m venv env
```

3. **Activer l'environnement virtuel**
   - Windows:
     ```bash
     env\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source env/bin/activate
     ```

4. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

5. **Télécharger les ressources NLTK** (automatique au premier lancement)
   - Les ressources nécessaires seront téléchargées automatiquement
   - Si nécessaire, vous pouvez les télécharger manuellement:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

6. **Installer et configurer Ollama** (pour les fonctionnalités IA)
   - Télécharger Ollama depuis [https://ollama.ai](https://ollama.ai)
   - Installer le modèle requis:
   ```bash
   ollama run deepseek-r1:1.5b
   ```
   - Vérifier que Ollama est démarré:
   ```bash
   ollama serve
   ```

## 📖 Utilisation

### Lancement de l'interface Streamlit

```bash
streamlit run app/interface.py
```

L'application sera accessible dans votre navigateur à l'adresse `http://localhost:8501`

### Workflow recommandé

1. **Importation des données**
   - Accédez à l'onglet "📊 Importation des données"
   - Téléversez votre fichier CSV Scopus
   - Configurez les options de nettoyage (suppression des doublons)
   - Cliquez sur "🔧 Préparer les données"

2. **Analyses bibliométriques**
   - Accédez à l'onglet "📈 Analyses bibliométriques"
   - Explorez les différents onglets:
     - Évolution annuelle
     - Top auteurs
     - Top journaux
     - Top mots-clés
     - Statistiques générales
   - Téléchargez les résultats en CSV si nécessaire

3. **Réseaux et visualisations**
   - Accédez à l'onglet "🕸️ Réseaux et visualisations"
   - Générez le réseau de co-occurrence des mots-clés
   - Explorez le réseau interactif
   - Créez un nuage de mots
   - Effectuez un clustering thématique
   - Téléchargez les visualisations

4. **Analyse IA**
   - Accédez à l'onglet "🤖 Analyse IA"
   - Analysez un graphe généré
   - Analysez les tendances scientifiques
   - Analysez un cluster spécifique
   - Obtenez des recommandations de recherche

5. **Export PDF**
   - Accédez à l'onglet "📄 Export PDF"
   - Configurez les options d'inclusion
   - Générez le rapport PDF complet
   - Téléchargez le PDF

### Utilisation en ligne de commande

Vous pouvez également utiliser les modules directement dans votre code Python:

```python
from app.load_data import load_scopus_csv, remove_duplicates
from app.preprocess import preprocess_dataframe
from app.bibliometry import annual_evolution, top_authors, top_keywords
from app.networks import create_cooccurrence_network
from app.visualizations import plot_annual_evolution, plot_network_graph

# Charger les données
df = load_scopus_csv("data/scopus_data.csv", chunksize=5000)
df = remove_duplicates(df)
df_processed = preprocess_dataframe(df)

# Analyses
annual_df = annual_evolution(df_processed)
top_auth = top_authors(df_processed, top_n=20)
top_kw = top_keywords(df_processed, top_n=50)

# Visualisations
plot_annual_evolution(annual_df, "exports/evolution")
```

## 🏗️ Architecture

```
Simulation_Smart_Port/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # Point d'entrée
│   ├── interface.py            # Interface Streamlit
│   ├── load_data.py            # Importation CSV optimisée
│   ├── preprocess.py           # Préprocessing et nettoyage
│   ├── bibliometry.py          # Analyses bibliométriques
│   ├── networks.py              # Création de réseaux
│   ├── nlp_models.py            # Clustering et LDA
│   ├── visualizations.py       # Génération de graphiques
│   ├── ai_analysis.py          # Intégration Ollama
│   ├── pdf_generator.py        # Génération PDF
│   └── utils.py                # Fonctions utilitaires
│
├── data/                       # Dossier pour les fichiers CSV
├── exports/                    # Dossier pour les exports (PDF, images, CSV)
│
├── requirements.txt            # Dépendances Python
└── README.md                   # Ce fichier
```

## 📦 Modules

### `load_data.py`
- `load_scopus_csv()`: Chargement optimisé de fichiers CSV volumineux
- `detect_duplicates()`: Détection de doublons
- `remove_duplicates()`: Suppression de doublons

### `preprocess.py`
- `normalize_text()`: Normalisation de texte
- `remove_stopwords()`: Suppression des stopwords
- `clean_abstract()`: Nettoyage des abstracts
- `clean_keywords()`: Nettoyage des mots-clés
- `preprocess_dataframe()`: Préprocessing complet d'un DataFrame

### `bibliometry.py`
- `annual_evolution()`: Évolution annuelle des publications
- `top_authors()`: Top auteurs
- `top_journals()`: Top journaux
- `top_keywords()`: Top mots-clés
- `co_occurrence_matrix()`: Matrice de co-occurrence
- `get_statistics()`: Statistiques générales

### `networks.py`
- `create_cooccurrence_network()`: Réseau de co-occurrence
- `create_coauthors_network()`: Réseau de co-auteurs
- `get_network_metrics()`: Métriques de réseau
- `filter_network_by_degree()`: Filtrage par degré

### `nlp_models.py`
- `tfidf_clustering()`: Clustering TF-IDF + KMeans
- `get_cluster_keywords()`: Extraction de mots-clés par cluster
- `lda_topic_modeling()`: Modélisation LDA
- `get_lda_topics()`: Extraction des topics LDA

### `visualizations.py`
- `plot_annual_evolution()`: Graphique d'évolution annuelle
- `plot_top_items()`: Graphique en barres
- `plot_network_graph()`: Visualisation de réseau
- `create_wordcloud()`: Nuage de mots
- `create_interactive_network()`: Réseau interactif PyVis
- `plot_cluster_distribution()`: Distribution des clusters

### `ai_analysis.py`
- `ask_ai()`: Fonction principale de communication avec Ollama
- `analyze_cluster()`: Analyse d'un cluster
- `analyze_trends()`: Analyse des tendances
- `analyze_graph()`: Analyse d'un graphe
- `generate_research_recommendations()`: Recommandations de recherche
- `generate_comprehensive_analysis()`: Analyse complète

### `pdf_generator.py`
- `generate_pdf()`: Génération d'un PDF simple
- `generate_comprehensive_report()`: Génération d'un rapport complet

### `utils.py`
- Fonctions utilitaires diverses (gestion de fichiers, normalisation, etc.)

## 🔧 Dépendances

Les principales dépendances sont listées dans `requirements.txt`:

- **pandas, numpy**: Traitement de données
- **streamlit**: Interface utilisateur
- **matplotlib, seaborn, plotly**: Visualisations
- **networkx, pyvis**: Réseaux et graphes
- **nltk, scikit-learn, gensim**: NLP et machine learning
- **wordcloud**: Nuages de mots
- **reportlab, fpdf**: Génération PDF
- **requests**: Communication avec Ollama

## 🤖 Configuration Ollama

### Installation d'Ollama

1. Télécharger depuis [https://ollama.ai](https://ollama.ai)
2. Installer selon votre système d'exploitation
3. Démarrer le serveur Ollama:
   ```bash
   ollama serve
   ```

### Installation du modèle

```bash
ollama run deepseek-r1:1.5b
```

### Vérification

Pour vérifier que tout fonctionne:

```python
from app.ai_analysis import ask_ai
response = ask_ai("Bonjour, peux-tu te présenter?")
print(response)
```

### Configuration personnalisée

Si vous utilisez un autre modèle ou endpoint, modifiez les constantes dans `app/ai_analysis.py`:

```python
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "votre-modele"
```

## 📊 Format des données d'entrée

Le fichier CSV Scopus doit contenir au minimum les colonnes suivantes:

- `Title`: Titre de la publication
- `Abstract`: Résumé
- `Author Keywords`: Mots-clés (séparés par `;` ou `,`)
- `Authors`: Liste des auteurs (séparés par `;`)
- `Year`: Année de publication
- `Source title`: Titre de la source/journal

## 🐛 Dépannage

### Erreur de chargement CSV
- Vérifiez l'encodage du fichier (UTF-8 recommandé)
- Vérifiez que les colonnes requises sont présentes
- Réduisez la taille des chunks si nécessaire

### Erreur Ollama
- Vérifiez que Ollama est démarré: `ollama serve`
- Vérifiez que le modèle est installé: `ollama list`
- Vérifiez la connexion: `curl http://localhost:11434/api/generate`

### Erreur de mémoire
- Réduisez la taille des chunks lors du chargement
- Filtrez les données avant le traitement
- Utilisez le filtrage des réseaux pour réduire le nombre de nœuds

## 📝 Notes

- Les fichiers générés sont sauvegardés dans le dossier `exports/`
- Les visualisations sont générées en haute résolution (300 DPI)
- Les analyses IA peuvent prendre quelques secondes selon la complexité
- Pour de très gros fichiers (>100k lignes), le traitement peut être long

## 📄 Licence

Ce projet est fourni tel quel pour usage académique et de recherche.

## 👥 Contribution

Les contributions sont les bienvenues! N'hésitez pas à ouvrir une issue ou une pull request.

## 📧 Support

Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt du projet.

---

**Bonnes analyses bibliométriques! 🚢📊**

