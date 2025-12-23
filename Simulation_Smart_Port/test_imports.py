"""
Script de test pour vérifier que tous les imports fonctionnent
"""

import sys
import os

# Ajouter le répertoire app au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

print("Test des imports...")
print("=" * 50)

try:
    print("✓ Import streamlit...")
    import streamlit as st
    
    print("✓ Import pandas...")
    import pandas as pd
    
    print("✓ Import modules de l'application...")
    from app.load_data import load_scopus_csv, remove_duplicates
    from app.preprocess import preprocess_dataframe
    from app.bibliometry import annual_evolution, top_authors, top_keywords
    from app.networks import create_cooccurrence_network
    from app.visualizations import plot_annual_evolution
    from app.ai_analysis import ask_ai
    from app.pdf_generator import generate_pdf
    from app.utils import get_export_path
    
    print("\n" + "=" * 50)
    print("✅ Tous les imports sont réussis!")
    print("=" * 50)
    print("\nVous pouvez maintenant lancer Streamlit avec:")
    print("  streamlit run app/interface.py")
    
except ImportError as e:
    print(f"\n❌ Erreur d'import: {e}")
    print("\nVérifiez que toutes les dépendances sont installées:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Erreur: {e}")
    sys.exit(1)

