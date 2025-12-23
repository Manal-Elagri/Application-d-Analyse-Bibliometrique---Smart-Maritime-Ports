"""
Point d'entrée principal de l'application
Peut être utilisé pour des scripts en ligne de commande
"""

import sys
import os

# Ajouter le répertoire parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("Application d'analyse bibliométrique - Smart Maritime Ports")
    print("Pour lancer l'interface Streamlit, utilisez:")
    print("  streamlit run app/interface.py")
    print("\nOu importez les modules directement dans votre code Python.")

