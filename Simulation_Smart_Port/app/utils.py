"""
Fonctions utilitaires pour l'application bibliométrique
"""

import os
import re
from pathlib import Path
from typing import List, Optional
import unicodedata


def ensure_dir(path: str) -> None:
    """Crée un répertoire s'il n'existe pas."""
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_filename(filename: str) -> str:
    """Nettoie un nom de fichier pour éviter les caractères invalides."""
    # Supprime les caractères invalides
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limite la longueur
    if len(filename) > 200:
        filename = filename[:200]
    return filename


def normalize_unicode(text: str) -> str:
    """Normalise les caractères Unicode."""
    if not isinstance(text, str):
        return ""
    # Normalise en NFC (Canonical Composition)
    text = unicodedata.normalize('NFC', text)
    return text


def safe_str(value) -> str:
    """Convertit une valeur en string de manière sécurisée."""
    if value is None:
        return ""
    if isinstance(value, float) and (value != value):  # NaN check
        return ""
    return str(value).strip()


def get_export_path(filename: str) -> str:
    """Retourne le chemin complet pour un fichier d'export."""
    ensure_dir("exports")
    return os.path.join("exports", clean_filename(filename))


def format_number(num: int) -> str:
    """Formate un nombre avec des séparateurs de milliers."""
    return f"{num:,}".replace(",", " ")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Tronque un texte à une longueur maximale."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


