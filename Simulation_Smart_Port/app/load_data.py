"""
Module d'importation optimisée pour fichiers CSV Scopus volumineux
"""

import pandas as pd
import os
from typing import Optional, List
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_scopus_csv(
    file_path: str,
    chunksize: Optional[int] = 5000,
    columns_to_keep: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Charge un fichier CSV Scopus de manière optimisée.
    
    Args:
        file_path: Chemin vers le fichier CSV
        chunksize: Taille des chunks pour chargement progressif (None = tout charger)
        columns_to_keep: Liste des colonnes à conserver (None = toutes)
    
    Returns:
        DataFrame pandas avec les données nettoyées
    """
    if columns_to_keep is None:
        columns_to_keep = [
            "Title",
            "Abstract",
            "Author Keywords",
            "Authors",
            "Year",
            "Source title"
        ]
    
    logger.info(f"Chargement du fichier: {file_path}")
    
    # Vérification de l'existence du fichier
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    # Lecture avec gestion des erreurs
    try:
        if chunksize:
            # Chargement par chunks pour fichiers volumineux
            chunks = []
            total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
            
            logger.info(f"Chargement progressif par chunks de {chunksize} lignes...")
            
            # Gestion de la compatibilité pandas
            # Note: low_memory n'est pas supporté avec engine="python"
            try:
                # pandas >= 1.3.0
                csv_reader = pd.read_csv(
                    file_path,
                    chunksize=chunksize,
                    dtype=str,
                    on_bad_lines="skip",
                    encoding="utf-8",
                    engine="python"
                )
            except TypeError:
                # pandas < 1.3.0
                csv_reader = pd.read_csv(
                    file_path,
                    chunksize=chunksize,
                    dtype=str,
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    encoding="utf-8",
                    engine="python"
                )
            
            for chunk in tqdm(
                csv_reader,
                total=(total_rows // chunksize) + 1,
                desc="Chargement"
            ):
                # Filtrer les colonnes si nécessaire
                available_cols = [col for col in columns_to_keep if col in chunk.columns]
                if available_cols:
                    chunk = chunk[available_cols]
                    chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(f"Chargement terminé: {len(df)} lignes chargées")
            
        else:
            # Chargement direct pour fichiers plus petits
            logger.info("Chargement direct du fichier...")
            try:
                df = pd.read_csv(
                    file_path,
                    dtype=str,
                    on_bad_lines="skip",
                    encoding="utf-8",
                    engine="python"
                )
            except TypeError:
                df = pd.read_csv(
                    file_path,
                    dtype=str,
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    encoding="utf-8",
                    engine="python"
                )
            
            # Filtrer les colonnes si nécessaire
            available_cols = [col for col in columns_to_keep if col in df.columns]
            if available_cols:
                df = df[available_cols]
            
            logger.info(f"Chargement terminé: {len(df)} lignes chargées")
        
        # Nettoyage initial
        df = df.dropna(how='all')  # Supprime les lignes complètement vides
        
        return df
        
    except UnicodeDecodeError:
        # Essai avec d'autres encodages
        logger.warning("Erreur d'encodage UTF-8, tentative avec latin-1...")
        try:
            try:
                df = pd.read_csv(
                    file_path,
                    dtype=str,
                    on_bad_lines="skip",
                    encoding="latin-1",
                    engine="python"
                )
            except TypeError:
                df = pd.read_csv(
                    file_path,
                    dtype=str,
                    error_bad_lines=False,
                    warn_bad_lines=False,
                    encoding="latin-1",
                    engine="python"
                )
            available_cols = [col for col in columns_to_keep if col in df.columns]
            if available_cols:
                df = df[available_cols]
            return df.dropna(how='all')
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            raise


def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Détecte et retourne les doublons dans le DataFrame.
    
    Args:
        df: DataFrame à analyser
        subset: Colonnes à utiliser pour détecter les doublons (None = toutes)
    
    Returns:
        DataFrame contenant uniquement les doublons
    """
    if subset is None:
        subset = ["Title", "Authors", "Year"]
    
    # Ne garder que les colonnes disponibles
    subset = [col for col in subset if col in df.columns]
    
    if not subset:
        logger.warning("Aucune colonne disponible pour la détection de doublons")
        return pd.DataFrame()
    
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    logger.info(f"{len(duplicates)} doublons détectés")
    
    return duplicates


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Supprime les doublons du DataFrame.
    
    Args:
        df: DataFrame à nettoyer
        subset: Colonnes à utiliser pour détecter les doublons
    
    Returns:
        DataFrame sans doublons
    """
    initial_count = len(df)
    
    if subset is None:
        subset = ["Title", "Authors", "Year"]
    
    subset = [col for col in subset if col in df.columns]
    
    if not subset:
        logger.warning("Aucune colonne disponible pour la suppression de doublons")
        return df
    
    df_cleaned = df.drop_duplicates(subset=subset, keep='first')
    removed_count = initial_count - len(df_cleaned)
    
    logger.info(f"{removed_count} doublons supprimés ({initial_count} -> {len(df_cleaned)} lignes)")
    
    return df_cleaned

