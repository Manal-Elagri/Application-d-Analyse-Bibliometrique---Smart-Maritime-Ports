"""
Module pour la génération de rapports PDF améliorés
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NumberedCanvas:
    """Classe pour ajouter des numéros de page"""
    def __init__(self, canvas, doc):
        self.canvas = canvas
        self.doc = doc
        
    def draw_page_number(self):
        self.canvas.saveState()
        self.canvas.setFont('Helvetica', 9)
        page_num = self.canvas.getPageNumber()
        text = f"Page {page_num}"
        self.canvas.drawRightString(7.5*inch, 0.75*inch, text)
        self.canvas.restoreState()


def generate_comprehensive_report(
    title: str,
    stats: dict,
    images: Dict[str, str],
    tables: Dict[str, pd.DataFrame],
    analyses: Dict[str, str],
    output_path: str = "exports/comprehensive_report.pdf"
) -> str:
    """
    Génère un rapport PDF complet et professionnel avec tous les éléments.
    
    Args:
        title: Titre du rapport
        stats: Statistiques générales
        images: Dictionnaire {nom: chemin_image}
        tables: Dictionnaire {nom: DataFrame}
        analyses: Dictionnaire {nom: texte_analyse}
        output_path: Chemin de sortie
    
    Returns:
        Chemin du fichier PDF généré
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    logger.info(f"Génération du rapport PDF complet: {output_path}")
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=60,
        bottomMargin=60
    )
    
    styles = getSampleStyleSheet()
    
    # Styles personnalisés améliorés
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1a237e'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor('#616161'),
        alignment=TA_CENTER,
        spaceAfter=30,
        fontStyle='italic'
    )
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#283593'),
        spaceAfter=15,
        spaceBefore=20,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#3f51b5'),
        borderPadding=8,
        backColor=colors.HexColor('#e3f2fd')
    )
    
    subsection_style = ParagraphStyle(
        'Subsection',
        parent=styles['Heading3'],
        fontSize=13,
        textColor=colors.HexColor('#3949ab'),
        spaceAfter=10,
        spaceBefore=15,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#212121'),
        alignment=TA_JUSTIFY,
        spaceAfter=12,
        leading=14
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=20,
        bulletIndent=10
    )
    
    story = []
    
    # Page de titre
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Rapport d'Analyse Bibliométrique", subtitle_style))
    story.append(Spacer(1, 0.3*inch))
    date_str = datetime.now().strftime("%d %B %Y")
    story.append(Paragraph(f"Généré le {date_str}", styles['Normal']))
    story.append(PageBreak())
    
    # Table des matières
    story.append(Paragraph("Table des matières", section_style))
    toc_items = []
    if stats:
        toc_items.append("1. Statistiques générales")
    if images:
        toc_items.append("2. Visualisations")
    if tables:
        toc_items.append("3. Tableaux de données")
    if analyses:
        toc_items.append("4. Analyses par Intelligence Artificielle")
    toc_items.append("5. Conclusion")
    
    for item in toc_items:
        story.append(Paragraph(f"• {item}", bullet_style))
    story.append(PageBreak())
    
    # Section 1: Statistiques générales
    if stats:
        story.append(Paragraph("1. Statistiques générales", section_style))
        
        # Créer un tableau pour les statistiques
        stats_data = [['Métrique', 'Valeur']]
        for key, value in stats.items():
            stats_data.append([key.replace('_', ' ').title(), str(value)])
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 0.3*inch))
    
    # Section 2: Visualisations
    if images:
        story.append(PageBreak())
        story.append(Paragraph("2. Visualisations", section_style))
        
        for img_name, img_path in images.items():
            if os.path.exists(img_path):
                try:
                    story.append(Paragraph(img_name.replace('_', ' ').title(), subsection_style))
                    
                    # Redimensionner l'image pour s'adapter à la page
                    try:
                        from PIL import Image as PILImage
                        img_pil = PILImage.open(img_path)
                        img_width, img_height = img_pil.size
                        aspect_ratio = img_height / img_width
                        
                        max_width = 6.5 * inch
                        max_height = 7 * inch
                        
                        if aspect_ratio > (max_height / max_width):
                            # Image plus haute que large
                            display_height = max_height
                            display_width = display_height / aspect_ratio
                        else:
                            # Image plus large que haute
                            display_width = max_width
                            display_height = display_width * aspect_ratio
                        
                        img = Image(img_path, width=display_width, height=display_height)
                    except (ImportError, Exception):
                        # Fallback: taille par défaut
                        img = Image(img_path, width=6*inch, height=6*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.3*inch))
                    
                except Exception as e:
                    logger.warning(f"Impossible d'ajouter l'image {img_path}: {e}")
                    story.append(Paragraph(f"[Image non disponible: {img_name}]", body_style))
    
    # Section 3: Tableaux
    if tables:
        story.append(PageBreak())
        story.append(Paragraph("3. Tableaux de données", section_style))
        
        for table_name, df in tables.items():
            if df is not None and not df.empty:
                story.append(Paragraph(table_name.replace('_', ' ').title(), subsection_style))
                
                # Limiter le nombre de lignes pour le PDF
                df_display = df.head(50)  # Maximum 50 lignes
                
                # Préparer les données pour le tableau
                table_data = [df_display.columns.tolist()]
                for _, row in df_display.iterrows():
                    table_data.append([str(val)[:50] for val in row.values])  # Limiter la longueur
                
                # Créer le tableau
                pdf_table = Table(table_data)
                pdf_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
                ]))
                
                story.append(pdf_table)
                if len(df) > 50:
                    story.append(Paragraph(f"<i>(Affichage des 50 premières lignes sur {len(df)} au total)</i>", body_style))
                story.append(Spacer(1, 0.3*inch))
    
    # Section 4: Analyses IA
    if analyses:
        story.append(PageBreak())
        story.append(Paragraph("4. Analyses par Intelligence Artificielle", section_style))
        
        for analysis_name, analysis_text in analyses.items():
            if analysis_text:
                story.append(Paragraph(analysis_name.replace('_', ' ').title(), subsection_style))
                
                # Diviser le texte en paragraphes
                paragraphs = analysis_text.split('\n\n')
                for para in paragraphs:
                    if para.strip():
                        # Nettoyer le texte
                        para_clean = para.strip().replace('<', '&lt;').replace('>', '&gt;')
                        # Formater les listes à puces
                        if para_clean.startswith('-') or para_clean.startswith('•'):
                            story.append(Paragraph(para_clean, bullet_style))
                        else:
                            story.append(Paragraph(para_clean, body_style))
                
                story.append(Spacer(1, 0.2*inch))
    
    # Section 5: Conclusion
    story.append(PageBreak())
    story.append(Paragraph("5. Conclusion", section_style))
    conclusion_text = """
    Ce rapport présente une analyse bibliométrique complète des publications scientifiques 
    sur les Smart Maritime Ports. Les visualisations, statistiques et analyses par intelligence 
    artificielle fournissent une vue d'ensemble approfondie de l'état actuel de la recherche 
    dans ce domaine.
    
    Les résultats peuvent être utilisés pour identifier les tendances émergentes, les domaines 
    de recherche prometteurs et les opportunités de collaboration scientifique.
    """
    story.append(Paragraph(conclusion_text, body_style))
    
    # Générer le PDF
    try:
        doc.build(story)
        logger.info(f"Rapport PDF complet généré: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Erreur lors de la génération du PDF: {e}")
        raise


def generate_pdf(
    title: str,
    plot_image_path: Optional[str] = None,
    ai_analysis_text: str = "",
    output_path: str = "exports/report.pdf",
    additional_images: Optional[list] = None
) -> str:
    """
    Génère un PDF simple avec titre, images et texte d'analyse IA.
    (Fonction de compatibilité)
    """
    images_dict = {}
    if plot_image_path:
        images_dict['Graphique principal'] = plot_image_path
    if additional_images:
        for i, img_path in enumerate(additional_images):
            images_dict[f'Graphique {i+1}'] = img_path
    
    analyses_dict = {}
    if ai_analysis_text:
        analyses_dict['Analyse principale'] = ai_analysis_text
    
    return generate_comprehensive_report(
        title=title,
        stats={},
        images=images_dict,
        tables={},
        analyses=analyses_dict,
        output_path=output_path
    )
