import re
import fitz
import streamlit as st
import spacy
import csv
import nltk
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de la page
st.set_page_config(
    page_title="RH Analytics - Analyse de CV",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer un style CSS personnalisé
st.markdown("""
<style>
    .main-header {color:#1E88E5; font-size:42px; font-weight:bold; text-align:center; margin-bottom:30px;}
    .sub-header {color:#0D47A1; font-size:28px; font-weight:bold; margin-top:30px; margin-bottom:20px;}
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {font-size:32px; font-weight:bold; color:#1E88E5;}
    .metric-label {font-size:16px; color:#555;}
    .success-box {background-color: #E8F5E9; padding:15px; border-radius:5px; border-left:5px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)

# Téléchargement des ressources nécessaires
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')

download_nltk_resources()

# Charger les modèles SpaCy
@st.cache_resource
def load_spacy_models():
    models = {}
    
    try:
        models['en'] = spacy.load('en_core_web_sm')
    except:
        st.warning("Modèle anglais (en_core_web_sm) non disponible. Installation en cours...")
        spacy.cli.download('en_core_web_sm')
        models['en'] = spacy.load('en_core_web_sm')
    
    try:
        models['fr'] = spacy.load('fr_core_news_md')
    except:
        st.warning("Modèle français (fr_core_news_md) non disponible. Installation en cours...")
        spacy.cli.download('fr_core_news_md')
        models['fr'] = spacy.load('fr_core_news_md')
    
    # Tenter de charger le modèle personnalisé
    try:
        models['custom'] = spacy.load('/content/drive/MyDrive/ModelNER_projet')
        st.success("✅ Modèle personnalisé chargé avec succès!")
    except:
        models['custom'] = None
        st.warning("⚠️ Modèle personnalisé non disponible. Utilisation des modèles standards.")
    
    # Modèles de compétences
    try:
        models['en_skills'] = spacy.load('en_skills_model')
    except:
        models['en_skills'] = None
    
    try:
        models['fr_skills'] = spacy.load('fr_skills_model')
    except:
        models['fr_skills'] = None
    
    return models

# Extraction d'entités avec modèle combiné
def extract_entities_combined(text, models, lang_code):
    base_model = models['fr'] if lang_code == 'fr' else models['en']
    base_doc = base_model(text)
    
    # Si le modèle personnalisé est disponible, l'utiliser aussi
    custom_entities = []
    if models['custom']:
        custom_doc = models['custom'](text)
        custom_entities = [(ent.text, ent.label_) for ent in custom_doc.ents]
    
    # Combiner les entités
    combined_entities = [(ent.text, ent.label_) for ent in base_doc.ents]
    combined_entities.extend(custom_entities)
    
    return base_doc, combined_entities

# Extraction du nom
def extract_name(doc, entities):
    for text, label in entities:
        if label in ('PERSON', 'PER'):
            tokens = text.split()
            if len(tokens) >= 2:
                return tokens[0], ' '.join(tokens[1:])
    return "", ""

# Extraction de l'email
def extract_email(doc):
    matcher = spacy.matcher.Matcher(doc.vocab)
    matcher.add('EMAIL', [[{'LIKE_EMAIL': True}]])
    for _, start, end in matcher(doc):
        return doc[start:end].text
    return ""

# Extraction du numéro de téléphone
def extract_phone(text, lang_code):
    if lang_code == 'fr':
        patterns = [
            r"\b(?:\+33\s*(?:\(0\)\s*)?|0)[1-9](?:[\s.\-/]*\d{2}){4}\b"
        ]
    else:
        patterns = [
            r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?){2}\d{4}\b"
        ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group()
    
    # Pattern générique au cas où
    fb = re.search(r"\b[+\d][\d\s().-]{6,}\d\b", text)
    return fb.group() if fb else ""

# Extraction de sections génériques
def extract_section(text, header, lang_code='fr'):
    # Tenir compte des variations d'accentuation et de casse
    header_pattern = f"{header}"
    pattern = rf"{header_pattern}\s*[:\n](.*?)(?=\n[A-ZÉÈÀÂÔÛÊÇ][a-zéèàâêîôûç]*[:\n]|$)"
    m = re.search(pattern, text, re.S | re.IGNORECASE)
    
    if not m:
        # Essayer avec un pattern plus souple
        pattern = rf"{header_pattern}.*?\n(.*?)(?=\n\s*[A-ZÉÈÀÂÔÛÊÇ][a-zéèàâêîôûç]*|$)"
        m = re.search(pattern, text, re.S | re.IGNORECASE)
    
    return m.group(1).strip() if m else ""

# Extraction du profil
def extract_profile(text, lang_code):
    headers = ['Profil', 'À propos', 'About', 'Summary', 'Résumé', 'Présentation']
    for header in headers:
        section = extract_section(text, header, lang_code)
        if section:
            return section
    
    # Si aucune section trouvée, essayer d'extraire les premiers paragraphes
    paragraphs = text.split('\n\n')
    if len(paragraphs) > 1:
        for p in paragraphs[:3]:  # Regarder les 3 premiers paragraphes
            if len(p.split()) > 15:  # Au moins 15 mots
                return p
    
    return ""

# Extraction des compétences
def extract_skills(doc, text, entities, models, lang_code):
    # Extraire la section compétences
    headers = ['Compétences', 'Skills', 'Expertise', 'Technical Skills']
    section = ""
    for header in headers:
        section = extract_section(text, header, lang_code)
        if section:
            break
    
    # Extraire les éléments de la section
    items = re.split(r"[\n,•]+", section)
    skills = [i.strip() for i in items if i.strip()]
    
    # Utiliser le modèle de compétences si disponible
    skills_model_key = 'fr_skills' if lang_code == 'fr' else 'en_skills'
    if models.get(skills_model_key):
        skills_doc = models[skills_model_key](text)
        for ent in skills_doc.ents:
            if ent.label_ == 'SKILL':
                skills.append(ent.text)
    
    # Utiliser le modèle personnalisé pour les compétences
    if models['custom']:
        custom_doc = models['custom'](text)
        for ent in custom_doc.ents:
            if ent.label_ in ['SKILL', 'COMPETENCE', 'TECHNOLOGY']:
                skills.append(ent.text)
    
    # Retirer les doublons et les éléments vides
    unique_skills = []
    for skill in skills:
        cleaned = skill.strip()
        if cleaned and cleaned not in unique_skills:
            unique_skills.append(cleaned)
    
    return unique_skills

# Extraction des formations
def extract_education(text, lang_code):
    headers = ['Formation', 'Education', 'Études', 'Parcours académique']
    section = ""
    for header in headers:
        section = extract_section(text, header, lang_code)
        if section:
            break
    
    lines = [l.strip() for l in section.splitlines() if l.strip()]
    
    educations = []
    current_education = ""
    
    for line in lines:
        # Détecter si c'est une nouvelle entrée (date ou diplôme)
        if re.search(r'\b\d{4}\b', line) or any(edu_term in line.lower() for edu_term in ['diplôme', 'master', 'licence', 'bac', 'certificat', 'brevet', 'ingénieur', 'bachelor', 'degree']):
            if current_education:
                educations.append(current_education)
            current_education = line
        else:
            if current_education:
                current_education += " | " + line
            else:
                current_education = line
    
    if current_education:
        educations.append(current_education)
    
    return educations

# Extraction des certifications
def extract_certifications(text, lang_code):
    headers = ['Certifications', 'Certificats', 'Accréditations']
    section = ""
    for header in headers:
        section = extract_section(text, header, lang_code)
        if section:
            break
    
    items = re.split(r"[\n•]+", section)
    return [item.strip() for item in items if item.strip()]

# Extraction des langues
def extract_languages(text, lang_code):
    headers = ['Langues', 'Languages', 'Compétences linguistiques']
    section = ""
    for header in headers:
        section = extract_section(text, header, lang_code)
        if section:
            break
    
    items = re.split(r"[\n,•]+", section)
    languages = []
    for item in items:
        lang = item.strip(' •\u2022').strip()
        if lang:
            languages.append(lang)
    
    return languages

# Détection du niveau d'expérience
def extract_experience_level(doc, text, entities):
    # Chercher d'abord des mentions directes d'années d'expérience
    years_pattern = r"(\d+)[\s+](?:ans|années|an|year|years)[\s+](?:d'expérience|d'expériences|experience)"
    years_match = re.search(years_pattern, text, re.IGNORECASE)
    
    if years_match:
        years = int(years_match.group(1))
        if years < 3:
            return "Junior"
        elif years < 8:
            return "Intermédiaire"
        else:
            return "Senior"
    
    # Rechercher des verbes liés au leadership
    leadership_verbs = ['diriger', 'manager', 'superviser', 'lead', 'manage', 'direct', 'oversee']
    mid_verbs = ['développer', 'concevoir', 'analyser', 'implémenter', 'develop', 'design', 'analyze', 'implement']
    
    verbs = [t.text.lower() for t in doc if t.pos_ == 'VERB']
    
    if any(lv in ' '.join(verbs).lower() for lv in leadership_verbs):
        return "Senior"
    elif any(mv in ' '.join(verbs).lower() for mv in mid_verbs):
        return "Intermédiaire"
    
    return "Junior"

# Calcul du score de correspondance
def compute_matching_score(text, job_keywords=None):
    # Liste par défaut de mots-clés
    default_keywords = ['python', 'sql', 'data', 'machine learning', 'analyse', 'project']
    
    # Utiliser les mots-clés fournis ou les mots-clés par défaut
    keywords = job_keywords if job_keywords else default_keywords
    
    # Créer un vecteur pour le texte et les mots-clés
    vectorizer = CountVectorizer()
    try:
        vectors = vectorizer.fit_transform([text.lower(), ' '.join(keywords).lower()])
        similarity = cosine_similarity(vectors)
        return round(similarity[0][1] * 100, 2)
    except:
        # En cas d'erreur, retourner un score par défaut
        return 50.0

# Traitement des fichiers PDF
def process_pdf_files(uploaded_files, models, lang, job_keywords=None):
    lang_code = 'fr' if lang == 'French' else 'en'
    results = []
    
    with st.spinner('Traitement des CVs en cours...'):
        progress_bar = st.progress(0)
        
        for i, uploaded in enumerate(uploaded_files):
            try:
                # Lecture du PDF
                pdf = fitz.open(stream=uploaded.read(), filetype='pdf')
                text = '\n'.join(page.get_text() for page in pdf)
                
                # Extraction des entités combinées
                doc, entities = extract_entities_combined(text, models, lang_code)
                
                # Extraction des informations
                first_name, last_name = extract_name(doc, entities)
                profile = extract_profile(text, lang_code)
                skills = extract_skills(doc, text, entities, models, lang_code)
                education = extract_education(text, lang_code)
                certifications = extract_certifications(text, lang_code)
                languages = extract_languages(text, lang_code)
                experience_level = extract_experience_level(doc, text, entities)
                
                # Calcul du score de correspondance
                score = compute_matching_score(text, job_keywords)
                
                # Ajout des résultats
                results.append({
                    'Nom du fichier': uploaded.name,
                    'Prénom' if lang_code == 'fr' else 'First Name': first_name,
                    'Nom' if lang_code == 'fr' else 'Last Name': last_name,
                    'Profil' if lang_code == 'fr' else 'Profile': profile[:200] + '...' if len(profile) > 200 else profile,
                    'Email': extract_email(doc),
                    'Téléphone' if lang_code == 'fr' else 'Phone': extract_phone(text, lang_code),
                    'Compétences' if lang_code == 'fr' else 'Skills': skills,
                    'Formations' if lang_code == 'fr' else 'Education': education,
                    'Certifications': certifications,
                    'Langues' if lang_code == 'fr' else 'Languages': languages,
                    'Niveau expérience' if lang_code == 'fr' else 'Experience Level': experience_level,
                    'Score': score
                })
                
                # Mise à jour de la barre de progression
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            except Exception as e:
                st.error(f"Erreur lors du traitement de {uploaded.name}: {str(e)}")
    
    progress_bar.empty()
    return results

# Interface utilisateur principale
def main():
    # Chargement des modèles
    models = load_spacy_models()
    
    # En-tête
    st.markdown("<div class='main-header'>📊 RH Analytics - Analyse intelligente de CV</div>", unsafe_allow_html=True)
    
    # Menu latéral
    with st.sidebar:
        st.markdown("## ⚙️ Paramètres")
        
        # Sélection de la langue
        lang = st.selectbox('Langue des CVs', ['French', 'English'], index=0)
        
        # Section des mots-clés de poste
        st.markdown("### 🎯 Mots-clés du poste")
        default_keywords = "python, sql, data, machine learning, analyse, project"
        job_keywords = st.text_area(
            "Entrez les mots-clés du poste (séparés par des virgules)",
            value=default_keywords,
            help="Ces mots-clés seront utilisés pour calculer un score de correspondance"
        )
        job_keywords_list = [k.strip() for k in job_keywords.split(',') if k.strip()]
        
        # Section de téléchargement
        st.markdown("### 📤 Téléchargement des CVs")
        uploaded_files = st.file_uploader(
            "Téléversez les CVs (format PDF)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Vous pouvez télécharger plusieurs fichiers PDF"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} fichier(s) téléversé(s)")
    
    # Corps principal
    if uploaded_files:
        # Traitement des fichiers
        results = process_pdf_files(uploaded_files, models, lang, job_keywords_list)
        
        if results:
            # Convertir les résultats en DataFrame
            df = pd.DataFrame(results)
            
            # Création de tabs pour l'affichage des résultats
            tab1, tab2, tab3 = st.tabs(["📋 Vue d'ensemble", "📊 Statistiques", "👤 Profils détaillés"])
            
            with tab1:
                st.markdown("<div class='sub-header'>📋 Aperçu des candidats</div>", unsafe_allow_html=True)
                
                # Colonnes importantes pour la vue d'ensemble - selon les spécifications demandées
                overview_cols = [
                    'Nom' if lang == 'French' else 'Last Name',
                    'Prénom' if lang == 'French' else 'First Name',
                    'Email',
                    'Téléphone' if lang == 'French' else 'Phone',
                    'Profil' if lang == 'French' else 'Profile',
                    'Formations' if lang == 'French' else 'Education',
                    'Compétences' if lang == 'French' else 'Skills',
                    'Langues' if lang == 'French' else 'Languages',
                    'Niveau expérience' if lang == 'French' else 'Experience Level'
                ]
                
                # Affichage du DataFrame filtré
                st.dataframe(df[overview_cols], use_container_width=True)
                
                # Section de téléchargement des résultats
                st.markdown("### 📥 Télécharger les résultats")
                sep = ';' if lang == 'French' else ','
                csv_data = df.to_csv(index=False, sep=sep)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv_data,
                        file_name=f'resultats_cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    )
                with col2:
                    # Format Excel pour préserver les listes
                    excel_filename = f'resultats_cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                    
                    # Création d'un buffer binaire en mémoire pour Excel
                    buffer = BytesIO()
                    
                    # Sauvegarde du dataframe en Excel
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, index=False, sheet_name='Résultats')
                    
                    # Préparation des données pour le téléchargement
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=buffer,
                        file_name=excel_filename,
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
            
            with tab2:
                st.markdown("<div class='sub-header'>📊 Statistiques et insights</div>", unsafe_allow_html=True)
                
                # Métriques clés
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{len(df)}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Candidats analysés</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{df['Score'].mean():.1f}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Score moyen de correspondance</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    top_score = df['Score'].max()
                    top_candidates = df[df['Score'] == top_score].shape[0]
                    st.markdown(f"<div class='metric-value'>{top_score:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-label'>Score le plus élevé ({top_candidates} candidat{'s' if top_candidates > 1 else ''})</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Graphiques
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution des scores
                    fig_scores = px.histogram(
                        df, 
                        x='Score',
                        nbins=10,
                        title="Distribution des scores",
                        labels={'Score': 'Score de correspondance (%)'},
                        color_discrete_sequence=['#1E88E5']
                    )
                    fig_scores.update_layout(bargap=0.1)
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                with col2:
                    # Répartition des niveaux d'expérience
                    exp_col = 'Niveau expérience' if lang == 'French' else 'Experience Level'
                    exp_counts = df[exp_col].value_counts().reset_index()
                    exp_counts.columns = ['Niveau', 'Nombre']
                    
                    fig_exp = px.pie(
                        exp_counts,
                        values='Nombre',
                        names='Niveau',
                        title="Répartition des niveaux d'expérience",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_exp.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_exp, use_container_width=True)
                
                # Analyse des compétences
                st.markdown("#### 💡 Analyse des compétences")
                skills_col = 'Compétences' if lang == 'French' else 'Skills'
                
                # Extraction de toutes les compétences
                all_skills = []
                for skills_list in df[skills_col]:
                    all_skills.extend(skills_list)
                
                # Comptage des compétences
                skill_counts = pd.Series(all_skills).value_counts().head(15)
                
                # Graphique des compétences les plus fréquentes
                fig_skills = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Top 15 des compétences les plus mentionnées",
                    labels={'x': 'Nombre de mentions', 'y': 'Compétence'},
                    color=skill_counts.values,
                    color_continuous_scale=px.colors.sequential.Blues
                )
                st.plotly_chart(fig_skills, use_container_width=True)
                
                # Suggestions basées sur l'analyse
                st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                st.markdown("#### 🔍 Insights pour le recrutement")
                
                top_skills = skill_counts.head(5).index.tolist()
                top_skills_str = ", ".join(top_skills)
                
                st.markdown(f"""
                * Les compétences les plus répandues parmi les candidats sont: **{top_skills_str}**
                * {df[exp_col].value_counts().idxmax()} est le niveau d'expérience le plus représenté ({df[exp_col].value_counts().max()} candidats)
                * {len(df[df['Score'] > 75])} candidats ont un score de correspondance supérieur à 75%
                """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab3:
                st.markdown("<div class='sub-header'>👤 Profils détaillés des candidats</div>", unsafe_allow_html=True)
                
                # Sélection du candidat
                candidate_idx = st.selectbox(
                    "Sélectionnez un candidat",
                    options=range(len(df)),
                    format_func=lambda i: f"{df.iloc[i]['Prénom' if lang == 'French' else 'First Name']} {df.iloc[i]['Nom' if lang == 'French' else 'Last Name']} - {df.iloc[i]['Nom du fichier']}"
                )
                
                # Affichage du profil détaillé
                candidate = df.iloc[candidate_idx]
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader(f"{candidate['Prénom' if lang == 'French' else 'First Name']} {candidate['Nom' if lang == 'French' else 'Last Name']}")
                    st.caption(f"📄 {candidate['Nom du fichier']}")
                    
                    # Informations de contact
                    st.markdown("#### 📞 Contact")
                    st.markdown(f"**Email:** {candidate['Email']}")
                    st.markdown(f"**Téléphone:** {candidate['Téléphone' if lang == 'French' else 'Phone']}")
                    
                    # Profil
                    st.markdown("#### 👤 Profil")
                    st.markdown(candidate['Profil' if lang == 'French' else 'Profile'])
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    # Score et niveau d'expérience
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### 📊 Évaluation")
                    score_color = "green" if candidate['Score'] >= 75 else "orange" if candidate['Score'] >= 50 else "red"
                    st.markdown(f"**Score de correspondance:** <span style='color:{score_color};font-weight:bold;'>{candidate['Score']}%</span>", unsafe_allow_html=True)
                    st.markdown(f"**Niveau d'expérience:** {candidate['Niveau expérience' if lang == 'French' else 'Experience Level']}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Compétences
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### 🛠️ Compétences")
                
                # Affichage des compétences avec mise en évidence des mots-clés
                skills_list = candidate['Compétences' if lang == 'French' else 'Skills']
                if job_keywords_list and skills_list:
                    for skill in skills_list:
                        if any(kw.lower() in skill.lower() for kw in job_keywords_list):
                            st.markdown(f"- <span style='background-color:#E3F2FD;padding:2px 5px;border-radius:3px;'>{skill}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"- {skill}")
                else:
                    for skill in skills_list:
                        st.markdown(f"- {skill}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Formation et certifications
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### 🎓 Formation")
                    education_list = candidate['Formations' if lang == 'French' else 'Education']
                    if education_list:
                        for edu in education_list:
                            st.markdown(f"- {edu}")
                    else:
                        st.markdown("*Aucune formation spécifiée*")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("#### 📜 Certifications")
                    cert_list = candidate['Certifications']
                    if cert_list:
                        for cert in cert_list:
                            st.markdown(f"- {cert}")
                    else:
                        st.markdown("*Aucune certification spécifiée*")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Langues
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### 🌍 Langues")
                lang_list = candidate['Langues' if lang == 'French' else 'Languages']
                if lang_list:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    for i, lang_item in enumerate(lang_list):
                        with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                            st.markdown(f"- {lang_item}")
                else:
                    st.markdown("*Aucune langue spécifiée*")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Recommandation RH
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### 💼 Recommandation RH")
                
                # Calcul d'une recommandation basée sur le score et l'expérience
                if candidate['Score'] >= 80:
                    recommendation = "⭐⭐⭐ **Excellent candidat** - À contacter en priorité"
                elif candidate['Score'] >= 65:
                    recommendation = "⭐⭐ **Bon candidat** - À considérer sérieusement"
                elif candidate['Score'] >= 50:
                    recommendation = "⭐ **Candidat potentiel** - À considérer si les meilleurs profils ne sont pas disponibles"
                else:
                    recommendation = "❌ **Profil non adapté** - Ne correspond pas aux critères du poste"
                
                st.markdown(recommendation)
                
                # Actions RH
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("📞 Planifier un entretien", key=f"interview_{candidate_idx}")
                with col2:
                    st.button("✉️ Contacter par email", key=f"email_{candidate_idx}")
                with col3:
                    st.button("❌ Rejeter la candidature", key=f"reject_{candidate_idx}")
                
                st.markdown("</div>", unsafe_allow_html=True)

def customize_interface():
    """Applique des personnalisations supplémentaires à l'interface"""
    # Ajout d'un logo ou d'une image d'en-tête (si nécessaire)
    # Pourrait être complété par une image de l'entreprise ou un logo RH
    pass

if __name__ == '__main__':
    main()