from pathlib import Path

import numpy as np
import pandas as pd
import json
import joblib
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION DEEP LEARNING — Les modèles non encore importés restent désactivés.
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepLab | Projet Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

NAMES = [
    "Avion", "Automobile", "Oiseau", "Chat", "Cerf",
    "Chien", "Grenouille", "Cheval", "Bateau", "Camion",
]

PAGES = {
    "Accueil": {"icon": "⌂", "tag": "Vue d'ensemble"},
    "TP1 — Bank Data": {"icon": "◈", "tag": "Données tabulaires"},
    "TP2 — Student Performance": {"icon": "◇", "tag": "Prédiction académique"},
    "TP3 — CIFAR-10": {"icon": "▧", "tag": "Vision par ordinateur"},
    "TP5 — Tuberculose": {"icon": "✳", "tag": "Imagerie médicale"},
    "Détection de la pneumonie": {"icon": "✚", "tag": "Imagerie médicale"},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#172b3d; --muted:#52677a; --line:#dfe9ec; --accent:#087d7b; }
html,body,[class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {
  background:radial-gradient(ellipse 65% 35% at 90% -5%,rgba(104,213,196,.15),transparent 72%),
             linear-gradient(150deg,#fafdfd 0%,#f3f9f8 54%,#eef5f7 100%);
  color:var(--ink);
}
[data-testid="stHeader"] {background:rgba(250,253,253,.90);}
[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid #e1e9ed;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:#52677a;}
.block-container {max-width:1220px;padding-top:2.5rem;padding-bottom:4rem;}
h1,h2,h3,h4 {font-family:'Manrope',sans-serif!important;letter-spacing:-.045em;color:#173046!important;}
h1 {font-size:clamp(2rem,3.8vw,3.1rem)!important;font-weight:800!important;}
h2 {font-weight:800!important;}
p,label {color:#425a6c;}
[data-testid="stSidebar"] .stRadio label {padding:7px 0;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {gap:7px;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {border-radius:9px;padding:9px 11px;}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {background:#eaf7f3;}
[data-testid="stFileUploader"] {background:#ffffff;border:1px solid var(--line);border-radius:15px;padding:8px;}
[data-testid="stMetric"] {background:#ffffff;border:1px solid var(--line);border-radius:15px;padding:17px;box-shadow:0 6px 18px rgba(22,67,83,.035);}
.stButton>button[kind="primary"] {background:#078782!important;color:#ffffff!important;font-weight:800!important;border:none!important;border-radius:10px!important;padding:.65rem 1.35rem!important;}
.stButton>button[kind="primary"]:hover {background:#056a68!important;}
.stButton>button:not([kind="primary"]) {border-radius:10px!important;}
hr {border-color:var(--line)!important;}
.eyebrow {font-size:.74rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#087d7b;margin-bottom:10px;}
.hero {padding:30px 32px 27px;background:linear-gradient(116deg,#ffffff 0%,#f0fbf8 67%,#dff4ef 100%);border:1px solid #d6ebe6;border-radius:22px;margin-bottom:26px;position:relative;overflow:hidden;box-shadow:0 15px 40px rgba(28,87,93,.06);}
.hero:after {content:'✦';position:absolute;font-size:180px;right:25px;top:-45px;color:rgba(5,138,127,.065);pointer-events:none;}
.hero h1 {margin:0 0 8px 0;font-size:2.65rem!important;line-height:1.15;color:#123349!important;}
.hero p {margin:0;color:#506a78;max-width:730px;font-size:1rem;line-height:1.65;}
.hero .note {margin-top:18px;font-size:.79rem;color:#087d7b;font-weight:700;letter-spacing:.06em;}
.panel {background:#ffffff;border:1px solid #e0e9ed;border-radius:17px;padding:23px 23px 20px;margin-bottom:14px;min-height:165px;box-shadow:0 8px 25px rgba(26,80,88,.04);}
.panel .icon {font-size:1.65rem;display:block;margin-bottom:15px;color:#098a82;}
.panel h3 {margin:0 0 9px;color:#173349;font-size:1.1rem!important;letter-spacing:-.025em!important;}
.panel p {margin:0 0 15px;font-size:.88rem;line-height:1.55;color:#526b7b;}
.tag {display:inline-flex;border:1px solid #b8e7d7;background:#e4f7ee;color:#086e5e;font-size:.72rem;padding:6px 10px;border-radius:100px;font-weight:800;letter-spacing:.025em;}
.tag.pending {border-color:#f3d9a9;background:#fff7e8;color:#94620e;}
.section-label {font-size:.85rem;font-weight:800;color:#294d60;margin:16px 0 15px;letter-spacing:.04em;text-transform:uppercase;}
.note-panel {border-radius:13px;background:#eaf7f5;border:1px solid #d6ebe5;border-left:3px solid #0a988e;padding:16px 19px;color:#315a62;line-height:1.65;margin:8px 0 20px;}
.small {color:#647b8b;font-size:.82rem;}
.side-brand {font-size:1.5rem;font-weight:800;color:#143c49;letter-spacing:-.05em;margin:8px 0 3px;}
.side-sub {font-size:.77rem;color:#6b8190;margin-bottom:22px;}
.footer {font-size:.8rem;color:#647b8a;padding:10px 0 5px;}
@media(max-width:720px){.hero{padding:24px 20px}.hero h1{font-size:2rem!important}.block-container{padding-top:1.5rem}}
</style>
""", unsafe_allow_html=True)


def hero(kicker: str, heading: str, description: str, foot: str = "") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div>'
        f'<h1>{heading}</h1><p>{description}</p>'
        f'{f"<div class=\"note\">{foot}</div>" if foot else ""}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, description: str, icon: str, ready: bool = False) -> None:
    state = '<span class="tag">PRÉDICTION ACTIVE</span>' if ready else '<span class="tag pending">INTÉGRATION À VENIR</span>'
    st.markdown(
        f'<div class="panel"><span class="icon">{icon}</span>'
        f'<h3>{title}</h3><p>{description}</p>{state}</div>',
        unsafe_allow_html=True,
    )



@st.cache_resource(show_spinner="Chargement du CNN Tuberculose…")
def load_tuberculosis():
    from tensorflow import keras

    path = MODELS / "meilleur_cnn_tuberculosis.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier absent : {path}")
    model = keras.models.load_model(path, compile=False)
    if model.input_shape[-3:] != (224, 224, 3) or model.output_shape[-1] != 1:
        raise ValueError("Dimensions incompatibles avec le TP Tuberculose.")
    return model


@st.cache_resource(show_spinner="Chargement du CNN Pneumonia optimisé…")
def load_pneumonia():
    from tensorflow import keras

    path = MODELS / "meilleur_cnn_pneumonia_optimise.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier absent : {path}")
    model = keras.models.load_model(path, compile=False)
    if model.input_shape[-3:] != (224, 224, 3) or model.output_shape[-1] != 1:
        raise ValueError("Dimensions incompatibles avec le modèle Pneumonia : attendu 224 × 224 × 3 → 1.")
    return model


@st.cache_resource(show_spinner="Chargement du CNN CIFAR-10…")
def load_cifar():
    from tensorflow import keras
    path = MODELS / "meilleur_cnn_ameliore.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Fichier absent : {path}")
    return keras.models.load_model(path, compile=False)


@st.cache_resource(show_spinner="Chargement des artefacts bancaires…")
def load_bank():
    """Charger uniquement des artefacts créés par notre notebook de confiance."""
    from tensorflow import keras

    path_model = MODELS / "modele_banque.keras"
    path_preproc = MODELS / "preprocesseur_banque.joblib"
    path_columns = MODELS / "colonnes_banque.json"
    for path in (path_model, path_preproc, path_columns):
        if not path.is_file():
            raise FileNotFoundError(f"Fichier manquant : {path.name}")

    model = keras.models.load_model(path_model, compile=False)
    preproc = joblib.load(path_preproc)
    with path_columns.open(encoding="utf-8") as f:
        columns = json.load(f)
    expected = [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign",
        "pdays", "previous", "poutcome",
    ]
    if columns != expected:
        raise ValueError("L'ordre ou les noms des 15 colonnes ne correspondent pas au TP1.")
    if model.input_shape[-1] != 58 or model.output_shape[-1] != 1:
        raise ValueError("Dimensions incompatibles : le modèle attendu est 58 → 1.")
    return model, preproc, columns


def get_bank_categories(preproc):
    """Lire les catégories réellement apprises, sans les inventer."""
    try:
        categorical_pipeline = preproc.named_transformers_["categoriel"]
        encoder = categorical_pipeline.named_steps["encodage"]
        cat_cols = list(preproc.transformers_[1][2])
        return {column: [str(v) for v in values]
                for column, values in zip(cat_cols, encoder.categories_)}
    except (KeyError, AttributeError, IndexError, TypeError) as exc:
        raise ValueError("Catégories du préprocesseur introuvables.") from exc


BANK_LABELS = {
    "job": "Profession", "marital": "Situation matrimoniale",
    "education": "Niveau d'éducation", "default": "Défaut de crédit",
    "housing": "Prêt immobilier", "loan": "Prêt personnel",
    "contact": "Canal de contact", "month": "Mois du contact",
    "day_of_week": "Jour de la semaine", "poutcome": "Résultat de la campagne précédente",
}

BANK_TRANSLATIONS = {
    "yes": "Oui", "no": "Non", "unknown": "Inconnu",
    "single": "Célibataire", "married": "Marié(e)", "divorced": "Divorcé(e)",
    "admin.": "Administration", "blue-collar": "Ouvrier(ère)",
    "entrepreneur": "Entrepreneur(e)", "housemaid": "Employé(e) de maison",
    "management": "Cadre / Management", "retired": "Retraité(e)",
    "self-employed": "Indépendant(e)", "services": "Services",
    "student": "Étudiant(e)", "technician": "Technicien(ne)",
    "unemployed": "Sans emploi", "basic.4y": "Cycle de base (4 ans)",
    "basic.6y": "Cycle de base (6 ans)", "basic.9y": "Cycle de base (9 ans)",
    "high.school": "Lycée", "illiterate": "Sans scolarisation",
    "professional.course": "Formation professionnelle", "university.degree": "Université",
    "cellular": "Téléphone mobile", "telephone": "Téléphone fixe",
    "nonexistent": "Aucune campagne précédente", "failure": "Échec", "success": "Succès",
    "mon": "Lundi", "tue": "Mardi", "wed": "Mercredi", "thu": "Jeudi", "fri": "Vendredi",
    "jan": "Janvier", "feb": "Février", "mar": "Mars", "apr": "Avril",
    "may": "Mai", "jun": "Juin", "jul": "Juillet", "aug": "Août",
    "sep": "Septembre", "oct": "Octobre", "nov": "Novembre", "dec": "Décembre",
}


def category_input(field, categories, help_text=None):
    choices = categories.get(field)
    if not choices:
        raise ValueError(f"Aucune modalité connue pour {field}.")
    return st.selectbox(
        BANK_LABELS[field], choices,
        format_func=lambda raw: f"{BANK_TRANSLATIONS.get(raw, raw)} · {raw}",
        help=help_text,
    )


with st.sidebar:
    st.markdown('<div class="side-brand">✳ DeepLab</div><div class="side-sub">Projet académique · Deep Learning</div>', unsafe_allow_html=True)
    selected = st.radio(
        "Navigation",
        list(PAGES),
        format_func=lambda page: f'{PAGES[page]["icon"]}   {page}',
    )
    st.divider()
    st.caption("5 projets · TensorFlow / Keras")
    st.caption("Un seul espace pour explorer, tester et comparer.")

if selected == "Accueil":
    hero("Votre espace de démonstration", "L'intelligence artificielle, en pratique.",
         "Cinq expériences de Deep Learning regroupées dans une application interactive. "
         "Explorez les cinq modèles entraînés et réalisez des simulations interactives.",
         "05 / 05 MODULES DE PRÉDICTION ACTIFS")
    st.markdown('<div class="section-label">Explorer les projets</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        card("Bank Data", "Probabilité de souscription d’un dépôt bancaire avec les variables du TP1.", "◈", ready=True)
        card("Tuberculose", "Analyse pédagogique de radiographies pulmonaires.", "✳", ready=True)
    with b:
        card("Student Performance", "Modèle prédictif pour les performances étudiantes.", "◇", ready=True)
        card("Pneumonie", "Classification d'images pulmonaires avec un CNN.", "✚", ready=True)
    with c:
        card("CIFAR-10", "Reconnaissance d'images parmi dix catégories.", "▧", ready=True)
    st.markdown('<div class="note-panel">Sélectionnez un projet dans le menu latéral. '
                'Les cinq modules sont intégrés. Les analyses de radiographies ont une finalité exclusivement pédagogique.</div>', unsafe_allow_html=True)

elif selected == "TP3 — CIFAR-10":
    hero("Projet 03 · Vision par ordinateur", "Classification CIFAR-10",
         "Importez une image et laissez le réseau convolutif l'associer à l'une des dix catégories du jeu CIFAR-10.",
         "MODÈLE CNN · 32 × 32 PIXELS · 10 CLASSES")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("01  ·  Importer une image")
        upload = st.file_uploader("JPG ou PNG", type=["jpg", "jpeg", "png"], key="cifar_upload")
        if upload is not None:
            try:
                photo = Image.open(upload).convert("RGB")
            except Exception:
                st.error("Le fichier ne peut pas être lu comme une image.")
                st.stop()
            st.image(photo, caption=f"Image importée · {photo.width} × {photo.height} px", use_container_width=True)
        else:
            st.markdown('<div class="note-panel">Ajoutez une image pour activer la prédiction. '
                        'Les images proches du jeu CIFAR-10 donnent généralement des résultats plus pertinents.</div>',
                        unsafe_allow_html=True)
    with right:
        st.subheader("02  ·  Analyser")
        st.markdown("**Votre modèle**")
        st.write("CNN amélioré · TensorFlow / Keras")
        st.write("**Entrée :** 32 × 32 × 3 · **Sortie :** 10 scores")
        st.markdown('<div class="small">Prétraitement supposé : pixels RVB / 255. '
                    'À confirmer avec le notebook d’entraînement.</div>', unsafe_allow_html=True)
        st.write("")
        predict = st.button("✦  Lancer la prédiction", type="primary", disabled=upload is None, use_container_width=True)
    if upload is not None and predict:
        with st.spinner("Inférence en cours…"):
            resized = photo.resize((32, 32), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            arr = np.expand_dims(arr, axis=0)
            try:
                model = load_cifar()
                scores = np.asarray(model.predict(arr, verbose=0))[0]
                if len(scores) != 10:
                    raise ValueError(f"Le modèle a produit {len(scores)} sorties au lieu de 10.")
            except Exception as exc:
                st.error(f"Impossible de prédire : {exc}")
                st.stop()
        best = int(np.argmax(scores))
        st.divider()
        st.subheader("03  ·  Résultats")
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Classe prédite", NAMES[best])
        with k2:
            st.metric("Score du modèle", f"{float(scores[best]) * 100:.2f} %")
        st.markdown("#### Répartition des scores")
        st.bar_chart({name: float(score) for name, score in zip(NAMES, scores)}, horizontal=True)
        st.caption("Ces scores ne sont pas des garanties de justesse. La performance mesurée "
                   "sur CIFAR-10 ne garantit pas la fiabilité sur des photos quelconques.")

elif selected == "TP1 — Bank Data":
    hero(
        "Projet 01 · Prédiction bancaire", "Bank Data · Souscription",
        "Découvrez la probabilité estimée de souscription à un dépôt bancaire. "
        "Renseignez les caractéristiques client et les informations de campagne : "
        "notre préprocesseur les convertit en 58 variables pour le réseau de neurones.",
        "MODÈLE DNN OPTIMISÉ · 15 CHAMPS · 58 VARIABLES EN ENTRÉE",
    )
    try:
        bank_model, bank_preproc, bank_columns = load_bank()
        bank_categories = get_bank_categories(bank_preproc)
    except Exception as exc:
        st.error(f"Connexion du TP1 impossible : {exc}")
        st.info("Vérifiez les trois fichiers dans le dossier models : "
                "modele_banque.keras, preprocesseur_banque.joblib et colonnes_banque.json.")
        st.stop()

    st.markdown('<div class="section-label">Fiche de prédiction · Campagne bancaire</div>',
                unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.metric("Champs d'origine", "15")
    with b:
        st.metric("Variables après préparation", "58")
    with c:
        st.metric("Statut du modèle", "Opérationnel")

    with st.form("form_banque", clear_on_submit=False):
        st.markdown("### 01 · Profil du client")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            age = st.number_input("Âge", min_value=18, max_value=110, value=38, step=1)
            job = category_input("job", bank_categories)
        with c2:
            marital = category_input("marital", bank_categories)
            education = category_input("education", bank_categories)
        with c3:
            default = category_input("default", bank_categories)
            housing = category_input("housing", bank_categories)
            loan = category_input("loan", bank_categories)

        st.divider()
        st.markdown("### 02 · Informations sur le contact")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            contact = category_input("contact", bank_categories)
            month = category_input("month", bank_categories)
        with c2:
            day = category_input("day_of_week", bank_categories)
        with c3:
            duration = st.number_input(
                "Durée du dernier contact (secondes)",
                min_value=0, max_value=20000, value=180, step=10,
                help="Variable disponible après l'appel. Une prévision avant l'appel ne peut pas utiliser sa durée réelle.",
            )

        st.divider()
        st.markdown("### 03 · Historique des campagnes")
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            campaign = st.number_input("Nombre de contacts pendant cette campagne", 1, 200, 1)
            previous = st.number_input("Contacts lors des campagnes précédentes", 0, 200, 0)
        with c2:
            pdays = st.number_input(
                "Jours depuis le précédent contact (pdays)", 0, 999, 999,
                help="999 est le code du jeu de données pour « aucun contact précédent ».",
            )
        with c3:
            poutcome = category_input("poutcome", bank_categories)

        st.write("")
        submitted = st.form_submit_button(
            "✦  Analyser la souscription", type="primary", use_container_width=True
        )

    if submitted:
        values = {
            "age": age, "job": job, "marital": marital, "education": education,
            "default": default, "housing": housing, "loan": loan,
            "contact": contact, "month": month, "day_of_week": day,
            "duration": duration, "campaign": campaign, "pdays": pdays,
            "previous": previous, "poutcome": poutcome,
        }
        with st.spinner("Préparation des données et prédiction en cours…"):
            try:
                observation = pd.DataFrame([values], columns=bank_columns)
                transformed = bank_preproc.transform(observation)
                transformed = np.asarray(transformed, dtype=np.float32)
                if transformed.shape != (1, 58):
                    raise ValueError(f"Dimensions obtenues : {transformed.shape}, attendues : (1, 58)")
                probability = float(bank_model.predict(transformed, verbose=0)[0][0])
                if not np.isfinite(probability):
                    raise ValueError("La prédiction contient une valeur non définie.")
                probability = float(np.clip(probability, 0.0, 1.0))
            except Exception as exc:
                st.error(f"Prédiction impossible : {exc}")
                st.stop()

        st.divider()
        st.markdown("### 04 · Résultat de votre simulation")
        result = "Souscription probable" if probability >= 0.5 else "Souscription peu probable"
        v1, v2 = st.columns([1, 1.3], gap="large")
        with v1:
            st.metric("Résultat du modèle", result)
            st.metric("Score de souscription", f"{probability * 100:.1f} %")
        with v2:
            st.write("**Répartition des scores**")
            st.bar_chart(pd.DataFrame({
                "Résultat": ["Souscription", "Non-souscription"],
                "Score (%)": [100 * probability, 100 * (1 - probability)],
            }).set_index("Résultat"))
        st.markdown(
            '<div class="note-panel">Le seuil de classification est fixé à 50 %. '
            'Ce résultat est une estimation statistique à visée pédagogique, et non une garantie de décision réelle.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="note-panel">Complétez les champs, puis cliquez sur '
                    '<strong>Analyser la souscription</strong>. Le formulaire conserve les valeurs '
                    'du jeu de données et le prétraitement appris pendant le TP1.</div>',
                    unsafe_allow_html=True)
    st.caption("Important : la durée du contact est connue seulement après l’appel. "
               "Cette démonstration reproduit le TP1, elle n’est pas un outil de ciblage avant appel.")

elif selected == "TP2 — Student Performance":
    hero(
        "Projet 02 · Analyse pédagogique",
        "Student Performance · Estimation des résultats",
        "Estimez l'indice de performance académique d'un étudiant à partir "
        "de ses habitudes d'étude et de ses résultats précédents. "
        "Les cinq informations sont transformées automatiquement en six valeurs pour notre DNN.",
        "RÉGRESSION · 5 INFORMATIONS · 6 VARIABLES PRÉPARÉES",
    )

    @st.cache_resource(show_spinner="Chargement du modèle Student Performance…")
    def load_student():
        from tensorflow import keras
        model_path = MODELS / "modele_etudiant.keras"
        preproc_path = MODELS / "preprocesseur_etudiant.joblib"
        columns_path = MODELS / "colonnes_etudiant.json"
        for path in (model_path, preproc_path, columns_path):
            if not path.is_file():
                raise FileNotFoundError(f"Fichier manquant : {path.name}")
        model = keras.models.load_model(model_path, compile=False)
        preproc = joblib.load(preproc_path)
        with columns_path.open(encoding="utf-8") as f:
            columns = json.load(f)
        expected = [
            "Hours Studied", "Previous Scores", "Extracurricular Activities",
            "Sleep Hours", "Sample Question Papers Practiced",
        ]
        if columns != expected:
            raise ValueError("Les colonnes ne correspondent pas aux cinq champs du TP2.")
        if model.input_shape[-1] != 6 or model.output_shape[-1] != 1:
            raise ValueError("Dimensions du modèle incompatibles : attendu 6 → 1.")
        return model, preproc, columns

    try:
        student_model, student_preproc, student_columns = load_student()
        # Les catégories proviennent du OneHotEncoder réellement ajusté.
        student_options = None
        for name, transformer, cols in student_preproc.transformers_:
            if "Extracurricular Activities" in list(cols) and hasattr(transformer, "named_steps"):
                enc = transformer.named_steps.get("encodage")
                if enc is not None:
                    student_options = [str(v) for v in enc.categories_[0]]
        if not student_options:
            raise ValueError("Modalités des activités extrascolaires introuvables dans le préprocesseur.")
    except Exception as exc:
        st.error(f"Impossible de charger le TP2 : {exc}")
        st.info("Vérifiez modele_etudiant.keras, preprocesseur_etudiant.joblib "
                "et colonnes_etudiant.json dans le dossier models.")
        st.stop()

    st.markdown('<div class="section-label">Simulateur · Profil académique</div>',
                unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Informations saisies", "5")
    with m2:
        st.metric("Variables préparées", "6")
    with m3:
        st.metric("Type de modèle", "Régression")

    with st.form("form_etudiant"):
        st.markdown("### 01 · Parcours scolaire")
        a, b = st.columns(2, gap="large")
        with a:
            hours = st.number_input("Heures d'étude par jour", min_value=1, max_value=9,
                                    value=5, step=1)
            prev = st.number_input("Notes précédentes (sur 100)", min_value=40,
                                   max_value=99, value=75, step=1)
        with b:
            activities = st.selectbox(
                "Activités extrascolaires",
                student_options,
                format_func=lambda x: "Oui" if x.lower() == "yes" else
                                      "Non" if x.lower() == "no" else x,
            )
            sleep = st.number_input("Heures de sommeil par nuit", min_value=4,
                                    max_value=9, value=7, step=1)

        st.divider()
        st.markdown("### 02 · Préparation aux examens")
        papers = st.number_input("Sujets d'examen pratiqués", min_value=0,
                                 max_value=9, value=4, step=1)
        st.caption("Les valeurs proposées suivent les plages du jeu de données du TP2.")
        submitted_student = st.form_submit_button(
            "✦  Estimer la performance", type="primary", use_container_width=True
        )

    if submitted_student:
        values = {
            "Hours Studied": hours,
            "Previous Scores": prev,
            "Extracurricular Activities": activities,
            "Sleep Hours": sleep,
            "Sample Question Papers Practiced": papers,
        }
        with st.spinner("Prétraitement des données et calcul de l'estimation…"):
            try:
                observation = pd.DataFrame([values], columns=student_columns)
                prepared = student_preproc.transform(observation)
                prepared = np.asarray(prepared, dtype=np.float32)
                if prepared.shape != (1, 6):
                    raise ValueError(f"Forme obtenue {prepared.shape} au lieu de (1, 6).")
                prediction = float(student_model.predict(prepared, verbose=0)[0][0])
                if not np.isfinite(prediction):
                    raise ValueError("Le modèle a retourné une valeur non définie.")
            except Exception as exc:
                st.error(f"Erreur de prédiction : {exc}")
                st.stop()

        st.divider()
        st.markdown("### 03 · Résultat de la simulation")
        c1, c2 = st.columns([1, 1.4], gap="large")
        with c1:
            st.metric("Indice de performance estimé", f"{prediction:.1f}")
            st.caption("Valeur estimée par un modèle de régression, pas une probabilité.")
        with c2:
            st.markdown('<div class="note-panel"><strong>Comment lire ce résultat ?</strong><br>'
                        "Le réseau prédit l'indice « Performance Index » du jeu de données. "
                        "Il ne s'agit pas d'une note officielle ni d'une garantie de réussite."
                        '</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="note-panel">Renseignez les cinq champs et cliquez sur '
                    '<strong>Estimer la performance</strong> pour calculer le résultat '
                    'de votre simulation.</div>', unsafe_allow_html=True)

elif selected == "TP5 — Tuberculose":
    hero(
        "Projet 05 · Imagerie médicale", "Analyse de radiographie · Tuberculose",
        "Explorez la classification expérimentale d'une radiographie pulmonaire "
        "avec notre réseau de neurones convolutif.",
        "CNN · 224 × 224 PIXELS · NORMAL / TUBERCULOSIS",
    )
    st.markdown(
        '<div class="note-panel"><strong>Utilisation pédagogique uniquement.</strong> '
        'Cette prédiction automatisée ne permet ni de diagnostiquer ni d’exclure la tuberculose. '
        'En cas de doute médical, consultez un professionnel de santé.</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Classes du modèle", "2")
    with k2:
        st.metric("Résolution d'entrée", "224 × 224")
    with k3:
        st.metric("Seuil de décision", "50 %")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("01 · Importer une radiographie")
        tb_upload = st.file_uploader(
            "Choisissez une image JPG ou PNG", type=["jpg", "jpeg", "png"],
            key="tuberculosis_upload",
        )
        image_tb = None
        if tb_upload is not None:
            try:
                image_tb = Image.open(tb_upload).convert("RGB")
            except Exception:
                st.error("Ce fichier ne peut pas être ouvert comme image.")
                st.stop()
            st.image(
                image_tb,
                caption=f"Radiographie importée · {image_tb.width} × {image_tb.height} px",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="note-panel">Importez une radiographie pulmonaire pour '
                'activer la simulation. Utilisez de préférence une image issue du jeu '
                'de données du TP.</div>', unsafe_allow_html=True,
            )

    with right:
        st.subheader("02 · Lancer l'analyse")
        st.markdown("**Modèle :** CNN Tuberculosis")
        st.write("**Préparation :** redimensionnement 224 × 224, conversion RVB, pixels / 255.")
        st.write("**Étiquettes :** 0 = Normal · 1 = Tuberculosis")
        st.write("")
        tb_predict = st.button(
            "✦  Analyser la radiographie", type="primary",
            disabled=image_tb is None, use_container_width=True,
            key="tb_predict_button",
        )

    if tb_predict and image_tb is not None:
        with st.spinner("Analyse du CNN en cours…"):
            try:
                prepared_image = image_tb.resize((224, 224), Image.Resampling.BILINEAR)
                pixels = np.asarray(prepared_image, dtype=np.float32) / 255.0
                image_batch = np.expand_dims(pixels, axis=0)
                tb_model = load_tuberculosis()
                score_tb = float(np.asarray(tb_model.predict(image_batch, verbose=0)).reshape(-1)[0])
                if not np.isfinite(score_tb) or not 0 <= score_tb <= 1:
                    raise ValueError("Score non valide retourné par le modèle.")
            except Exception as exc:
                st.error(f"Analyse impossible : {exc}")
                st.stop()

        predicted_tb = "Tuberculosis" if score_tb >= 0.5 else "Normal"
        predicted_score = score_tb if score_tb >= 0.5 else 1.0 - score_tb
        st.divider()
        st.subheader("03 · Résultat de la simulation")
        res_left, res_right = st.columns([1, 1.2], gap="large")
        with res_left:
            st.metric("Classe prédite", predicted_tb)
            st.metric("Score attribué à cette classe", f"{predicted_score * 100:.2f} %")
        with res_right:
            st.markdown("**Scores des deux catégories**")
            st.bar_chart(
                pd.DataFrame({
                    "Classe": ["Normal", "Tuberculosis"],
                    "Score du modèle (%)": [100 * (1 - score_tb), 100 * score_tb],
                }).set_index("Classe"),
                horizontal=True,
            )
        st.caption(
            "Le résultat correspond au score d'un modèle expérimental entraîné "
            "sur un jeu de données déterminé. Il ne s'agit pas d'une probabilité "
            "clinique validée ni d'une décision médicale."
        )

elif selected == "Détection de la pneumonie":
    hero(
        "Projet complémentaire · Imagerie médicale", "Analyse de radiographie · Pneumonie",
        "Explorez la classification expérimentale d'une radiographie pulmonaire "
        "avec notre CNN optimisé à partir de la recherche Hyperband.",
        "CNN OPTIMISÉ · 224 × 224 PIXELS · NORMAL / PNEUMONIA",
    )
    st.markdown(
        '<div class="note-panel"><strong>Usage pédagogique uniquement.</strong> '
        'Cette simulation ne peut ni diagnostiquer ni exclure une pneumonie. '
        'En cas de symptômes ou de doute médical, consultez un professionnel de santé.</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Images de test", "624")
    with m2:
        st.metric("Accuracy sur le test", "83,33 %")
    with m3:
        st.metric("AUC sur le test", "0,9051")

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("01 · Importer une radiographie")
        pneumonia_upload = st.file_uploader(
            "Choisissez une image JPG ou PNG",
            type=["jpg", "jpeg", "png"],
            key="pneumonia_upload",
        )
        pneumonia_image = None
        if pneumonia_upload is not None:
            try:
                pneumonia_image = Image.open(pneumonia_upload).convert("RGB")
            except Exception:
                st.error("Ce fichier ne peut pas être ouvert comme image.")
                st.stop()
            st.image(
                pneumonia_image,
                caption=f"Radiographie importée · {pneumonia_image.width} × {pneumonia_image.height} px",
                use_container_width=True,
            )
        else:
            st.markdown(
                '<div class="note-panel">Importez une radiographie du jeu de test Pneumonia '
                'pour réaliser une première simulation.</div>',
                unsafe_allow_html=True,
            )

    with right:
        st.subheader("02 · Lancer l'analyse")
        st.markdown("**Modèle :** CNN Pneumonia optimisé")
        st.write("**Préparation :** redimensionnement 224 × 224, conversion RVB, pixels / 255.")
        st.write("**Classes :** 0 = NORMAL · 1 = PNEUMONIA")
        st.write("**Seuil de classification :** 50 %")
        st.write("")
        pneumonia_predict = st.button(
            "✦  Analyser la radiographie",
            type="primary",
            disabled=pneumonia_image is None,
            use_container_width=True,
            key="pneumonia_predict_button",
        )

    if pneumonia_predict and pneumonia_image is not None:
        with st.spinner("Analyse du CNN en cours…"):
            try:
                resized = pneumonia_image.resize((224, 224), Image.Resampling.BILINEAR)
                pixels = np.asarray(resized, dtype=np.float32) / 255.0
                batch = np.expand_dims(pixels, axis=0)
                pneumonia_model = load_pneumonia()
                score_positive = float(
                    np.asarray(pneumonia_model.predict(batch, verbose=0)).reshape(-1)[0]
                )
                if not np.isfinite(score_positive) or not (0 <= score_positive <= 1):
                    raise ValueError("Score non valide retourné par le modèle.")
            except Exception as exc:
                st.error(f"Analyse impossible : {exc}")
                st.stop()

        predicted_class = "PNEUMONIA" if score_positive >= 0.5 else "NORMAL"
        class_score = score_positive if score_positive >= 0.5 else 1.0 - score_positive
        st.divider()
        st.subheader("03 · Résultat de la simulation")
        res_left, res_right = st.columns([1, 1.2], gap="large")
        with res_left:
            st.metric("Classe prédite", predicted_class)
            st.metric("Score attribué à cette classe", f"{class_score * 100:.2f} %")
        with res_right:
            st.markdown("**Scores des deux catégories**")
            st.bar_chart(
                pd.DataFrame({
                    "Classe": ["NORMAL", "PNEUMONIA"],
                    "Score du modèle (%)": [100 * (1 - score_positive), 100 * score_positive],
                }).set_index("Classe"),
                horizontal=True,
            )
        st.caption(
            "Ce score issu d'un modèle expérimental n'est pas une probabilité clinique "
            "validée et ne doit pas servir à prendre une décision médicale. "
            "Sur le jeu de test, le rappel de la classe PNEUMONIA était de 83,33 % "
            "au seuil de 0,5, avec 65 faux négatifs sur 390 cas positifs."
        )

st.divider()
st.markdown('<div class="footer">DEEP LAB  ·  PROJET DE DEEP LEARNING &nbsp;&nbsp; | &nbsp;&nbsp; TensorFlow · Keras · Streamlit</div>', unsafe_allow_html=True)
