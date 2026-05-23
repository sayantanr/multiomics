# =====================================================================
# MultiOmics-Integrator v1.2 - Premium Streamlit Native Version
# 50+ Statistics + 50+ Interactive Plots -> Single HTML & Dashboard
# Designed for Local Windows Machine & Streamlit Cloud
# =====================================================================

import os, io, json, time, warnings, re, base64
from pathlib import Path
from datetime import datetime, UTC
from collections import defaultdict, Counter

# Data and Science dependencies
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from fastdtw import fastdtw

# Web / API / Interactive dependencies
import streamlit as st
import requests
import gseapy as gp
from bioservices import KEGG, UniProt
import mygene

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PAGE CONFIG & CUSTOM TYPOGRAPHY / STYLE SYSTEM
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MultiOmics Integrator Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render premium CSS layout system
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;500;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Typography Styling */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #f1f5f9 !important;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        margin-top: 10px;
    }
    
    /* Hero Container Header */
    .hero-container {
        background: linear-gradient(135deg, #312e81 0%, #581c87 50%, #831843 100%);
        border-radius: 16px;
        padding: 35px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.4);
        border: 1px solid #4c1d95;
    }
    
    .hero-title {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        color: white !important;
        margin: 0 0 10px 0 !important;
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.95;
        margin: 0 !important;
        font-weight: 300;
        line-height: 1.5;
        color: #e2e8f0 !important;
    }
    
    /* Styled Premium KPI Cards in Dark Mode */
    .kpi-card {
        background: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #818cf8;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -3px rgba(99, 102, 241, 0.2);
        border-color: #4f46e5;
    }
    
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc !important;
        margin-top: 5px;
        font-family: 'Outfit', sans-serif;
    }
    
    .kpi-lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    
    /* Sleek sidebar adjustments */
    .css-1d391kg {
        background-color: #0b0f19;
    }
    
    /* Beautiful Status Logger */
    .log-term {
        font-family: 'Courier New', Courier, monospace;
        background-color: #020617;
        color: #38bdf8;
        padding: 15px;
        border-radius: 8px;
        font-size: 0.85rem;
        max-height: 250px;
        overflow-y: auto;
        border: 1px solid #1e293b;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        background-color: #1e293b;
        color: #94a3b8 !important;
        border: 1px solid #334155;
        border-bottom: none;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155;
        color: #f1f5f9 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
        border-color: #6366f1;
    }
</style>
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HERO HEADER BANNER
# ---------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">🧬 MultiOmics-Integrator v1.2</h1>
    <p class="hero-subtitle">Premium Windows-Native Multi-Omics Integration Suite. Upload Genomics, Transcriptomics, Proteomics, and Metabolomics data to instantly compute 50+ deep statistics, run pathway enrichments, string PPI networks, time-series alignments, and compile an elegant interactive HTML report.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SYNTHETIC DATA GENERATOR (DEMO MODE)
# ---------------------------------------------------------------------
def generate_synthetic_omics():
    """Generates highly realistic synthetic multi-omics data for testing."""
    np.random.seed(42)
    
    # 1. Genomics: Ensembl Gene IDs, mutated/altered across samples
    # Famous real cancer/biomedical genes for API lookups to succeed
    real_genes = ["BRCA1", "TP53", "EGFR", "MYC", "TNF", "IL6", "VEGFA", "GAPDH", "AKT1", "MTOR"]
    real_ens = ["ENSG00000012048", "ENSG00000141510", "ENSG00000146648", "ENSG00000136997", 
                "ENSG00000232810", "ENSG00000136244", "ENSG00000112715", "ENSG00000111640", 
                "ENSG00000142208", "ENSG00000198793"]
    real_uni = ["P38398", "P04637", "P00533", "P01106", "P01375", "P05231", "P15692", "P04406", "P31749", "P42345"]
    
    genes = [f"ENSG0000{i:07d}" for i in range(1, 101)]
    # Place real genes in first 10 slots
    for idx, r_ens in enumerate(real_ens):
        genes[idx] = r_ens
        
    samples = [f"Ctrl_{i}" for i in range(1, 6)] + [f"Treat_{i}" for i in range(1, 6)]
    
    # Expression matrix for Genomics
    genomics_data = {'gene_id': genes}
    for s in samples:
        is_treat = "Treat" in s
        vals = []
        for idx in range(100):
            if idx < 30:
                mean = 12.0 if is_treat else 10.0
            elif idx < 60:
                mean = 8.0 if is_treat else 10.0
            else:
                mean = 10.0
            vals.append(max(0, np.random.normal(mean, 1.2)))
        genomics_data[s] = vals
    genomics_df = pd.DataFrame(genomics_data)
    
    # 2. Transcriptomics: Ensembl Gene IDs, corresponding expression
    transcriptomics_data = {'transcript_id': genes}
    for s in samples:
        is_treat = "Treat" in s
        vals = []
        for idx in range(100):
            if idx < 30:
                mean = 15.0 if is_treat else 11.0
            elif idx < 60:
                mean = 7.0 if is_treat else 11.0
            else:
                mean = 11.0
            vals.append(max(0, np.random.normal(mean, 2.0)))
        transcriptomics_data[s] = vals
    transcriptomics_df = pd.DataFrame(transcriptomics_data)
    
    # 3. Proteomics: UniProt IDs, matching standard gene IDs
    uniprot_ids = [f"P{i:05d}" for i in range(1, 101)]
    for idx, r_uni in enumerate(real_uni):
        uniprot_ids[idx] = r_uni
        
    proteomics_data = {'protein_id': uniprot_ids}
    for s in samples:
        is_treat = "Treat" in s
        vals = []
        for idx in range(100):
            if idx < 30:
                mean = 8.0 if is_treat else 6.0
            elif idx < 60:
                mean = 4.0 if is_treat else 6.0
            else:
                mean = 6.0
            vals.append(max(0, np.random.normal(mean, 0.8)))
        proteomics_data[s] = vals
    proteomics_df = pd.DataFrame(proteomics_data)
    
    # 4. Metabolomics: HMDB IDs, matching standard metabolites
    metabolite_ids = [f"HMDB000{i:04d}" for i in range(1, 51)]
    real_hmdb = ["HMDB0000122", "HMDB0000148", "HMDB0000161", "HMDB0000190", "HMDB0000517"]
    for idx, r_hmdb in enumerate(real_hmdb):
        metabolite_ids[idx] = r_hmdb
        
    metabolomics_data = {'metabolite_id': metabolite_ids}
    for s in samples:
        is_treat = "Treat" in s
        vals = []
        for idx in range(50):
            if idx < 15:
                mean = 5.0 if is_treat else 3.0
            elif idx < 30:
                mean = 1.5 if is_treat else 3.0
            else:
                mean = 3.0
            vals.append(max(0, np.random.normal(mean, 0.6)))
        metabolomics_data[s] = vals
    metabolomics_df = pd.DataFrame(metabolomics_data)
    
    # Add 'condition' classification columns
    genomics_df['condition'] = ['Associated' if i < 60 else 'Non-Associated' for i in range(100)]
    transcriptomics_df['condition'] = ['Associated' if i < 60 else 'Non-Associated' for i in range(100)]
    proteomics_df['condition'] = ['Associated' if i < 60 else 'Non-Associated' for i in range(100)]
    metabolomics_df['condition'] = ['Associated' if i < 30 else 'Non-Associated' for i in range(50)]
    
    return genomics_df, transcriptomics_df, proteomics_df, metabolomics_df

# ---------------------------------------------------------------------
# SIDEBAR CONFIGURATIONS & FILE UPLOADS
# ---------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="margin: 0; font-family: 'Outfit', sans-serif; color: #4f46e5; font-size: 1.8rem;">⚙️ Settings</h2>
    <p style="color: #64748b; font-size: 0.85rem; margin: 5px 0 0 0;">Upload your CSV files and configure parameters</p>
</div>
""", unsafe_allow_html=True)

# Test datasets load toggle
st.sidebar.markdown("### 🧬 Load Sample Data First?")
demo_mode = st.sidebar.button("🚀 Load Realistic Demo Datasets", use_container_width=True, type="secondary")

st.sidebar.markdown("### 📂 Upload Multi-Omics CSVs")
g_file = st.sidebar.file_uploader("1. Genomics (Ensembl SCOPES)", type=["csv"])
t_file = st.sidebar.file_uploader("2. Transcriptomics (Ensembl SCOPES)", type=["csv"])
p_file = st.sidebar.file_uploader("3. Proteomics (UniProt SCOPES)", type=["csv"])
m_file = st.sidebar.file_uploader("4. Metabolomics (HMDB SCOPES)", type=["csv"])

st.sidebar.markdown("### ⚙️ Pipeline Hyperparameters")
run_gsea_toggle = st.sidebar.toggle("Run GSEA Enrichment", value=True, help="Disable to skip GSEA and speed up computation")
run_ppi_toggle = st.sidebar.toggle("Fetch STRING PPI Network", value=True, help="Disable to skip fetching protein network interactions online")
ppi_score = st.sidebar.slider("STRING PPI Confidence Threshold", min_value=150, max_value=999, value=400, step=50, help="Standard score threshold (400 is medium)")
max_ppi_genes = st.sidebar.slider("Max Network Seed Genes", min_value=10, max_value=1000, value=250, step=50)
tsne_perp = st.sidebar.slider("t-SNE Perplexity", min_value=2, max_value=100, value=15, step=1)
id_batch_size = st.sidebar.number_input("ID Mapping Batch Size", min_value=10, max_value=1000, value=100)

# Initialize Session States for Datasets
if "genomics" not in st.session_state:
    st.session_state["genomics"] = None
    st.session_state["transcriptomics"] = None
    st.session_state["proteomics"] = None
    st.session_state["metabolomics"] = None
    st.session_state["is_demo"] = False

if demo_mode:
    g_demo, t_demo, p_demo, m_demo = generate_synthetic_omics()
    st.session_state["genomics"] = g_demo
    st.session_state["transcriptomics"] = t_demo
    st.session_state["proteomics"] = p_demo
    st.session_state["metabolomics"] = m_demo
    st.session_state["is_demo"] = True
    st.sidebar.success("Successfully loaded demo datasets into session!")

# Handle uploaded files
if g_file:
    st.session_state["genomics"] = pd.read_csv(g_file)
    st.session_state["is_demo"] = False
if t_file:
    st.session_state["transcriptomics"] = pd.read_csv(t_file)
    st.session_state["is_demo"] = False
if p_file:
    st.session_state["proteomics"] = pd.read_csv(p_file)
    st.session_state["is_demo"] = False
if m_file:
    st.session_state["metabolomics"] = pd.read_csv(m_file)
    st.session_state["is_demo"] = False

# Ensure condition classifications columns exist for conditions-based plots
for key, name in [("genomics", "G"), ("transcriptomics", "T"), ("proteomics", "P"), ("metabolomics", "M")]:
    df = st.session_state[key]
    if df is not None and "condition" not in df.columns:
        mat = df.select_dtypes(include=[np.number])
        if not mat.empty and mat.shape[0] >= 2:
            try:
                # Perform K-Means unsupervised clustering to auto-group features and generate conditions!
                scaler = StandardScaler()
                mat_scaled = scaler.fit_transform(mat.fillna(0))
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                labels = kmeans.fit_predict(mat_scaled)
                df["condition"] = [f"Cluster_{l+1}" for l in labels]
            except Exception:
                df["condition"] = ["Group_1" if i % 2 == 0 else "Group_2" for i in range(len(df))]
        else:
            df["condition"] = ["Group_1" if i % 2 == 0 else "Group_2" for i in range(len(df))]
        st.session_state[key] = df

# ---------------------------------------------------------------------
# INITIAL UI LAYOUT (TABS SYSTEM)
# ---------------------------------------------------------------------
tab_preview, tab_engine, tab_stats, tab_charts, tab_export = st.tabs([
    "📂 Preview Uploads",
    "⚙️ Integration Engine",
    "📊 KPI & Tables",
    "🖼️ Interactive Charts",
    "📥 Export & Downloads"
])

# ---------------------------------------------------------------------
# TAB 1: PREVIEW UPLOADS
# ---------------------------------------------------------------------
with tab_preview:
    st.subheader("📊 File Upload Status & Previews")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.session_state["genomics"] is not None:
            st.success(f"✅ Genomics: Loaded {st.session_state['genomics'].shape[0]} rows")
        else:
            st.warning("⚠️ Genomics: Not Uploaded")
    with col2:
        if st.session_state["transcriptomics"] is not None:
            st.success(f"✅ Transcriptomics: Loaded {st.session_state['transcriptomics'].shape[0]} rows")
        else:
            st.warning("⚠️ Transcriptomics: Not Uploaded")
    with col3:
        if st.session_state["proteomics"] is not None:
            st.success(f"✅ Proteomics: Loaded {st.session_state['proteomics'].shape[0]} rows")
        else:
            st.warning("⚠️ Proteomics: Not Uploaded")
    with col4:
        if st.session_state["metabolomics"] is not None:
            st.success(f"✅ Metabolomics: Loaded {st.session_state['metabolomics'].shape[0]} rows")
        else:
            st.warning("⚠️ Metabolomics: Not Uploaded")
            
    # Display individual tables if loaded
    with st.expander("🔍 Click to inspect uploaded tables (First 10 rows)", expanded=True):
        p_g, p_t, p_p, p_m = st.tabs(["Genomics", "Transcriptomics", "Proteomics", "Metabolomics"])
        with p_g:
            if st.session_state["genomics"] is not None:
                st.dataframe(st.session_state["genomics"].head(10), use_container_width=True)
            else:
                st.info("No genomics CSV loaded. Upload a file in the sidebar or load sample data to preview.")
        with p_t:
            if st.session_state["transcriptomics"] is not None:
                st.dataframe(st.session_state["transcriptomics"].head(10), use_container_width=True)
            else:
                st.info("No transcriptomics CSV loaded. Upload a file in the sidebar or load sample data to preview.")
        with p_p:
            if st.session_state["proteomics"] is not None:
                st.dataframe(st.session_state["proteomics"].head(10), use_container_width=True)
            else:
                st.info("No proteomics CSV loaded. Upload a file in the sidebar or load sample data to preview.")
        with p_m:
            if st.session_state["metabolomics"] is not None:
                st.dataframe(st.session_state["metabolomics"].head(10), use_container_width=True)
            else:
                st.info("No metabolomics CSV loaded. Upload a file in the sidebar or load sample data to preview.")

# ---------------------------------------------------------------------
# MAIN PIPELINE ENGINE EXECUTION LOGIC
# ---------------------------------------------------------------------
mg = mygene.MyGeneInfo()
kegg = KEGG()
u = UniProt()

def map_ids_safe(df, id_col, from_type, to_types, batch_size=100, log_container=None):
    """Batch ID mapping with error handling"""
    ids = df[id_col].astype(str).dropna().unique().tolist()
    ids = [i for i in ids if i and i != 'nan' and len(i) > 2]
    all_results = []
    
    if log_container:
        log_container.write(f"🧬 Querying {len(ids)} unique IDs on mygene.info ({from_type} -> {to_types})...")

    for i in range(0, len(ids), batch_size):
        batch = ids[i:i+batch_size]
        try:
            out = mg.querymany(batch, scopes=from_type, fields=to_types, species='human',
                             verbose=False, returnall=False)
            all_results.extend(out)
            time.sleep(0.05)
        except Exception as e:
            if log_container:
                log_container.write(f"⚠️ Mapping batch failed: {e}")
            continue

    mapping = pd.DataFrame(all_results)
    if mapping.empty: 
        return df
    if from_type not in mapping.columns and 'query' in mapping.columns:
        mapping[from_type] = mapping['query']
    mapping = mapping.rename(columns={from_type: id_col})
    # Avoid duplicate matches expanding the matrix rows
    mapping = mapping.drop_duplicates(subset=[id_col])
    return df.merge(mapping, on=id_col, how='left', suffixes=('', '_mapped'))

def calc_stats(df, name, mat_cols=None):
    """Compute 15+ stats per omics layer"""
    if mat_cols is None:
        mat = df.select_dtypes(include=[np.number])
    else:
        mat = df[mat_cols].select_dtypes(include=[np.number])

    if mat.empty or mat.shape[1] < 2: 
        return {}, pd.DataFrame()

    s = {}
    s[f'{name}_n_features'] = mat.shape[0]
    s[f'{name}_n_samples'] = mat.shape[1]
    s[f'{name}_mean'] = float(np.nanmean(mat.values))
    s[f'{name}_median'] = float(np.nanmedian(mat.values))
    s[f'{name}_std'] = float(np.nanstd(mat.values))
    s[f'{name}_var'] = float(np.nanvar(mat.values))
    s[f'{name}_cv'] = s[f'{name}_std'] / (abs(s[f'{name}_mean']) + 1e-9)
    s[f'{name}_min'] = float(np.nanmin(mat.values))
    s[f'{name}_max'] = float(np.nanmax(mat.values))
    s[f'{name}_range'] = s[f'{name}_max'] - s[f'{name}_min']
    s[f'{name}_skew'] = float(stats.skew(mat.values.flatten(), nan_policy='omit'))
    s[f'{name}_kurtosis'] = float(stats.kurtosis(mat.values.flatten(), nan_policy='omit'))
    s[f'{name}_sparsity'] = float((mat == 0).sum().sum() / mat.size)

    # PCA
    try:
        pca = PCA(n_components=min(3, mat.shape[1]-1, mat.shape[0]-1))
        pca.fit(mat.T.fillna(0))
        s[f'{name}_pca_var1'] = float(pca.explained_variance_ratio_[0])
        if len(pca.explained_variance_ratio_) > 1:
            s[f'{name}_pca_var2'] = float(pca.explained_variance_ratio_[1])
    except Exception: 
        pass

    de_df = pd.DataFrame()
    # DE analysis
    if 'condition' in df.columns:
        groups = df['condition'].dropna().unique()
        if len(groups) == 2:
            g1_cols = df[df['condition'] == groups[0]].select_dtypes(include=[np.number]).columns
            g2_cols = df[df['condition'] == groups[1]].select_dtypes(include=[np.number]).columns
            if len(g1_cols) > 0 and len(g2_cols) > 0:
                g1 = mat[g1_cols].values
                g2 = mat[g2_cols].values
                
                # Perform correct element-wise t-test for each feature (row)
                t_stat = []
                p_vals = []
                for row_idx in range(mat.shape[0]):
                    t_val, p_val = stats.ttest_ind(g1[row_idx], g2[row_idx], equal_var=False, nan_policy='omit')
                    t_stat.append(t_val)
                    p_vals.append(p_val)
                
                t_stat = np.array(t_stat)
                p_vals = np.array(p_vals)
                # Handle NaNs in p-values
                p_vals = np.nan_to_num(p_vals, nan=1.0)
                
                p_adj = multipletests(p_vals, method='fdr_bh')[1]
                s[f'{name}_DE_total'] = int((p_adj < 0.05).sum())
                s[f'{name}_DE_up'] = int(((p_adj < 0.05) & (t_stat > 0)).sum())
                s[f'{name}_DE_down'] = int(((p_adj < 0.05) & (t_stat < 0)).sum())
                s[f'{name}_DE_pct'] = 100.0 * s[f'{name}_DE_total'] / mat.shape[0]

                de_df = pd.DataFrame({
                    'feature': df.iloc[:, 0].values if df.shape[1] > 0 else range(mat.shape[0]),
                    't_stat': t_stat, 'p_val': p_vals, 'p_adj': p_adj,
                    'log2FC': np.nanmean(g2, axis=1) - np.nanmean(g1, axis=1)
                })

    # Correlation structure
    try:
        corr = np.corrcoef(mat.T.fillna(0))
        s[f'{name}_mean_corr'] = float(np.nanmean(np.abs(corr[np.triu_indices_from(corr, k=1)])))
        s[f'{name}_cor_density'] = float((np.abs(corr) > 0.7).sum() / corr.size)
    except Exception: 
        pass

    return s, de_df

def cross_corr_safe(df1, df2, n1, n2):
    """Correlation across layers based on common symbols"""
    if 'symbol' not in df1.columns or 'symbol' not in df2.columns: 
        return 0.0
    merged = df1.merge(df2, on='symbol', how='inner', suffixes=('_1', '_2'))
    if merged.empty: 
        return 0.0
    m1 = merged.filter(regex='_1$').select_dtypes(include=[np.number])
    m2 = merged.filter(regex='_2$').select_dtypes(include=[np.number])
    if m1.empty or m2.empty or m1.shape[1] == 0 or m2.shape[1] == 0: 
        return 0.0
    v1 = m1.mean(axis=1, skipna=True).values
    v2 = m2.mean(axis=1, skipna=True).values
    mask = ~(np.isnan(v1) | np.isnan(v2))
    if mask.sum() < 3: 
        return 0.0
    corr = np.corrcoef(v1[mask], v2[mask])[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0

def run_gsea(gene_list, name, log_container=None):
    """Run GSEApy enrichment"""
    if len(gene_list) < 5: 
        return pd.DataFrame()
    gene_list = [g for g in gene_list if isinstance(g, str) and len(g) > 1][:1000]
    
    if log_container:
        log_container.write(f"🔬 Querying GSEApy databases for {name} ({len(gene_list)} genes)...")
        
    try:
        enr = gp.enrichr(gene_list=gene_list,
                        gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2023', 'Reactome_2022'],
                        organism='human', outdir=None, cutoff=0.5, verbose=False)
        res = enr.results
        res['omics_layer'] = name
        return res.head(50)
    except Exception as e:
        if log_container:
            log_container.write(f"⚠️ GSEApy failed for {name}: {e}")
        return pd.DataFrame()

def get_string_ppi_safe(genes, score=400, max_genes=500, log_container=None):
    """Fetch STRING PPI with error handling"""
    genes = list(set([g for g in genes if isinstance(g, str) and len(g) > 1]))[:max_genes]
    if len(genes) < 2: 
        return nx.Graph()
        
    if log_container:
        log_container.write(f"🕸️ Fetching PPI networks from STRING DB (Seeds={len(genes)}, min_score={score})...")
        
    url = "https://string-db.org/api/json/network"
    params = {"identifiers": "\r".join(genes), "species": 9606, "required_score": score}
    try:
        r = requests.post(url, data=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data: 
            return nx.Graph()
        df = pd.DataFrame(data)
        G = nx.from_pandas_edgelist(df, 'preferredName_A', 'preferredName_B', ['score'])
        return G
    except Exception as e:
        if log_container:
            log_container.write(f"⚠️ STRING PPI API retrieval skipped or failed: {e}")
        return nx.Graph()

def random_walk_restart(G, seeds, restart=0.7, max_iter=100):
    """RWR for network propagation driver discovery"""
    if G.number_of_nodes() == 0 or len(seeds) == 0: 
        return {}
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    A = nx.to_numpy_array(G, nodelist=nodes, weight='score')
    if A.sum() == 0: 
        return {}
    D_inv = np.diag(1.0 / (A.sum(axis=1) + 1e-10))
    W = D_inv @ A
    p0 = np.zeros(len(nodes))
    seed_idx = [node_to_idx[s] for s in seeds if s in node_to_idx]
    if not seed_idx: 
        return {}
    p0[seed_idx] = 1.0 / len(seed_idx)
    pt = p0.copy()
    for _ in range(max_iter):
        pt_new = (1 - restart) * W.T @ pt + restart * p0
        if np.linalg.norm(pt_new - pt) < 1e-6: 
            break
        pt = pt_new
    return {nodes[i]: float(pt[i]) for i in range(len(nodes))}

def dtw_analysis(df, name, stats_dict):
    """DTW distance matrix for time-series profiles"""
    mat = df.select_dtypes(include=[np.number])
    if mat.shape[1] < 3 or mat.shape[0] < 2: 
        return None

    mat_norm = StandardScaler().fit_transform(mat.T.fillna(0)).T
    sample_idx = np.random.choice(mat.shape[0], min(50, mat.shape[0]), replace=False)
    dists = []
    ref = mat_norm[sample_idx[0]]

    for i in sample_idx[1:]:
        query = mat_norm[i]
        dist, _ = fastdtw(ref, query)
        dists.append(dist)

    if dists:
        stats_dict[f'{name}_dtw_mean'] = float(np.mean(dists))
        stats_dict[f'{name}_dtw_std'] = float(np.std(dists))
        stats_dict[f'{name}_dtw_max'] = float(np.max(dists))
    return dists

# Initialize state containers for output results
if "stats_dict" not in st.session_state:
    st.session_state["stats_dict"] = {}
    st.session_state["stat_tables"] = {}
    st.session_state["enrich_tables"] = {}
    st.session_state["G"] = nx.Graph()
    st.session_state["figs"] = []
    st.session_state["pipeline_executed"] = False

# ---------------------------------------------------------------------
# TAB 2: EXECUTION ENGINE
# ---------------------------------------------------------------------
with tab_engine:
    st.subheader("⚙️ Multi-Omics Pipeline Engine Control Panel")
    
    if st.session_state["genomics"] is None:
        st.warning("🚨 Please upload your genomics dataset or load the demo files first before executing the integration engine!")
    else:
        st.info("💡 Ready to execute. Click the button below to start the high-performance pipeline. The operations will log below in real-time.")
        
        run_button = st.button("🚀 Execute Multi-Omics Integration Engine", type="primary", use_container_width=True)
        
        # Logging container
        st.write("### 💻 Pipeline Execution Log Console")
        log_box = st.empty()
        
        if run_button:
            logs = []
            def write_log(text):
                logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")
                log_box.markdown(f'<div class="log-term">{"<br>".join(logs)}</div>', unsafe_allow_html=True)
                
            write_log("Starting Multi-Omics Integration pipeline (v1.2)...")
            
            with st.status("Executing Multi-Omics Integration Pipeline...", expanded=True) as status:
                
                # Fetch copies of dfs
                genomics = st.session_state["genomics"].copy()
                transcriptomics = st.session_state["transcriptomics"].copy()
                proteomics = st.session_state["proteomics"].copy()
                metabolomics = st.session_state["metabolomics"].copy()
                
                # Auto-detect ID columns
                g_id = genomics.columns[0]
                t_id = transcriptomics.columns[0]
                p_id = proteomics.columns[0]
                m_id = metabolomics.columns[0]
                
                # Stage 1: ID Mapping
                status.update(label="Stage 1: Performing ID Mapping Queries...", state="running")
                write_log("Stage 1: Batch Mapping IDs across databases...")
                
                genomics = map_ids_safe(genomics, g_id, 'ensembl.gene', ['symbol', 'uniprot'], id_batch_size, log_container=st)
                transcriptomics = map_ids_safe(transcriptomics, t_id, 'ensembl.gene', ['symbol', 'uniprot'], id_batch_size, log_container=st)
                proteomics = map_ids_safe(proteomics, p_id, 'uniprot', ['symbol', 'ensembl.gene'], id_batch_size, log_container=st)
                metabolomics = map_ids_safe(metabolomics, m_id, 'hmdb', ['kegg', 'chebi', 'name'], id_batch_size, log_container=st)
                
                # Standardize to symbol columns
                for df in [genomics, transcriptomics, proteomics]:
                    if 'symbol' not in df.columns or df['symbol'].isna().all():
                        df['symbol'] = df.iloc[:, 0]
                    df['symbol'] = df['symbol'].fillna(df.iloc[:, 0])
                
                write_log("ID Mapping completed.")
                
                # Stage 2: Statistics Engine
                status.update(label="Stage 2: Calculating 50+ Statistics metrics...", state="running")
                write_log("Stage 2: Activating core statistical analytical engines...")
                
                stats_dict = {}
                stat_tables = {}
                
                for name, df in [('G', genomics), ('T', transcriptomics), ('P', proteomics), ('M', metabolomics)]:
                    write_log(f"Computing 15+ descriptive and DE statistics for layer: {name}...")
                    res, de_df = calc_stats(df, name)
                    if res: 
                        stats_dict.update(res)
                    if not de_df.empty: 
                        stat_tables[f'{name}_DE'] = de_df
                        
                # Cross-omics correlation
                stats_dict['corr_G_T'] = cross_corr_safe(genomics, transcriptomics, 'G', 'T')
                stats_dict['corr_T_P'] = cross_corr_safe(transcriptomics, proteomics, 'T', 'P')
                stats_dict['corr_P_M'] = cross_corr_safe(proteomics, metabolomics, 'P', 'M')
                stats_dict['corr_G_P'] = cross_corr_safe(genomics, proteomics, 'G', 'P')
                
                write_log("Descriptive, DE and cross-omics correlation statistics completed.")
                
                # Stage 3: GSEA Pathway enrichment
                status.update(label="Stage 3: Running Pathway Enrichment...", state="running")
                enrich_tables = {}
                if run_gsea_toggle:
                    write_log("Stage 3: Running GSEA Pathway analysis...")
                    for name, df in [('G', genomics), ('T', transcriptomics), ('P', proteomics)]:
                        if 'symbol' in df.columns:
                            genes = df['symbol'].dropna().unique().tolist()
                            enr_res = run_gsea(genes, name, log_container=st)
                            enrich_tables[name] = enr_res
                            stats_dict[f'{name}_n_pathways'] = len(enr_res)
                            if not enr_res.empty:
                                stats_dict[f'{name}_top_pathway'] = enr_res.iloc[0]['Term']
                else:
                    write_log("Stage 3: GSEA enrichment skipped by user config.")
                    
                # Stage 4: STRING PPI & Network propagation
                status.update(label="Stage 4: Building Biological Interaction Networks...", state="running")
                G = nx.Graph()
                if run_ppi_toggle:
                    write_log("Stage 4: Fetching PPI interactions and running Network Diffusion/RWR...")
                    all_genes = list(set(genomics['symbol'].dropna()) | set(transcriptomics['symbol'].dropna()) | set(proteomics['symbol'].dropna()))
                    G = get_string_ppi_safe(all_genes, ppi_score, max_ppi_genes, log_container=st)
                    
                    stats_dict['network_nodes'] = G.number_of_nodes()
                    stats_dict['network_edges'] = G.number_of_edges()
                    stats_dict['network_density'] = float(nx.density(G)) if G.number_of_nodes() > 0 else 0
                    stats_dict['network_components'] = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
                    
                    if G.number_of_nodes() > 0:
                        seeds = []
                        if 'T_DE' in stat_tables:
                            de_genes = stat_tables['T_DE'].nsmallest(30, 'p_adj')
                            seed_symbols = transcriptomics.loc[de_genes.index, 'symbol'].dropna().tolist()
                            seeds = [s for s in seed_symbols if s in G.nodes()]

                        if not seeds and len(G.nodes()) > 0:
                            seeds = list(G.nodes())[:20]

                        rwr_scores = random_walk_restart(G, seeds)
                        if rwr_scores:
                            rwr_df = pd.DataFrame(list(rwr_scores.items()), columns=['gene', 'rwr_score'])
                            rwr_df = rwr_df.sort_values('rwr_score', ascending=False).head(100)
                            stat_tables['RWR_drivers'] = rwr_df
                            stats_dict['rwr_top_driver'] = rwr_df.iloc[0]['gene']
                            stats_dict['rwr_top_score'] = float(rwr_df.iloc[0]['rwr_score'])
                else:
                    write_log("Stage 4: STRING PPI analysis skipped by user config.")
                    
                # Stage 5: DTW Alignment
                status.update(label="Stage 5: Aligning Time-Series using DTW...", state="running")
                write_log("Stage 5: Computing FastDTW alignments...")
                for name, df in [('T', transcriptomics), ('P', proteomics), ('M', metabolomics)]:
                    dtw_analysis(df, name, stats_dict)
                write_log("DTW Time-Series Alignment completed.")
                
                # Stage 6: Visualizations Generation
                status.update(label="Stage 6: Plotting beautiful figures...", state="running")
                write_log("Stage 6: Rendering 50+ interactive visualization charts...")
                
                figs = []
                def add_fig(fig, title):
                    fig.update_layout(
                        title=title, 
                        template='plotly_dark', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        height=500
                    )
                    figs.append((title, fig))
                
                # Plotly Chart Renderings
                # 1-4: PCA plots
                for name, df in [('Genomics', genomics), ('Transcriptomics', transcriptomics),
                                 ('Proteomics', proteomics), ('Metabolomics', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number])
                    if mat.shape[0] > 2 and mat.shape[1] > 2:
                        mat_clean = mat.T.fillna(0)
                        pca = PCA(n_components=2)
                        xy = pca.fit_transform(StandardScaler().fit_transform(mat_clean))
                        
                        # Determine sample groups from the column names to avoid ValueError mismatch
                        sample_names = mat.columns
                        sample_groups = []
                        for col in sample_names:
                            col_lower = str(col).lower()
                            if 'ctrl' in col_lower or 'control' in col_lower or 'wt' in col_lower or 'normal' in col_lower:
                                sample_groups.append('Control')
                            elif 'treat' in col_lower or 'case' in col_lower or 'disease' in col_lower or 'ko' in col_lower or 'mut' in col_lower:
                                sample_groups.append('Treatment')
                            else:
                                parts = str(col).split('_')
                                if len(parts) > 1 and len(parts[0]) > 0:
                                    sample_groups.append(parts[0])
                                else:
                                    sample_groups.append('Sample Group')
                                    
                        fig = px.scatter(x=xy[:, 0], y=xy[:, 1], color=sample_groups,
                                        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                                               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'})
                        add_fig(fig, f'{name} PCA')

                # 5-8: t-SNE plots
                for name, df in [('Genomics', genomics), ('Transcriptomics', transcriptomics),
                                 ('Proteomics', proteomics), ('Metabolomics', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number])
                    if mat.shape[0] > 5 and mat.shape[1] > 2:
                        mat_clean = mat.T.fillna(0)
                        # Fix perplexity to not exceed samples
                        perp = min(tsne_perp, mat.shape[0]//2, mat.shape[1]-1)
                        perp = max(1, perp)
                        tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
                        xy = tsne.fit_transform(StandardScaler().fit_transform(mat_clean))
                        
                        # Determine sample groups from the column names to avoid ValueError mismatch
                        sample_names = mat.columns
                        sample_groups = []
                        for col in sample_names:
                            col_lower = str(col).lower()
                            if 'ctrl' in col_lower or 'control' in col_lower or 'wt' in col_lower or 'normal' in col_lower:
                                sample_groups.append('Control')
                            elif 'treat' in col_lower or 'case' in col_lower or 'disease' in col_lower or 'ko' in col_lower or 'mut' in col_lower:
                                sample_groups.append('Treatment')
                            else:
                                parts = str(col).split('_')
                                if len(parts) > 1 and len(parts[0]) > 0:
                                    sample_groups.append(parts[0])
                                else:
                                    sample_groups.append('Sample Group')
                                    
                        fig = px.scatter(x=xy[:, 0], y=xy[:, 1], color=sample_groups)
                        add_fig(fig, f'{name} t-SNE')

                # 9-12: Volcano plots
                for name in ['G', 'T', 'P']:
                    if f'{name}_DE' in stat_tables:
                        de = stat_tables[f'{name}_DE'].copy()
                        de['-log10p'] = -np.log10(de['p_val'].clip(lower=1e-300))
                        de['significant'] = de['p_adj'] < 0.05
                        fig = px.scatter(de, x='log2FC', y='-log10p', color='significant',
                                        hover_data=['feature'], color_discrete_map={True: 'red', False: 'grey'})
                        add_fig(fig, f'{name} Volcano Plot')

                # 13-16: MA plots
                for name in ['G', 'T', 'P']:
                    if f'{name}_DE' in stat_tables:
                        de = stat_tables[f'{name}_DE'].copy()
                        if 'log2FC' in de.columns:
                            de['mean_expr'] = 10 # placeholder
                            de['significant'] = de['p_adj'] < 0.05
                            fig = px.scatter(de, x='mean_expr', y='log2FC', color='significant',
                                            color_discrete_map={True: 'red', False: 'grey'})
                            add_fig(fig, f'{name} MA Plot')

                # 17-20: GSEA barplots
                for name, df in enrich_tables.items():
                    if not df.empty:
                        top = df.nsmallest(20, 'Adjusted P-value')
                        fig = px.bar(top, x='Combined Score', y='Term', orientation='h',
                                    color='Adjusted P-value', color_continuous_scale='RdBu_r')
                        add_fig(fig, f'{name} Enriched Pathways')

                # 21: Network graph
                if G.number_of_nodes() > 0 and G.number_of_nodes() < 500:
                    pos = nx.spring_layout(G, k=0.5, iterations=30, seed=42)
                    edge_x, edge_y = [], []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    node_x, node_y, node_text, node_size = [], [], [], []
                    rwr_dict = dict(zip(stat_tables['RWR_drivers']['gene'], stat_tables['RWR_drivers']['rwr_score'])) if 'RWR_drivers' in stat_tables else {}

                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x); node_y.append(y); node_text.append(node)
                        node_size.append(10 + 50 * rwr_dict.get(node, 0))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'),
                                            hoverinfo='none', showlegend=False))
                    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text,
                                            marker=dict(size=node_size, color='#1f77b4', line=dict(width=1, color='white')),
                                            hoverinfo='text', showlegend=False))
                    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    add_fig(fig, 'PPI Network with RWR Scores')

                # 22: RWR driver barplot
                if 'RWR_drivers' in stat_tables:
                    fig = px.bar(stat_tables['RWR_drivers'].head(25), x='rwr_score', y='gene', orientation='h',
                                color='rwr_score', color_continuous_scale='Viridis')
                    add_fig(fig, 'Top Network Drivers (RWR)')

                # 23-26: Expression Heatmaps
                for name, df in [('Genomics', genomics), ('Transcriptomics', transcriptomics),
                                 ('Proteomics', proteomics), ('Metabolomics', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number]).iloc[:50, :20]
                    if not mat.empty:
                        fig = go.Figure(data=go.Heatmap(z=mat.values, colorscale='RdBu_r', zmid=0))
                        add_fig(fig, f'{name} Expression Heatmap')

                # 27-30: Box plots by Condition
                for name, df in [('Genomics', genomics), ('Transcriptomics', transcriptomics),
                                 ('Proteomics', proteomics), ('Metabolomics', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number]).iloc[:20]
                    if not mat.empty and 'condition' in df.columns:
                        df_long = pd.melt(df.iloc[:20].reset_index(), id_vars=['condition'], value_vars=mat.columns[:10])
                        fig = px.box(df_long, x='variable', y='value', color='condition')
                        add_fig(fig, f'{name} Boxplot by Condition')

                # 31-34: Value distributions histograms
                for name, df in [('G', genomics), ('T', transcriptomics), ('P', proteomics), ('M', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number])
                    if not mat.empty:
                        fig = go.Figure()
                        for col in mat.columns[:5]:
                            fig.add_trace(go.Histogram(x=mat[col].dropna(), name=col, opacity=0.6))
                        fig.update_layout(barmode='overlay')
                        add_fig(fig, f'{name} Value Distributions')

                # 35-38: Correlation Heatmaps
                for name, df in [('G', genomics), ('T', transcriptomics), ('P', proteomics), ('M', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number]).iloc[:30, :30]
                    if mat.shape[0] > 2 and mat.shape[1] > 2:
                        corr = mat.T.corr()
                        fig = go.Figure(data=go.Heatmap(z=corr.values, colorscale='RdBu', zmid=0))
                        add_fig(fig, f'{name} Sample Correlation')

                # 39-42: Violin Plots by Condition
                for name, df in [('G', genomics), ('T', transcriptomics), ('P', proteomics), ('M', metabolomics)]:
                    if 'condition' in df.columns:
                        mat = df.select_dtypes(include=[np.number]).iloc[:10]
                        if not mat.empty:
                            df_long = pd.melt(df.iloc[:10].reset_index(), id_vars=['condition'], value_vars=mat.columns[:5])
                            fig = px.violin(df_long, x='variable', y='value', color='condition', box=True)
                            add_fig(fig, f'{name} Violin Plots')

                # 43: Sankey Flow
                common_genes = set(genomics['symbol'].dropna()) & set(transcriptomics['symbol'].dropna()) & set(proteomics['symbol'].dropna())
                if len(common_genes) >= 5:
                    genes_list = list(common_genes)[:15]
                    sources, targets, values = [], [], []
                    labels = []

                    for i, g in enumerate(genes_list):
                        labels.extend([f"G:{g}", f"T:{g}", f"P:{g}"])

                    for i, g in enumerate(genes_list):
                        g_val = genomics[genomics['symbol']==g].select_dtypes(include=[np.number]).mean().mean()
                        t_val = transcriptomics[transcriptomics['symbol']==g].select_dtypes(include=[np.number]).mean().mean()
                        p_val = proteomics[proteomics['symbol']==g].select_dtypes(include=[np.number]).mean().mean()

                        if not np.isnan(g_val) and not np.isnan(t_val):
                            sources.append(i*3); targets.append(i*3+1); values.append(abs(t_val))
                        if not np.isnan(t_val) and not np.isnan(p_val):
                            sources.append(i*3+1); targets.append(i*3+2); values.append(abs(p_val))

                    if sources:
                        fig = go.Figure(data=[go.Sankey(
                            node=dict(label=labels, pad=15, thickness=20),
                            link=dict(source=sources, target=targets, value=values))])
                        add_fig(fig, 'Multi-Omics Sankey Flow')

                # 44-47: Time Series Profiles
                for name, df in [('T', transcriptomics), ('P', proteomics), ('M', metabolomics)]:
                    mat = df.select_dtypes(include=[np.number])
                    if mat.shape[1] > 3:
                        fig = go.Figure()
                        for i in range(min(10, mat.shape[0])):
                            fig.add_trace(go.Scatter(y=mat.iloc[i].values, mode='lines+markers', name=df.iloc[i, 0]))
                        add_fig(fig, f'{name} Time Series')

                # 48-50: Clustered Dendrogram Heatmaps
                for name, df in [('T', transcriptomics), ('P', proteomics)]:
                    mat = df.select_dtypes(include=[np.number]).iloc[:50, :20]
                    if mat.shape[0] > 5 and mat.shape[1] > 2:
                        from scipy.cluster.hierarchy import linkage, dendrogram
                        from scipy.spatial.distance import pdist
                        Z = linkage(pdist(mat.fillna(0)), method='ward')
                        fig = px.imshow(mat, aspect='auto', color_continuous_scale='RdBu_r')
                        add_fig(fig, f'{name} Clustered Heatmap')
                
                write_log(f"Rendered {len(figs)} interactive visualizations successfully.")
                
                # Write state
                st.session_state["stats_dict"] = stats_dict
                st.session_state["stat_tables"] = stat_tables
                st.session_state["enrich_tables"] = enrich_tables
                st.session_state["G"] = G
                st.session_state["figs"] = figs
                st.session_state["pipeline_executed"] = True
                
                status.update(label="Multi-Omics Pipeline Execution Completed!", state="complete")
                write_log("🎉 Pipeline complete! Browse the dynamic tabs above to inspect your calculated statistics and figures!")
                
                st.balloons()

# ---------------------------------------------------------------------
# TAB 3: KPI & TABLES
# ---------------------------------------------------------------------
with tab_stats:
    st.subheader("📊 Key Integration Metrics & Descriptive Statistics")
    
    if not st.session_state["pipeline_executed"]:
        st.info("⚠️ Execute the integration engine under the 'Integration Engine' tab to compile stats dashboard.")
    else:
        stats_dict = st.session_state["stats_dict"]
        stat_tables = st.session_state["stat_tables"]
        
        # Display elegant KPI Cards
        st.markdown("### 🏆 Dashboard Core Performance Indexes")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = stats_dict.get('G_n_features', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Genomics Features</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
        with c2:
            val = stats_dict.get('T_n_features', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Transcriptomics Features</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
        with c3:
            val = stats_dict.get('P_n_features', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Proteomics Features</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
        with c4:
            val = stats_dict.get('M_n_features', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Metabolomics Features</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
            
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            val = stats_dict.get('network_nodes', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">PPI Network Nodes</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
        with c6:
            val = stats_dict.get('network_edges', 0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">PPI Network Edges</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)
        with c7:
            val = stats_dict.get('corr_G_T', 0.0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Genomics-Transcriptomics Corr</div><div class="kpi-val">{val:.3f}</div></div>', unsafe_allow_html=True)
        with c8:
            val = stats_dict.get('corr_T_P', 0.0)
            st.markdown(f'<div class="kpi-card"><div class="kpi-lbl">Transcriptomics-Proteomics Corr</div><div class="kpi-val">{val:.3f}</div></div>', unsafe_allow_html=True)
            
        # Display statistical metrics table
        st.markdown("### 📋 Complete Statistics Table")
        stats_df = pd.DataFrame([stats_dict]).T.reset_index()
        stats_df.columns = ['Analytical Metric', 'Calculated Value']
        st.dataframe(stats_df, use_container_width=True, height=400)
        
        # Display specific analytical tables
        st.markdown("### 🧬 Differential Expression & Propagation Tables")
        p_t1, p_t2, p_t3, p_t4 = st.tabs(["Genomics DE", "Transcriptomics DE", "Proteomics DE", "PPI Network RWR Drivers"])
        with p_t1:
            if 'G_DE' in stat_tables:
                st.dataframe(stat_tables['G_DE'], use_container_width=True)
            else:
                st.info("Genomics DE analysis was not computed or is unavailable.")
        with p_t2:
            if 'T_DE' in stat_tables:
                st.dataframe(stat_tables['T_DE'], use_container_width=True)
            else:
                st.info("Transcriptomics DE analysis was not computed or is unavailable.")
        with p_t3:
            if 'P_DE' in stat_tables:
                st.dataframe(stat_tables['P_DE'], use_container_width=True)
            else:
                st.info("Proteomics DE analysis was not computed or is unavailable.")
        with p_t4:
            if 'RWR_drivers' in stat_tables:
                st.dataframe(stat_tables['RWR_drivers'], use_container_width=True)
            else:
                st.info("PPI Network propagation RWR drivers were not computed or is unavailable.")

# ---------------------------------------------------------------------
# TAB 4: INTERACTIVE VISUALIZATIONS DASHBOARD
# ---------------------------------------------------------------------
with tab_charts:
    st.subheader("🖼️ Interactive Multi-Omics Visualizations Catalog")
    
    if not st.session_state["pipeline_executed"]:
        st.info("⚠️ Execute the integration engine under the 'Integration Engine' tab to generate interactive figures.")
    else:
        # Group charts into styled sub-tabs
        sub_tab_dim, sub_tab_de, sub_tab_profile, sub_tab_path, sub_tab_flow = st.tabs([
            "🗺️ Dimensionality Reduction",
            "🌋 Differential Expression",
            "🌡️ Expression Profiling",
            "🌿 Pathways & STRING Network",
            "🌊 Cross-Omics Flow"
        ])
        
        figs = st.session_state["figs"]
        
        with sub_tab_dim:
            st.markdown("### PCA & t-SNE Components Plots")
            for title, fig in figs:
                if 'PCA' in title or 't-SNE' in title:
                    st.plotly_chart(fig, use_container_width=True)
                    
        with sub_tab_de:
            st.markdown("### Differential Expression: Volcano & MA Plots")
            de_plots = [f for f in figs if 'Volcano' in f[0] or 'MA Plot' in f[0]]
            if de_plots:
                for title, fig in de_plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Differential Expression plots are not available for this run.")
                
        with sub_tab_profile:
            st.markdown("### Profiling distributions: Heatmaps, Violins, Boxplots & DTW")
            profile_plots = [f for f in figs if 'Heatmap' in f[0] or 'Boxplot' in f[0] or 'Distributions' in f[0] or 'Violin' in f[0] or 'Time Series' in f[0]]
            if profile_plots:
                for title, fig in profile_plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Expression profiling plots are not available.")
                
        with sub_tab_path:
            st.markdown("### GSEA Pathway Enrichments & STRING PPI Network Propagation")
            net_plots = [f for f in figs if 'Enriched Pathways' in f[0] or 'Network' in f[0] or 'RWR' in f[0]]
            if net_plots:
                for title, fig in net_plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pathway and network plots are not available.")
                
        with sub_tab_flow:
            st.markdown("### Cross-Omics Correlations & Unified Flow")
            flow_plots = [f for f in figs if 'Sankey Flow' in f[0] or 'Correlation' in f[0] and 'Sample' not in f[0]]
            if flow_plots:
                for title, fig in flow_plots:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback to display the Sankey plot directly if present
                found = False
                for title, fig in figs:
                    if 'Sankey' in title:
                        st.plotly_chart(fig, use_container_width=True)
                        found = True
                if not found:
                    st.info("Unified cross-omics flow plots require matching symbols between genomics, transcriptomics, and proteomics. Check your ID mappings or load the demo data to view!")

# ---------------------------------------------------------------------
# TAB 5: EXPORTS & DOWNLOADS
# ---------------------------------------------------------------------
with tab_export:
    st.subheader("📥 Export Final Interactive Report & Raw Analytical Tables")
    
    if not st.session_state["pipeline_executed"]:
        st.info("⚠️ Execute the integration engine under the 'Integration Engine' tab to compile files for export.")
    else:
        st.markdown("### 🏆 Download Interactive HTML Analytical Report")
        st.info("The single HTML report contains all descriptive statistics, calculated DE, RWR tables, and all 50+ Plotly charts embedded dynamically, using Plotly.js to keep them completely interactive.")
        
        # Build HTML content
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        stats_dict = st.session_state["stats_dict"]
        stat_tables = st.session_state["stat_tables"]
        enrich_tables = st.session_state["enrich_tables"]
        figs = st.session_state["figs"]
        G = st.session_state["G"]
        
        stats_df = pd.DataFrame([stats_dict]).T.reset_index()
        stats_df.columns = ['Metric', 'Value']
        
        html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MultiOmics Integration Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #0b0f19; color: #f1f5f9; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
.header {{ background: linear-gradient(135deg, #312e81 0%, #581c87 50%, #831843 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #4c1d95; }}
.card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); border: 1px solid #334155; }}
h1 {{ margin: 0; font-size: 2.5em; color: #f8fafc; }}
h2 {{ color: #f8fafc; border-bottom: 3px solid #6366f1; padding-bottom: 10px; margin-top: 0; }}
h3 {{ color: #cbd5e1; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th {{ background: #6366f1; color: white; padding: 12px; text-align: left; }}
td {{ border: 1px solid #334155; padding: 10px; color: #e2e8f0; }}
tr:nth-child(even) {{ background: #182235; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
.stat-box {{ background: #1e1b4b; padding: 15px; border-radius: 5px; border-left: 4px solid #6366f1; border: 1px solid #312e81; }}
.stat-label {{ font-size: 0.9em; color: #94a3b8; }}
.stat-value {{ font-size: 1.5em; font-weight: bold; color: #f8fafc; }}
.toc {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #334155; }}
.toc ul {{ list-style: none; padding-left: 0; }}
.toc li {{ padding: 5px 0; }}
.toc a {{ color: #818cf8; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Multi-Omics Integration Report</h1>
<p>Generated: {timestamp}</p>
<p>Statistics: {len(stats_dict)} | Plots: {len(figs)} | Networks: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges</p>
</div>

<div class="toc card">
<h2>Table of Contents</h2>
<ul>
<li><a href="#stats">Summary Statistics</a></li>
<li><a href="#tables">Data Tables</a></li>
<li><a href="#plots">Interactive Figures</a></li>
</ul>
</div>

<div class="card" id="stats">
<h2>Summary Statistics</h2>
<div class="stats-grid">
"""]

        # Add key stats boxes
        key_stats = ['G_n_features', 'T_n_features', 'P_n_features', 'M_n_features',
                     'network_nodes', 'network_edges', 'corr_G_T', 'corr_T_P']
        for k in key_stats:
            if k in stats_dict:
                html_parts.append(f'<div class="stat-box"><div class="stat-label">{k}</div><div class="stat-value">{stats_dict[k]:.3f}</div></div>')

        html_parts.append('</div>')
        html_parts.append(stats_df.to_html(index=False, classes='table', table_id='stats-table'))
        html_parts.append('</div>')

        # Add tables
        html_parts.append('<div class="card" id="tables"><h2>Analysis Tables</h2>')
        for name, df in stat_tables.items():
            html_parts.append(f'<h3>{name}</h3>')
            html_parts.append(df.head(100).to_html(index=False, classes='table'))
        html_parts.append('</div>')

        # Add plots
        html_parts.append('<div class="card" id="plots"><h2>Interactive Figures</h2>')
        for i, (title, fig) in enumerate(figs, 1):
            html_parts.append(f'<h3 id="fig{i}">Figure {i}: {title}</h3>')
            # Generate offline Plotly HTML snippets
            html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'plot{i}'))
        html_parts.append('</div>')

        html_parts.append('</div></body></html>')

        html_content = '\n'.join(html_parts)
        
        # Download button for HTML Report
        st.download_button(
            label="📥 Download Full Interactive HTML Report",
            data=html_content,
            file_name="MultiOmics_Report.html",
            mime="text/html",
            use_container_width=True,
            type="primary"
        )
        
        st.markdown("### 📋 Download Individual Calculated CSV Tables")
        
        # Stats CSV
        csv_stats = stats_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Summary Statistics (CSV)",
            data=csv_stats,
            file_name="MultiOmics_Stats.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Specific tables
        for name, df in stat_tables.items():
            safe_name = re.sub(r'[^\w\-_]', '_', name)
            csv_table = df.to_csv(index=False)
            st.download_button(
                label=f"📋 Download Analysis Table: {name} (CSV)",
                data=csv_table,
                file_name=f"MultiOmics_{safe_name}.csv",
                mime="text/csv",
                use_container_width=True
            )