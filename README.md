# MultiOmics-Integrator v1.2

## Overview

**MultiOmics-Integrator** is a high‑performance, **dark‑mode Streamlit dashboard** for end‑to‑end multi‑omics data analysis on Windows. It seamlessly integrates genomics, transcriptomics, proteomics, and metabolomics layers, providing:

- Automated identifier standardisation (Ensembl, UniProt, HMDB → HGNC).
- Unsupervised **K‑Means** auto‑condition detection for missing metadata.
- Over **50 descriptive statistical metrics** per omics layer.
- Differential expression with Welch’s t‑test and Benjamini‑Hochberg FDR.
- Dimensionality reduction (PCA, t‑SNE).
- **GSEA** pathway enrichment (KEGG, GO, Reactome).
- **Network propagation** via Random Walk with Restart on STRING PPI networks.
- Fast Dynamic Time Warping (FastDTW) for cross‑omics time‑series alignment.
- Export of **interactive Plotly HTML reports** and a polished **research paper PDF** (`generate_paper.py`).

The UI features glass‑morphism, gradient accents, and transparent Plotly charts for a modern look.

---

## Demo

![Dashboard Mockup](file:///C:/Users/Admin/.gemini/antigravity/brain/657b0a38-5d1f-4590-b5ec-18a5379338c0/dashboard_mockup_1779559571305.png)

---

## Installation

```powershell
# Clone the repository (or copy the project folder)
git clone https://github.com/yourusername/MultiOmics-Integrator.git
cd MultiOmics-Integrator

# Install dependencies (recommended inside a virtualenv)
python -m venv venv
.\\venv\\Scripts\\activate
pip install -r requirements.txt
```

The `requirements.txt` contains:
```
streamlit
plotly
pandas
numpy
scipy
scikit-learn
statsmodels
gseapy
networkx
bioservices
mygene
requests
reportlab
```

## Quick Start

```powershell
# Launch the Streamlit dashboard
streamlit run sanu.py



- Open the Streamlit URL (usually `http://localhost:8501`).
- Upload your multi‑omics CSV matrices.
- Explore the automatically generated plots, tables, and network visualisations.
- Click **Export Report** to download a single HTML file with all Plotly charts.



```

## Architecture Highlights

- **Data Ingestion Layer** – Pandas CSV loader with automatic shape validation.
- **ID Mapping Engine** – `mygene` and `bioservices` APIs resolve Ensembl, UniProt, and HMDB IDs to HGNC symbols.
- **Statistical Engine** – SciPy/Statsmodels compute means, variances, sparsity, CV, Welch’s t‑test, and BH‑FDR.
- **Network & Pathway Engine** – `networkx` builds STRING PPI graphs; RWR diffuses signals.
- **Visualization Layer** – Plotly figures are rendered with transparent backgrounds; CSS provides glass‑morphism.
-

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Ensure code passes linting (`flake8`) and tests (`pytest`).
4. Submit a pull request with a clear description of changes.

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

## Acknowledgements

- **Streamlit** for rapid UI prototyping.
- **Plotly** for interactive visualisation.
- **ReportLab** for PDF generation.
- **MyGeneInfo**, **UniProt**, **HMDB**, **STRING** for biological data integration.

---

*Generated on* `2026-05-23` *by* **MultiOmics‑Integrator** *team.*
