👓 Aesthetic Clustering and Pricing Intelligence in Luxury Eyewear
Project Overview
This thesis explores the application of Computer Vision (CV) and Unsupervised Deep Learning to solve the problem of choice overload (analysis paralysis) in luxury e-commerce.

We propose a robust framework that utilizes Vision Transformer (ViT) embeddings combined with Agglomerative Clustering to segment vast product catalogs into visually coherent aesthetic families. The core innovation is the integration of this visual segmentation with pricing intelligence to provide actionable merchandising and competitive insights for luxury brands.

Key Research Question
Can Vision Transformer-based embeddings, when combined with Agglomerative Clustering, generate visually coherent groups of luxury eyewear products that not only offer actionable insights for merchandising and pricing decisions, but also shorten consumers’ cognitive evaluation time and reduce frustration caused by digital choice overload?

Methodology
1. Data Acquisition and Preprocessing
Data Source: 763 product images, names, and prices scraped from the official Italian e-commerce platforms of six luxury brands: Dolce & Gabbana, YSL, Prada, Fendi, Bottega Veneta, and Cartier.

Preprocessing: Robust image normalization steps were applied, including adaptive background crop and centering, to ensure consistency and isolate the product geometry from background noise.

2. Feature Extraction: The Dual Feature Strategy
The pipeline employs a dual feature strategy to generate independent feature sets for benchmarking.

Feature Set	Components	Purpose
Deep Features (ViT Embeddings)	ViT-Base (Masked Autoencoder) embeddings (768D).	Captures high-level semantic and stylistic aesthetic fingerprints (frame curvature, structural integrity).
Handcrafted Features (Benchmark)	HOG, Hu Moments (Shape); HSV Histograms (Color); Gabor Filters, LBP (Texture).	Serves as a baseline to quantitatively validate the incremental value and robustness of the deep learning model.


3. Clustering and Validation
Algorithm: Agglomerative Clustering was chosen for its interpretability and ability to reveal hierarchical aesthetic continuums.

Optimization: The optimal number of clusters (k) was determined using the Silhouette Score.

Price Integration: Cluster-level price statistics (Min/Mean/Max) were computed to quantify the Brand Price Premium associated with aesthetic distinctiveness.

Results and Business Impact
The analysis confirmed that ViT-based clustering consistently achieved superior coherence and separation compared to traditional handcrafted descriptors. The model successfully mapped product aesthetics into distinct, commercially meaningful families.

Key Business Implications:
Merchandising Optimization: Identifies design redundancies (oversaturated clusters) and highlights underrepresented design opportunities.

Pricing Strategy: Aligns price positioning with visual uniqueness and scarcity within the catalog.

Automated Interpretation: LLM AI was utilized to generate interpretations of these visual groupings, which grants a greater degree of automation to the system for future deployment and continuous analysis.

Consumer Experience: Enables the development of visual recommendation engines to reduce cognitive overload during product discovery.

Project Outputs (Visualizations)
The primary outputs are cluster maps and dendrograms, which are essential for visual interpretation.

Appendix A displays the first 7 Clusters, ordered by descending Pairwise Distance (separation) for the Bottega Veneta output.

Appendix B shows the Dendrogram for Cartier.

The remaining Outputs (Cluster Maps for Fendi, YSL, Prada, Dolce & Gabbana) can be reviewed at the following link:

[Insert Link Here to a Google Drive or external repository with all images]

---

# Post-Thesis Extensions

The thesis closed with a static F25 snapshot. After defense, the project was extended into a continuous, deployable competitive-intelligence pipeline. Four extensions are now in production.

## 1. Cross-Season Catalog Comparison (F25 ↔ S26)

A Spring/Summer 26 snapshot was collected from the same brands using the same scraping and preprocessing pipeline (Bottega Veneta n=162, D&G n=114). For each maison, the F25 and S26 catalogues are now compared on three axes:

- **Persistence / Novelty**: cosine similarity matching between embedded products across seasons. Products whose best match scores above a threshold are flagged as *persisted*; the rest are S26 novelties. Catalog renewal rate is reported per brand.
- **Price evolution**: KDE distributions of log-price overlaid per season to surface positioning shifts.
- **Palette delta**: per-product palettes are aggregated and the set of colours new in S26 (and lost from F25) is computed.

## 2. Multi-Backbone Embedding Comparison

Two additional backbones were benchmarked alongside the thesis ViT-MAE:

| Backbone     | Training              | Output | Best for                              |
|--------------|-----------------------|--------|---------------------------------------|
| ViT-MAE      | Pixel reconstruction  | 768-d  | Clustering (replicates thesis output) |
| ViT-DINO     | Self-distillation     | 768-d  | Cross-season product matching         |
| FashionCLIP  | Image-text (Farfetch) | 512-d  | Fashion-domain discrimination         |

MAE features collapse for instance discrimination (best-match cos sim ≈ 0.997 even between visually distinct products) — exactly what makes them strong for aesthetic-family clustering but unsuitable for cross-season matching. DINO and FashionCLIP produce wider spreads suitable for matching, with FashionCLIP additionally capturing fashion-domain semantics (silhouette, material, brand-specific motifs) from its pretraining on ~800K Farfetch image-text pairs.

The three backbones are exposed as a toggle in the frontend, so partitions can be compared side by side per maison and per season.

## 3. Zero-Shot Silhouette Classification via FashionCLIP

The unsupervised clusters serve their scientific purpose but require interpretation. A supervised-by-text classification pipeline was built on top of FashionCLIP to produce semantically named groupings without any labelling effort:

1. **Taxonomy of 15 eyewear silhouettes**: cat-eye, aviator, square, rectangular, round, oval, hexagonal, octagonal, shield, mask, butterfly, browline, navigator, oversized, rimless.
2. Each label is encoded as a text prompt via FashionCLIP's text encoder, placing it in the same 512-d joint embedding space as the product image features.
3. Per product, cos sim is computed against all 15 prompts and the argmax is assigned as the silhouette label.
4. Products with top-1 softmax confidence < 0.30 are routed to an **`experimental`** bucket — the model's own "I don't know" signal, which doubles as a candidate-hit surface (low confidence often coincides with editorial innovation, since visually unusual products don't fit any single named silhouette cleanly).

This is mathematically equivalent to K-Means with k=15 and **labelled, pre-computed centroids** — the text embeddings replace the data-discovered cluster means of unsupervised clustering.

Hand validation across twelve random products confirmed correct silhouette assignment in eleven cases; the remaining product was correctly classified by frame shape but visually dominated by a floral embellishment — a separate decoration axis outside the silhouette taxonomy.

## 4. Production Deployment — Conan Insight Hub

All extensions are exposed through a public landing-page front-end at [conan-insight-hub](https://github.com/Bcruzinfante57/conan-insight-hub), built with TanStack Start + Tailwind + Framer Motion and deployed via Lovable. Three sections consume live JSON exports from this repo:

- **Style Mix**: silhouette buckets per maison × season, with all-colour dots beneath each card (every unique palette bucket present in the cluster, sized by share-pct). Δ vs F25 is highlighted per silhouette.
- **Time Series**: F25 → S26 magazine-style spread, with the thesis-aligned F25 + MAE clusters preserved verbatim and DINO / MAE / FashionCLIP toggleable for Joint and S26 views.
- **Price KDE**: log-price density curves per brand per season, highlighting positioning drift.

The frontend is i18n-aware (Spanish, English, Italian).

---

Future Work

Data Expansion: complete S26 coverage for the four remaining brands (Cartier, Fendi, Prada, YSL); incorporate additional houses (Giorgio Armani, Gucci) to broaden embedding robustness.

K-Means Auto-Naming via FashionCLIP Centroids: combine the granularity of unsupervised clustering with the legibility of named labels. Run K-Means at k=15–25 on FashionCLIP image embeddings to discover sub-style variants (multiple cat-eye sub-lines distinguished by material or colour, for instance), then label each centroid by computing cos sim against the same 15 text prompts used in Section 3. Centroids below the confidence threshold are auto-labelled `experimental`. This unifies unsupervised structure discovery with semantic naming.

Threshold Calibration for FashionCLIP Matching: the matching threshold (currently 0.92, tuned for DINO) needs recalibration for FashionCLIP's tighter distribution (≈ 0.85), driven by valley detection on the cross-season best-match histogram.

Composite Novelty Score: rank "most disruptive" products per drop by combining visual novelty (1 − best cos sim vs own F25), colour novelty, price outlier, and intra-season rarity into a single score. Surfaces the products an editorial buyer should look at first.

Cross-Category Modeling: extend the methodology to cross-category upselling recommendations (e.g., suggesting jewelry visually similar to eyewear via shared FashionCLIP embeddings).

License
[Specify your license here, e.g., MIT License or specify "Proprietary - Research Use Only"]

Acknowledgements
I extend my deep gratitude to those who provided support in pursuing this objective:

Marco Brunitto: for the support, company, and camaraderie.

Frank Pagano: for the closeness and sympathy.

Saverio Serafino: for the openness and good disposition.

Manuela Balli: for the demonstration of trust.

Benedetta Sceppacerca: for the help in the research.

A final note from Benjamín Cruz Infante, a Chilean determined to seize his opportunity in the Luxury Fashion world in Milan. Thank you all from the bottom of my heart.
