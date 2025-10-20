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

Future Work
Data Expansion: Expand the dataset to include additional brands (e.g., Giorgio Armani, Gucci) to enhance embedding robustness.

Dashboard Deployment: Deploy the clustering model into an interactive web dashboard (using Dash or Streamlit) to allow dynamic exploration by brand, price range, and aesthetic dimension.

Cross-Category Modeling: Extend the methodology to create cross-category upselling recommendations (e.g., suggesting jewelry visually similar to eyewear).

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
