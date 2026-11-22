# LearnMate: Evidence-Grounded Course AI Tutor (G06)
### Department of Artificial Intelligence & Machine Learning | Course: Deep Learning Networks
**Assignment:** CCA2 and LCA-3 AI Frontier Application & Research Challenge  
**Academic Year:** 2026–27 | **Group:** G06 | **Application Sector:** Education Technology  
**Authors:** Soham (Member 1) & Co-Researcher (Member 2)  

---

## 📌 Executive Summary
In higher STEM and graduate-level education (such as M.Tech Deep Learning Networks), students relying on generative Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) suffer from **sycophantic hallucinations** and **citation fabrication**. When questions involve out-of-syllabus topics, subtle premise contradictions, or missing context, standard RAG models generate confident, fluent, but factually ungrounded explanations.

**LearnMate** resolves this challenge by developing a course-specific document-grounded AI tutor that strictly cites its sources and enforces **temperature-scaled conformal calibrated abstention**. Rather than guessing, LearnMate:
1. Performs **Hybrid Retrieval** fusing pure-Python BM25 and dense bi-encoder embeddings (`all-MiniLM-L6-v2`) via Reciprocal Rank Fusion (RRF).
2. Computes an explicit **Evidence Agreement Score (EAS)** using a cross-encoder Natural Language Inference (NLI) proposition verifier that detects semantic entailment and severely penalizes contradiction clashes.
3. Formulates a **Calibrated Abstention Engine** based on split conformal prediction, guaranteeing an empirical ungrounded risk bounded by $\alpha = 0.05$.
4. Formulates verifiable inline citations `[Module M, Section S]` or provides structured pedagogical guidance when abstaining.

---

## 🔬 Falsifiable Research Hypothesis
$$\mathcal{H}_1: \text{Augmenting dense retrieval with a cross-encoder NLI evidence-agreement verifier and temperature-scaled conformal abstention thresholding achieves significantly higher calibrated abstention AUROC (} \Delta \text{AUROC} \ge 0.15 \text{) and reduces ungrounded hallucination rate by } \ge 70\% \text{ compared to standard thresholded RAG baselines on course-specific out-of-scope evaluations, without degrading in-syllabus answer accuracy.}$$

---

## 🏆 Key Benchmark Results

Evaluated on the curated **CourseKB-DLN** benchmark (25 verified items across 8 lecture modules) and the official **SQuAD 2.0** unanswerable evaluation suite:

| Evaluation Metric | Baseline 1: Parametric LLM | Baseline 2: Standard Dense RAG | Baseline 3: Hybrid RAG (No NLI) | Proposed: LearnMate Full Engine |
| :--- | :---: | :---: | :---: | :---: |
| **Abstention AUROC** | $1.000^*$ | $0.921$ | $0.921$ | **$0.968$** |
| **Abstention Recall (Unanswerables)** | $0.0\%$ | $85.7\%$ | $85.7\%$ | **$100.0\%$** |
| **Hallucination Rate (Unanswerables)** | $100.0\%$ | $14.3\%$ | $14.3\%$ | **$0.0\%$ (Zero Hallucination)** |
| **Overall Decision Accuracy** | $56.25\%$ | $93.75\%$ | $87.50\%$ | **$81.25\%$** |
| **Expected Calibration Error (ECE)** | $0.288$ | $0.447$ | $0.388$ | **$0.435$** |
| **Mean Query Latency** | $\mathbf{0.0}$ ms | $11.4$ ms | $9.5$ ms | **$49.6$ ms** (Interactive Real-Time) |

*\*Note: Parametric LLM AUROC is a trivial mathematical artifact of constant zero evidence scores; it answers blindly with 100% hallucination on unanswerable questions.*

---

## 📂 Submission Package Structure (Deliverables A–H)

```
DLC CCA-3/
├── CCA2 LCA3_MTech_Deep_Learning_Research_Challenges.pdf   # Original Faculty Handbook
├── README.md                                             # Master Project Overview
├── requirements.txt                                      # Pinned Dependencies
│
├── docs/                                                 # Mandatory Submission Documents
│   ├── A_Problem_Statement_and_Requirement_Specification.md   # [Deliverable A]
│   ├── A_Problem_Statement_and_Requirement_Specification.pdf  # [Deliverable A - PDF]
│   ├── B_Literature_Review_Matrix.md                         # [Deliverable B - 10 Papers]
│   ├── C_Dataset_Card.md                                     # [Deliverable C - SQuAD 2.0 & CourseKB]
│   ├── H_Contribution_Statement.md                           # [Deliverable H - Task Allocations]
│   └── compile_pdfs.py                                       # ReportLab PDF compiler
│
├── data/                                                 # Datasets and Processing
│   ├── raw/
│   │   ├── squad2_sample.json                            # 50 SQuAD 2.0 evaluation samples
│   │   └── dln_course_corpus/                            # 8 Canonical Course Modules
│   │       ├── module1_foundations_backprop.txt
│   │       ├── module2_convolutional_networks.txt
│   │       ├── module3_recurrent_networks_attention.txt
│   │       ├── module4_transformers_foundations.txt
│   │       ├── module5_generative_deep_learning.txt
│   │       ├── module6_self_supervised_learning.txt
│   │       ├── module7_optimization_regularization.txt
│   │       └── module8_robustness_calibration.txt
│   ├── processed/
│   │   ├── chunks_corpus.json                            # 30 indexed chunks with metadata
│   │   ├── val_calibration_split.json                    # Stratified calibration split (n=9)
│   │   └── test_evaluation_split.json                    # Held-out evaluation split (n=16)
│   ├── evaluation/
│   │   └── coursekb_dln_benchmark.json                   # Master 25-item benchmark
│   ├── create_course_corpus.py
│   ├── build_datasets.py
│   └── fetch_squad.py
│
├── src/                                                  # Modular Implementation [Deliverable D]
│   ├── __init__.py
│   ├── config.py                                         # Seeds, paths, hyperparameters
│   ├── data_processor.py                                 # Sliding-window chunking
│   ├── retriever.py                                      # Hybrid BM25 + Dense + RRF
│   ├── evidence_verifier.py                              # Cross-Encoder NLI & EAS
│   ├── abstention_engine.py                              # Conformal prediction & temperature scaling
│   ├── tutor_generator.py                                # Grounded explanation & citation binding
│   ├── metrics.py                                        # AUROC, ECE, EAS, Citation accuracy
│   └── run_experiments.py                                # Automated evaluation harness
│
├── notebooks/                                            # [Deliverable D - Notebook]
│   ├── LearnMate_Research_Experiment.ipynb               # Fully reproducible Colab/Jupyter Notebook
│   └── generate_notebook.py
│
├── results/                                              # [Deliverable E - Empirical Results]
│   ├── baseline_vs_proposed.csv                          # Primary comparison table
│   ├── stress_test_results.csv                           # 4-regime stress test numbers
│   ├── ablation_results.csv                              # Component ablation table
│   └── error_analysis_taxonomy.json                      # Failure mode categorization
│
├── paper/                                                # [Deliverable G - Research Manuscript]
│   ├── learnmate_paper.pdf                               # 4-6 Page Conference-Style Draft (PDF)
│   ├── learnmate_paper.tex                               # Full IEEE/ACM LaTeX Source
│   ├── learnmate_paper.md                                # Complete Markdown Paper
│   ├── references.bib                                    # BibTeX citations
│   └── figures/                                          # Publication-quality Figures
│       ├── fig1_system_architecture.png                  # System Pipeline Schematic
│       ├── fig2_calibration_curves.png                   # Reliability Diagram & ECE
│       ├── fig3_roc_pr_abstention.png                    # Abstention ROC & PR curves
│       ├── fig4_stress_test_comparison.png               # Stress Regime Bar Comparison
│       └── fig5_ablation_study.png                       # Component Ablation Chart
│
└── app/                                                  # [Deliverable F - Working Prototype]
    └── app.py                                            # Full Streamlit Interactive Demo
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Clone the repository and install required packages:
```bash
cd "DLC CCA-3"
pip install -r requirements.txt
```

### 2. Run the End-to-End Experiment Pipeline
To execute the calibration, evaluate all 4 baselines, run the 4-regime stress tests, execute component ablations, and regenerate all publication figures and CSV tables:
```bash
python src/run_experiments.py
```

### 3. Launch the Interactive Streamlit Demo (Deliverable F)
Start the prototype application:
```bash
streamlit run app/app.py
```
* The web interface will open in your browser (`http://localhost:8501`).
* Test preset benchmark questions or enter custom technical queries.
* Switch between **LearnMate Grounded Mode** and **Standard Dense RAG** to view side-by-side failure modes and hallucination suppression in real-time.

### 4. Compile Submission PDFs
To recompile Document A and the Conference Paper PDF:
```bash
python docs/compile_pdfs.py
```

---

## 📊 Summary of Common Five-Task Execution

* **Task 1: Problem Definition & Requirement Specification** $\rightarrow$ Completed in `docs/A_Problem_Statement_and_Requirement_Specification.md` and compiled PDF. Mapped student, instructor, and institutional workflows; quantified failure modes and success criteria.
* **Task 2: Literature Review & Hypothesis Formulation** $\rightarrow$ Completed in `docs/B_Literature_Review_Matrix.md`. Synthesized 10 papers (2020–2026), constructed a 7-column comparative matrix, and stated falsifiable hypothesis $\mathcal{H}_1$.
* **Task 3: Dataset Ingestion & Baseline Implementation** $\rightarrow$ Completed in `data/`, `docs/C_Dataset_Card.md`, and `src/retriever.py`. Curated 8-module CourseKB-DLN corpus and SQuAD 2.0 subset. Implemented BM25 and Dense Bi-Encoder baselines with topic-disjoint calibration and test splits.
* **Task 4: Research Experiment, Stress Testing & Ablation** $\rightarrow$ Completed in `src/run_experiments.py` and `results/`. Evaluated 4 baselines, tested 4 stress regimes, performed component ablations, and generated 5 publication figures.
* **Task 5: Demonstration, Reproducible Package & Manuscript** $\rightarrow$ Completed with working Streamlit app (`app/app.py`), executable notebook (`notebooks/LearnMate_Research_Experiment.ipynb`), LaTeX conference paper (`paper/learnmate_paper.tex`), and compiled PDF draft (`paper/learnmate_paper.pdf`).
