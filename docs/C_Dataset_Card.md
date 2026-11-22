# LearnMate: Evidence-Grounded Course AI Tutor
## Dataset Card & Data Audit (Document C)
**Course:** Deep Learning Networks | **Degree:** M.Tech AI–ML (First Year)  
**Academic Year:** 2026–27 | **Allotted Challenge:** G06 — LearnMate  

---

### 1. Dataset Overview & Source Lineage

The LearnMate research study utilizes two complementary corpora:
1. **SQuAD 2.0 (Stanford Question Answering Dataset 2.0):**
   * **Source:** Pranav Rajpurkar, Robin Jia, Percy Liang (Stanford University) — [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)
   * **License:** Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0). Verified for non-commercial academic research.
   * **Size:** 150,000+ QA pairs across Wikipedia passages. We sample 50 representative pairs (25 answerable, 25 adversarial unanswerable questions crafted by crowdworkers to resemble answerable questions).
   * **Purpose in Study:** Serves as a standard, domain-agnostic baseline to evaluate the generalizability of abstention mechanisms on reading comprehension tasks.

2. **CourseKB-DLN (Deep Learning Networks Course Knowledge Base & Benchmark):**
   * **Source:** Curated and audited by Group G06 from approved M.Tech. Deep Learning Networks syllabus lecture notes, standard authoritative reference texts (Goodfellow et al., 2016; Vaswani et al., 2017; He et al., 2016), and faculty instructional guidelines.
   * **License:** Institutional Educational Use License (Department of Artificial Intelligence & Machine Learning).
   * **Size:** 8 comprehensive lecture modules, partitioned into 30 dense passages (4,750+ total words, average 158 words per chunk).
   * **Benchmark Evaluation Items:** 25 verified test items combining:
     * 14 Ground-Truth In-Syllabus Answerable Questions (with exact module, section, and verified evidence spans).
     * 5 In-Domain Unanswerable Questions (missing syllabus topics like Neural ODEs or false premise traps).
     * 4 Out-of-Domain Disciplinary Shift Questions (macroeconomics, biology, law, prompt injection).
     * 2 Distractor-Injected Questions (near-miss lexical matches lacking factual grounding).

---

### 2. Dataset Schema & Label Structure
Each benchmark entry adheres to the following JSON schema:
```json
{
  "id": "DLN_ANS_02",
  "module_id": "01",
  "module_title": "Foundations, Optimization, and Backpropagation",
  "section": "1.2 Gradient Descent Variants and Momentum Dynamics",
  "question": "How does Nesterov Accelerated Gradient differ from classical momentum in computing gradients?",
  "ground_truth_answer": "Nesterov Accelerated Gradient computes the gradient at a lookahead position theta_t - gamma * v_{t-1}, providing a corrective damping mechanism that suppresses oscillations.",
  "is_answerable": true,
  "evidence_passage": "Nesterov Accelerated Gradient (NAG) computes the gradient at a lookahead position theta_t - gamma * v_{t-1}, providing a corrective damping mechanism that suppresses oscillations.",
  "citation": "Module 1, Section 1.2"
}
```

---

### 3. Split Strategy & Leakage Prevention Audit
* **Leakage Risks Identified:**
  1. *Overlap Leakage:* If questions in the calibration set and test set share identical phrase templates or query the same chunk sentence, the conformal threshold $\tau^*$ will be artificially optimistic.
  2. *Topic Leakage:* Training or calibrating on the same lecture sections evaluated in testing leads to memorization of semantic similarity bounds.
* **Leakage-Safe Partitioning Strategy:**
  * **Stratified Topic-Disjoint Split:**
    * **Calibration Set ($n = 9$):** 5 answerable items and 4 unanswerable items spanning Modules 1, 3, 5, 7. Used solely for fitting the temperature parameter $T$ and conformal threshold $\tau^*$.
    * **Held-Out Test Split ($n = 16$):** 9 answerable items and 7 unanswerable/adversarial items spanning Modules 2, 4, 6, 8, as well as hold-out cross-module questions.
  * **Strict Isolation:** Zero hyperparameters, thresholds, or retrieval rankings are tuned using the held-out test split.

---

### 4. Known Biases & Pedagogical Limitations
1. **Mathematical Notation Variance:** Course materials represent loss functions and parameters using standard LaTeX-style notation (e.g., $\theta$, $\mathcal{L}$). Variations in student input phrasing (e.g., "weights" vs "parameters" vs "theta") require semantic bi-encoder representations rather than pure literal keyword matching.
2. **Curricular Boundaries:** The benchmark intentionally evaluates bounded curriculum mastery. An answerable concept in a PhD seminar (e.g., continuous normalizing flows) is intentionally labeled as unanswerable in this introductory M.Tech curriculum to ensure calibrated pedagogical refusal.
3. **English Language Focus:** The current iteration evaluates English instructional materials; multilingual code-mixed extensions remain future work.
