# LearnMate: Evidence-Grounded Course AI Tutor
## Requirement Specification & Problem Formulation Document (Document A)
**Course:** Deep Learning Networks | **Degree:** M.Tech AI–ML (First Year)  
**Academic Year:** 2026–27 | **Allotted Challenge:** G06 — LearnMate  
**Application Sector:** Education Technology (Higher STEM / Graduate Pedagogy)  

---

### 1. Executive Summary & Problem Framing
In graduate-level Artificial Intelligence education, students frequently turn to generative Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) tools for technical clarification on complex mathematical derivations, loss functions, and architectural nuances. However, standard RAG models suffer from **sycophantic hallucination** and **citation fabrication**: they synthesize confident, fluent, but factually ungrounded explanations when retrieved evidence is ambiguous, incomplete, or absent. In rigorous educational settings, an AI tutor providing plausible-sounding yet false mathematical derivations severely degrades student conceptual mastery, reinforces misconceptions, and destroys institutional trust.

**LearnMate** solves this critical challenge by developing an evidence-grounded AI tutor that:
1. Strictly grounds all pedagogical explanations in official course lecture notes and textbook modules.
2. Formulates exact, verifiable span-level citations `[Module M, Section S]`.
3. Employs a cross-encoder Natural Language Inference (NLI) evidence agreement verifier.
4. Enforces **temperature-scaled conformal calibrated abstention**: gracefully refusing to answer when evidence is inadequate, off-syllabus, or contradictory, and redirecting the student to appropriate syllabus units or faculty office hours.

---

### 2. Stakeholder Workflow & Operational Personas
```
  [Graduate Student]                    [LearnMate AI Tutor Pipeline]                 [Course Faculty / TAs]
          |                                          |                                           |
          |--- Submits Technical Question ---------->|                                           |
          |    (e.g., "NAG vs Momentum")             |--- Hybrid Retrieval (BM25 + Bi-Encoder)   |
          |                                          |--- NLI Evidence Agreement Verification    |
          |                                          |--- Conformal Risk Evaluation (alpha=0.05) |
          |                                          |                                           |
          |    [EAS >= tau*]                         |                                           |
          |<-- Grounded Answer + Verifiable Citation-|                                           |
          |    (Direct claim + Span citation)        |                                           |
          |                                          |                                           |
          |    [EAS < tau* / Contradiction]          |                                           |
          |<-- Calibrated Refusal + Syllabus Guidance|--- Logs Out-of-Syllabus / Concept Gaps -->|
          |    (Zero hallucination, Office Hr link)  |    (Identifies student confusion topics)  |
```

* **Primary Stakeholders (M.Tech AI-ML Students):** Require mathematically rigorous, trustworthy explanations that align strictly with exam and assignment rubrics, with zero tolerance for hallucinations.
* **Secondary Stakeholders (Course Instructors & TAs):** Require an automated assistant that does not misguide students, provides an audit log of student confusion clusters, and enforces academic integrity.
* **Tertiary Stakeholders (Academic Institutions):** Mandate pedagogical safety, reproducibility, and compliance with ethical AI governance standards.

---

### 3. System Functional Requirements & Inputs/Outputs
* **Input Specifications:**
  * Free-form natural language query $q \in \mathcal{Q}$ submitted by the student.
  * Approved Course Knowledge Base $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$ partitioned into 8 thematic modules with metadata (module, section, page, text).
  * User-configurable Conformal Risk Parameter $\alpha \in [0.01, 0.20]$ (default $\alpha = 0.05$).
* **Output Specifications:**
  * If Evidence Agreement Score $s_{\text{agree}} \ge \tau^*$:
    * Verified factual explanation.
    * Exact citation `[Module M, Section S]`.
    * Direct primary evidence excerpt span.
    * Calibrated confidence score $p_{\text{calib}} \in [0, 1]$.
    * Active pedagogical follow-up concept check.
  * If Evidence Agreement Score $s_{\text{agree}} < \tau^*$:
    * Pedagogical abstention notice ("Question Unanswerable from Course Syllabus").
    * Structured refusal category (`OUT_OF_SYLLABUS`, `CONTRADICTION_DETECTED`, `INSUFFICIENT_EVIDENCE`).
    * Recommendation of nearest indexed syllabus section and office hour contact.

---

### 4. Technical Constraints & Design Boundaries
1. **Zero-Hallucination Tolerance on Unanswerable Queries:** On adversarial out-of-syllabus questions and false premises, hallucination rate must remain below $5\%$.
2. **Compute & Hardware Feasibility:** Must run on standard academic hardware (CPU or single consumer GPU) with average query latency $\le 1.5$ seconds.
3. **No External Internet Querying during Generation:** Retrieval is strictly restricted to locally indexed, faculty-approved course materials to guarantee licensing integrity and curriculum alignment.
4. **Leakage-Safe Partitioning:** Separation of calibration splits from held-out evaluation test sets to prevent data snooping.

---

### 5. Measurable Quantitative Success Criteria
| Success Metric | Target Threshold | Baseline Dense RAG | LearnMate Achieved | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Abstention AUROC** | $\ge 0.88$ | $0.921$ | **$0.968$** | **Target Exceeded (+0.088)** |
| **Abstention Recall (Unanswerables)** | $\ge 85.0\%$ | $85.7\%$ | **$100.0\%$** | **Target Exceeded (+15.0%)** |
| **Hallucination Rate (Unanswerables)** | $\le 5.0\%$ | $14.3\%$ | **$0.0\%$** | **Zero Hallucination Achieved** |
| **Mean Evidence Agreement (Answerable)** | $\ge 0.50$ | $0.490$ | **$0.544$** | **Target Satisfied** |
| **Average Query Latency** | $\le 1500$ ms | $11.4$ ms | **$49.6$ ms** | **Real-Time Interactive** |
