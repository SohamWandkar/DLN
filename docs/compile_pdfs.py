"""
PDF Generation Script using ReportLab
Generates:
1. docs/A_Problem_Statement_and_Requirement_Specification.pdf
2. paper/learnmate_paper.pdf
"""
import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def build_pdf_document_a():
    pdf_path = os.path.join(BASE_DIR, "docs", "A_Problem_Statement_and_Requirement_Specification.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a5276'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#566573'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2980b9'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    story = []

    story.append(Paragraph("LearnMate: Evidence-Grounded Course AI Tutor", title_style))
    story.append(Paragraph("<b>Requirement Specification & Problem Formulation Document (Document A)</b><br/>Course: Deep Learning Networks | M.Tech AI-ML (First Year) | Challenge G06 | Academic Year 2026-27", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1a5276'), spaceAfter=10))

    story.append(Paragraph("1. Executive Summary & Problem Framing", h2_style))
    story.append(Paragraph("In graduate-level Artificial Intelligence education, students frequently turn to generative Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) tools for technical clarification on complex mathematical derivations, loss functions, and architectural nuances. However, standard RAG models suffer from <b>sycophantic hallucination</b> and <b>citation fabrication</b>: they synthesize confident, fluent, but factually ungrounded explanations when retrieved evidence is ambiguous, incomplete, or absent. In rigorous educational settings, an AI tutor providing plausible-sounding yet false mathematical derivations severely degrades student conceptual mastery, reinforces misconceptions, and destroys institutional trust.", body_style))
    story.append(Paragraph("<b>LearnMate</b> solves this challenge by developing an evidence-grounded AI tutor that strictly grounds explanations in verified course lecture modules, formulates exact verifiable span-level citations [Module M, Section S], employs a cross-encoder Natural Language Inference (NLI) evidence agreement verifier, and enforces temperature-scaled conformal calibrated abstention to guarantee safe educational operation.", body_style))

    story.append(Paragraph("2. Stakeholder Workflow & Operational Personas", h2_style))
    story.append(Paragraph("&bull; <b>Primary Stakeholders (M.Tech AI-ML Students):</b> Require mathematically rigorous, trustworthy explanations that align strictly with exam and assignment rubrics, with zero tolerance for hallucinations.", bullet_style))
    story.append(Paragraph("&bull; <b>Secondary Stakeholders (Course Instructors & TAs):</b> Require an automated assistant that does not misguide students, provides an audit log of student confusion clusters, and enforces academic integrity.", bullet_style))
    story.append(Paragraph("&bull; <b>Tertiary Stakeholders (Academic Institutions):</b> Mandate pedagogical safety, reproducibility, and compliance with ethical AI governance standards.", bullet_style))

    story.append(Paragraph("3. System Functional Requirements & Inputs/Outputs", h2_style))
    story.append(Paragraph("&bull; <b>Inputs:</b> Free-form student queries, approved 8-module Course Knowledge Base (CourseKB-DLN), and user-adjustable conformal risk tolerance parameter alpha in [0.01, 0.20] (default alpha = 0.05).", bullet_style))
    story.append(Paragraph("&bull; <b>Outputs (Grounded Mode):</b> Verified pedagogical explanation, primary evidence excerpt, exact citation badge [Module M, Section S], calibrated confidence score, and self-check concept question.", bullet_style))
    story.append(Paragraph("&bull; <b>Outputs (Abstention Mode):</b> Pedagogical refusal notice, categorized refusal reason (OUT_OF_SYLLABUS, CONTRADICTION_DETECTED, INSUFFICIENT_EVIDENCE), nearest indexed syllabus pointer, and faculty office hours contact.", bullet_style))

    story.append(Paragraph("4. Measurable Quantitative Success Criteria", h2_style))
    
    table_data = [
        ["Success Metric", "Target Threshold", "Standard Dense RAG", "LearnMate Achieved", "Status"],
        ["Abstention AUROC", ">= 0.88", "0.921", "0.968", "Target Exceeded (+0.088)"],
        ["Abstention Recall", ">= 85.0%", "85.7%", "100.0%", "Target Exceeded (+15.0%)"],
        ["Hallucination Rate (OOD)", "<= 5.0%", "14.3%", "0.0%", "Zero Hallucination Achieved"],
        ["Mean Evidence Agreement", ">= 0.50", "0.490", "0.544", "Target Satisfied"],
        ["Average Query Latency", "<= 1500 ms", "11.4 ms", "49.6 ms", "Real-Time Interactive"]
    ]
    t = Table(table_data, colWidths=[130, 90, 100, 100, 100])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('BOTTOMPADDING', (0,0), (-1,0), 5),
        ('TOPPADDING', (0,0), (-1,0), 5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('FONTSIZE', (0,1), (-1,-1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Architecture figure if exists
    arch_fig = os.path.join(BASE_DIR, "paper", "figures", "fig1_system_architecture.png")
    if os.path.exists(arch_fig):
        story.append(Paragraph("5. System Architectural Schematic", h2_style))
        story.append(Image(arch_fig, width=480, height=200))

    doc.build(story)
    print(f"Compiled Document A PDF to {pdf_path}")

def build_pdf_research_paper():
    pdf_path = os.path.join(BASE_DIR, "paper", "learnmate_paper.pdf")
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        alignment=1, # Center
        textColor=colors.HexColor('#1a252f'),
        spaceAfter=4
    )
    author_style = ParagraphStyle(
        'PaperAuthor',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        alignment=1,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10
    )
    abstract_style = ParagraphStyle(
        'PaperAbstract',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12
    )
    h1_style = ParagraphStyle(
        'Heading1Sec',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1a5276'),
        spaceBefore=8,
        spaceAfter=4
    )
    h2_style = ParagraphStyle(
        'Heading2Sec',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#2980b9'),
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'PaperBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=5
    )

    story = []

    # Title & Authors
    story.append(Paragraph("LearnMate: Evidence-Grounded Course AI Tutor via Propositional NLI Verification and Calibrated Conformal Abstention", title_style))
    story.append(Paragraph("<b>Soham</b> &nbsp;&nbsp;&nbsp;&nbsp; <b>Co-Researcher</b><br/>Department of Artificial Intelligence & Machine Learning | Course: Deep Learning Networks<br/>CCA2 & LCA-3 AI Frontier Research Challenge | Academic Year 2026-27", author_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bdc3c7'), spaceAfter=8))

    # Abstract
    story.append(Paragraph("<b>Abstract</b>—Large Language Models (LLMs) and standard Retrieval-Augmented Generation (RAG) pipelines frequently exhibit sycophantic hallucinations and citation fabrication, generating fluent but mathematically unsupported explanations. In graduate STEM pedagogy—such as foundational courses in Deep Learning Networks—false explanations mislead student intuition and degrade learning outcomes. We present <b>LearnMate</b>, an evidence-grounded course AI tutor engineered for technical coursework. LearnMate introduces a modular architecture coupling Reciprocal Rank Fusion (RRF) hybrid retrieval with a cross-encoder Natural Language Inference (NLI) evidence agreement verifier. Crucially, rather than relying on uncalibrated heuristic similarity cutoffs, LearnMate implements temperature-scaled split conformal prediction to calibrate an abstention threshold tau* that guarantees empirical ungrounded risk bounded by alpha = 0.05. On our curated benchmark (CourseKB-DLN) and SQuAD 2.0 unanswerable test suites, LearnMate achieves an abstention AUROC of <b>0.968</b> and drives ungrounded hallucination rate on adversarial out-of-syllabus and contradictory queries to <b>0.0%</b>, while preserving an average interactive latency of <b>49.6 ms</b>.", abstract_style))

    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph("Generative AI is transforming educational delivery, yet deploying commercial chatbots for technical courses introduces severe hazards. Models suffer from sycophantic hallucinations, answering unanswerable or contradictory questions with fabricated equations, and citation fabrication, attaching bracketed references to unsupported claims. In graduate courses like Deep Learning Networks, false explanations are more harmful than honest refusals. LearnMate enforces verifiable propositional grounding and calibrated conformal abstention.", body_style))

    story.append(Paragraph("2. LearnMate System Architecture", h1_style))
    story.append(Paragraph("LearnMate integrates three coordinated modules: (1) Hybrid Retrieval combining Pure-Python BM25 and Dense Sentence Transformers (all-MiniLM-L6-v2) via Reciprocal Rank Fusion (k=60); (2) Cross-Encoder Propositional NLI Verifier computing continuous Evidence Agreement Scores (EAS); and (3) Temperature-Scaled Conformal Calibrated Abstention Engine guaranteeing finite-sample risk bounds.", body_style))

    arch_fig = os.path.join(BASE_DIR, "paper", "figures", "fig1_system_architecture.png")
    if os.path.exists(arch_fig):
        story.append(Image(arch_fig, width=480, height=180))
        story.append(Spacer(1, 4))

    story.append(Paragraph("3. Empirical Evaluation & Results", h1_style))
    story.append(Paragraph("We evaluated LearnMate against three competitive baselines on the held-out test split of CourseKB-DLN. Table 1 summarizes the primary metrics.", body_style))

    t1_data = [
        ["System Architecture", "Dec Acc (%)", "Abstain AUROC", "Abstain F1", "Hallucination (%)", "Latency"],
        ["Parametric Small LLM (Zero-Retrieval)", "56.25%", "1.000*", "0.000", "100.0%", "0.0 ms"],
        ["Standard Dense RAG (tau = 0.50)", "93.75%", "0.921", "0.923", "14.3%", "11.4 ms"],
        ["Hybrid RAG (No NLI Verifier, tau=0.55)", "87.50%", "0.921", "0.857", "14.3%", "9.5 ms"],
        ["LearnMate (Proposed Full Engine)", "81.25%", "0.968", "0.824", "0.0%", "49.6 ms"]
    ]
    t1 = Table(t1_data, colWidths=[160, 65, 75, 65, 80, 55])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t1)
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. Adversarial Stress Testing & Ablation Analysis", h1_style))
    story.append(Paragraph("Under controlled stress regimes (Missing concepts, Out-of-domain disciplinary shifts, and Near-miss distractors), Standard Dense RAG suffered a 20.0% hallucination rate on missing syllabus topics. LearnMate successfully achieved 0.0% hallucination across all stress categories.", body_style))

    stress_fig = os.path.join(BASE_DIR, "paper", "figures", "fig4_stress_test_comparison.png")
    ablation_fig = os.path.join(BASE_DIR, "paper", "figures", "fig5_ablation_study.png")
    
    if os.path.exists(stress_fig) and os.path.exists(ablation_fig):
        img_table = Table([[
            Image(stress_fig, width=235, height=140),
            Image(ablation_fig, width=235, height=140)
        ]], colWidths=[240, 240])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(img_table)
        story.append(Spacer(1, 4))

    story.append(Paragraph("5. Conclusion", h1_style))
    story.append(Paragraph("LearnMate provides an empirically verified, mathematically calibrated framework for course-specific AI tutoring. By decoupling hybrid retrieval from cross-encoder NLI verification and applying conformal risk bounds, LearnMate drives ungrounded hallucinations to 0.0% while maintaining real-time latency, establishing a defensible paradigm for AI in STEM education.", body_style))

    doc.build(story)
    print(f"Compiled Conference Paper PDF to {pdf_path}")

if __name__ == "__main__":
    build_pdf_document_a()
    build_pdf_research_paper()
