"""
Tutor Response Generator for LearnMate
Constructs pedagogical, evidence-grounded answers with verifiable citations or structured abstentions.
Also generates baseline ungrounded responses to contrast failure modes.
"""
import os
import sys
import re
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

class TutorGenerator:
    """
    Pedagogical Course AI Tutor Generator.
    Produces either a verified, cited explanation or a structured educational abstention.
    """
    def __init__(self):
        pass

    def generate_grounded_response(self, 
                                   query: str, 
                                   retrieved_chunk: Dict[str, Any],
                                   decision: Dict[str, Any],
                                   evidence_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates grounded response when decision['action'] == 'ANSWER',
        or structured educational abstention when decision['action'] == 'ABSTAIN'.
        """
        if decision["should_abstain"]:
            return self._generate_abstention(query, retrieved_chunk, decision)

        citation = retrieved_chunk.get("citation", "Course Lecture Notes")
        module = retrieved_chunk.get("module", "Deep Learning Networks")
        section = retrieved_chunk.get("section", "Core Concepts")
        text = retrieved_chunk.get("text", "")

        # Extract most relevant sentence span as primary evidence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        q_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))
        best_sentence = text[:200]
        max_overlap = -1
        for s in sentences:
            s_tokens = set(re.findall(r"\b\w{3,}\b", s.lower()))
            overlap = len(q_tokens.intersection(s_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sentence = s.strip()

        # Build pedagogical answer
        direct_answer = f"{best_sentence}"
        
        explanation = (
            f"### Verified Explanation\n"
            f"{direct_answer}\n\n"
            f"**Primary Evidence Citation:** `[{citation}]`\n\n"
            f"**Pedagogical Context:**\n"
            f"In {module} ({section}), this principle ensures mathematical stability during training. "
            f"Specifically: *\"{text[:260]}...\"*\n\n"
            f"**Key Takeaway for Students:** Notice how the formulation directly ties into network convergence properties. "
            f"Review the derivations in {section} before implementing this in your assignment pipeline."
        )

        return {
            "mode": "GROUNDED_ANSWER",
            "citation": citation,
            "evidence_span": best_sentence,
            "evidence_agreement_score": decision.get("raw_agreement_score", 1.0),
            "calibrated_confidence": decision.get("calibrated_confidence", 1.0),
            "response_text": explanation
        }

    def _generate_abstention(self, 
                             query: str, 
                             retrieved_chunk: Dict[str, Any], 
                             decision: Dict[str, Any]) -> Dict[str, Any]:
        refusal_type = decision.get("refusal_type", "INSUFFICIENT_EVIDENCE")
        rationale = decision.get("rationale", "")
        module = retrieved_chunk.get("module", "None")
        section = retrieved_chunk.get("section", "None")
        
        abstention_text = (
            f"### LearnMate Tutor Notice: Question Unanswerable from Course Syllabus\n"
            f"**Status:** Abstaining to prevent ungrounded explanation or hallucination.\n\n"
            f"**Reason:** {rationale}\n\n"
        )

        if refusal_type == "CONTRADICTION_DETECTED":
            abstention_text += (
                f"**Pedagogical Guidance:** Your query appears to assume a premise that contradicts core deep learning principles "
                f"covered in `{module} | {section}`. Please re-check the mathematical definitions in the lecture slides."
            )
        elif refusal_type == "OUT_OF_SYLLABUS":
            abstention_text += (
                f"**Pedagogical Guidance:** This concept is outside our approved M.Tech Deep Learning Networks curriculum. "
                f"As an evidence-grounded tutor, I am constrained to provide only verified answers cited from our course materials. "
                f"Please consult external research literature or your academic advisor for topics outside DLN."
            )
        else:
            abstention_text += (
                f"**Pedagogical Guidance:** The course materials available in the syllabus repository do not contain sufficient "
                f"direct evidence to provide a fully supported answer. Closest indexed module: `{module} | {section}`. "
                f"Please raise this topic in instructor office hours or review the supplementary course reading list."
            )

        return {
            "mode": "CALIBRATED_ABSTENTION",
            "citation": "None (Abstained)",
            "evidence_span": "None",
            "refusal_type": refusal_type,
            "evidence_agreement_score": decision.get("raw_agreement_score", 0.0),
            "calibrated_confidence": decision.get("calibrated_confidence", 0.0),
            "response_text": abstention_text
        }

    def generate_naive_rag_response(self, 
                                    query: str, 
                                    retrieved_chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates standard uncalibrated RAG baseline that attempts to answer regardless of evidence support,
        demonstrating hallucinations and sycophantic behavior.
        """
        text = retrieved_chunk.get("text", "")
        citation = retrieved_chunk.get("citation", "Generic Reference")
        
        # Simulates naive LLM hallucinating plausibly around the query terms
        hallucination = (
            f"Based on the course materials in {citation}, the answer to '{query}' is that "
            f"this mechanism operates directly by integrating the parameters into the objective loss function. "
            f"As stated: '{text[:140]}...'. Therefore, the effect is observed across all training iterations."
        )
        return {
            "mode": "NAIVE_RAG_UNVERIFIED",
            "citation": citation,
            "response_text": hallucination,
            "is_hallucination_risk": True
        }
