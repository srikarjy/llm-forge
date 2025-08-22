"""
Fixed Genomics AI Quality Scorer with realistic scoring logic for ScientificLLM-Forge.

This module provides functionality for scoring and classifying scientific papers
based on their quality and relevance to genomics + AI/ML research.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class QualityLevel(Enum):
    """Quality classification levels."""
    FILTERED_OUT = "filtered_out"
    LOW_QUALITY = "low_quality"
    MEDIUM_QUALITY = "medium_quality"
    HIGH_QUALITY = "high_quality"
    GOLD_STANDARD = "gold_standard"


@dataclass
class QualityScore:
    """Quality score result for a scientific paper."""
    total_score: float
    quality_level: QualityLevel
    ai_ml_detected: bool
    component_scores: Dict[str, float]
    reasoning: List[str]
    keywords_found: List[str]
    benchmarks_used: List[str]
    validation_methods: List[str]


class GenomicsAIQualityScorer:
    """Score scientific papers for genomics + AI/ML quality and relevance."""
    
    def __init__(self):
        """Initialize the quality scorer."""
        self.logger = logging.getLogger(__name__)
        
        # AI/ML keywords for initial filtering
        self.ai_ml_keywords = [
            "transformer", "deep learning", "neural network", "machine learning",
            "CNN", "BERT", "attention", "convolutional", "pytorch", "tensorflow",
            "artificial intelligence", "foundation model", "pre-trained",
            "fine-tuning", "embedding", "representation learning", "supervised learning"
        ]
        
        # Innovation indicators (high-impact methodological advances)
        self.innovation_patterns = {
            "transformer_genomics": (["transformer", "attention", "BERT"], ["DNA", "genomics", "sequence"], 20),
            "foundation_model": (["pre-trained", "foundation model", "self-supervised"], [], 15),
            "long_range": (["long-range", "100kb", "distant regulatory"], [], 12),
            "interpretability": (["interpretable", "attention map", "visualization", "explainable"], [], 10),
            "novel_architecture": (["novel", "new architecture", "innovative approach"], [], 8)
        }
        
        # Genomics benchmark datasets
        self.genomics_benchmarks = [
            "ENCODE", "1000 Genomes", "TCGA", "GTEx", "gnomAD",
            "CAGE", "ChIP-seq", "ATAC-seq", "eQTL", "GWAS", "UK Biobank"
        ]
        
        # Validation indicators
        self.validation_keywords = [
            "cross-validation", "held-out", "test set", "baseline comparison",
            "state-of-the-art", "ablation study", "statistical significance",
            "p-value", "confidence interval", "benchmarking"
        ]
        
        # Reproducibility indicators
        self.reproducibility_keywords = [
            "code available", "github", "data available", "reproducible",
            "open source", "supplementary materials", "implementation details"
        ]
    
    def is_ai_ml_paper(self, text: str) -> bool:
        """Check if paper contains AI/ML content"""
        text_lower = text.lower()
        ai_ml_count = sum(1 for keyword in self.ai_ml_keywords if keyword in text_lower)
        return ai_ml_count >= 2

    def score_methodological_innovation(self, text: str) -> Tuple[float, List[str]]:
        """
        Score methodological innovation (0-35 points)
        Key insight: Transformer + Genomics = high innovation
        """
        text_lower = text.lower()
        score = 0
        reasons = []
        
        # Check each innovation pattern
        for pattern_name, (keywords, context_words, points) in self.innovation_patterns.items():
            if any(kw in text_lower for kw in keywords):
                # If context words specified, check for them too
                if context_words:
                    if any(ctx in text_lower for ctx in context_words):
                        score += points
                        reasons.append(f"{pattern_name.replace('_', ' ').title()}: {points} pts")
                else:
                    score += points
                    reasons.append(f"{pattern_name.replace('_', ' ').title()}: {points} pts")
        
        # Base innovation bonus for any AI/ML in genomics
        if any(ai in text_lower for ai in ["machine learning", "deep learning"]) and \
           any(bio in text_lower for bio in ["genomics", "DNA", "gene", "genetic"]):
            if score == 0:  # Only if no other innovation detected
                score += 8
                reasons.append("AI/ML applied to genomics: 8 pts")
        
        return min(score, 35), reasons

    def score_benchmark_usage(self, text: str) -> Tuple[float, List[str]]:
        """
        Score benchmark usage (0-30 points)
        Key insight: Standard genomics benchmarks indicate quality
        """
        text_lower = text.lower()
        score = 0
        reasons = []
        found_benchmarks = []
        
        for benchmark in self.genomics_benchmarks:
            if benchmark.lower() in text_lower:
                found_benchmarks.append(benchmark)
        
        benchmark_count = len(found_benchmarks)
        
        if benchmark_count >= 3:
            score = 30
            reasons.append(f"Multiple benchmarks ({benchmark_count}): {', '.join(found_benchmarks[:3])}")
        elif benchmark_count == 2:
            score = 25
            reasons.append(f"Two benchmarks: {', '.join(found_benchmarks)}")
        elif benchmark_count == 1:
            score = 20
            reasons.append(f"Standard benchmark: {found_benchmarks[0]}")
        
        return score, reasons

    def score_validation_rigor(self, text: str) -> Tuple[float, List[str]]:
        """
        Score validation rigor (0-25 points)
        """
        text_lower = text.lower()
        score = 0
        reasons = []
        
        # Statistical rigor
        if any(stat in text_lower for stat in ["p-value", "statistical significance", "confidence interval"]):
            score += 8
            reasons.append("Statistical significance testing: 8 pts")
        
        # Baseline comparisons
        if any(base in text_lower for base in ["baseline", "state-of-the-art", "comparison"]):
            score += 7
            reasons.append("Baseline comparisons: 7 pts")
        
        # Cross-validation
        if any(cv in text_lower for cv in ["cross-validation", "held-out", "validation set"]):
            score += 6
            reasons.append("Proper validation: 6 pts")
        
        # Ablation studies
        if "ablation" in text_lower:
            score += 4
            reasons.append("Ablation studies: 4 pts")
        
        return score, reasons

    def score_reproducibility(self, text: str) -> Tuple[float, List[str]]:
        """
        Score reproducibility (0-20 points)
        """
        text_lower = text.lower()
        score = 0
        reasons = []
        
        if any(code in text_lower for code in ["code available", "github", "open source"]):
            score += 12
            reasons.append("Code availability: 12 pts")
        
        if any(data in text_lower for data in ["data available", "supplementary"]):
            score += 5
            reasons.append("Data sharing: 5 pts")
        
        if any(detail in text_lower for detail in ["implementation details", "reproducible"]):
            score += 3
            reasons.append("Implementation details: 3 pts")
        
        return score, reasons

    def calculate_synergy_bonus(self, text: str, has_benchmarks: bool, innovation_score: float) -> Tuple[float, List[str]]:
        """
        Bonus points for combining AI/ML + genomics + benchmarks
        This addresses the core issue: papers with AI/ML + benchmarks should score higher
        """
        text_lower = text.lower()
        bonus = 0
        reasons = []
        
        # Major bonus: AI/ML + Genomics + Benchmark = high quality
        if has_benchmarks and innovation_score > 0:
            bonus += 15
            reasons.append("AI/ML + Genomics + Benchmark synergy: 15 pts")
        elif has_benchmarks:
            bonus += 10
            reasons.append("Genomics benchmark usage: 10 pts")
        
        # Publication venue bonus
        high_impact_terms = ["nature", "science", "cell", "bioinformatics", "nature methods"]
        if any(term in text_lower for term in high_impact_terms):
            bonus += 5
            reasons.append("High-impact venue: 5 pts")
        
        return bonus, reasons

    def score_paper(self, paper_data: Dict[str, Any]) -> QualityScore:
        """
        Main scoring function with realistic thresholds
        """
        self.logger.info(f"Scoring paper: {paper_data.get('title', 'Unknown')}")
        
        # Extract text for analysis
        title = paper_data.get('title', '')
        abstract = paper_data.get('abstract', '')
        journal = paper_data.get('journal', '')
        keywords = paper_data.get('keywords', [])
        mesh_terms = paper_data.get('mesh_terms', [])
        
        # Combine all text for analysis
        full_text = f"{title} {abstract} {journal} {' '.join(keywords)} {' '.join(mesh_terms)}"
        
        # Filter out non-AI/ML papers
        if not self.is_ai_ml_paper(full_text):
            return QualityScore(
                total_score=0,
                quality_level=QualityLevel.FILTERED_OUT,
                ai_ml_detected=False,
                component_scores={"reason": "Not an AI/ML paper"},
                reasoning=["Traditional biology paper - filtered out"],
                keywords_found=[],
                benchmarks_used=[],
                validation_methods=[]
            )
        
        # Score individual components
        innovation_score, innovation_reasons = self.score_methodological_innovation(full_text)
        benchmark_score, benchmark_reasons = self.score_benchmark_usage(full_text)
        validation_score, validation_reasons = self.score_validation_rigor(full_text)
        reproducibility_score, repro_reasons = self.score_reproducibility(full_text)
        
        # Calculate synergy bonus (this is key for realistic scoring)
        has_benchmarks = benchmark_score > 0
        synergy_score, synergy_reasons = self.calculate_synergy_bonus(
            full_text, has_benchmarks, innovation_score
        )
        
        # Total score
        total_score = innovation_score + benchmark_score + validation_score + reproducibility_score + synergy_score
        
        # Determine tier with realistic thresholds
        if total_score >= 90:
            quality_level = QualityLevel.GOLD_STANDARD
        elif total_score >= 70:
            quality_level = QualityLevel.HIGH_QUALITY
        elif total_score >= 50:
            quality_level = QualityLevel.MEDIUM_QUALITY
        elif total_score >= 30:
            quality_level = QualityLevel.LOW_QUALITY
        else:
            quality_level = QualityLevel.FILTERED_OUT
        
        component_scores = {
            "methodological_innovation": innovation_score,
            "benchmark_usage": benchmark_score,
            "validation_rigor": validation_score,
            "reproducibility": reproducibility_score,
            "synergy_bonus": synergy_score
        }
        
        all_reasons = (innovation_reasons + benchmark_reasons + 
                      validation_reasons + repro_reasons + synergy_reasons)
        
        # Extract keywords and benchmarks for compatibility
        keywords_found = [kw for kw in self.ai_ml_keywords if kw.lower() in full_text.lower()]
        benchmarks_used = [b for b in self.genomics_benchmarks if b.lower() in full_text.lower()]
        validation_methods = [v for v in self.validation_keywords if v.lower() in full_text.lower()]
        
        return QualityScore(
            total_score=total_score,
            quality_level=quality_level,
            ai_ml_detected=True,
            component_scores=component_scores,
            reasoning=all_reasons,
            keywords_found=keywords_found,
            benchmarks_used=benchmarks_used,
            validation_methods=validation_methods
        )
    
    def get_quality_summary(self, score: QualityScore) -> str:
        """Get a human-readable quality summary.
        
        Args:
            score: QualityScore object
            
        Returns:
            Formatted summary string
        """
        summary = f"Quality Score: {score.total_score}/100\n"
        summary += f"Quality Level: {score.quality_level.value.replace('_', ' ').title()}\n"
        summary += f"AI/ML Detected: {'Yes' if score.ai_ml_detected else 'No'}\n\n"
        
        summary += "Component Scores:\n"
        for component, points in score.component_scores.items():
            summary += f"  {component.replace('_', ' ').title()}: {points}\n"
        
        if score.keywords_found:
            summary += f"\nAI/ML Keywords Found: {', '.join(set(score.keywords_found))}\n"
        
        if score.benchmarks_used:
            summary += f"Benchmarks Used: {', '.join(set(score.benchmarks_used))}\n"
        
        if score.validation_methods:
            summary += f"Validation Methods: {', '.join(set(score.validation_methods))}\n"
        
        if score.reasoning:
            summary += f"\nReasoning:\n"
            for reason in score.reasoning:
                summary += f"  • {reason}\n"
        
        return summary 