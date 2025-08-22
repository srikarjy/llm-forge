#!/usr/bin/env python3
"""
Quality Scorer Example for ScientificLLM-Forge.

This example demonstrates how to use the GenomicsAIQualityScorer
to evaluate scientific papers for genomics + AI/ML research.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.quality_scorer import GenomicsAIQualityScorer, QualityLevel
from utils.logger import setup_logger


def main():
    """Demonstrate quality scoring functionality."""
    
    # Setup logging
    logger = setup_logger("quality_scorer_example", level="INFO")
    
    # Initialize the quality scorer
    scorer = GenomicsAIQualityScorer()
    
    logger.info("🧪 Genomics AI Quality Scorer Example")
    logger.info("=" * 60)
    
    # Example 1: DNABERT (Gold Standard)
    logger.info("\n📄 Example 1: DNABERT Paper (Expected: Gold Standard)")
    dnabert_paper = {
        "title": "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers for DNA sequence analysis",
        "abstract": """
        We present DNABERT, a novel foundation model for DNA sequence analysis using transformer architecture. 
        Our model leverages self-attention mechanisms to capture long-range dependencies in genomic sequences. 
        We evaluate DNABERT on multiple benchmark datasets including ENCODE, 1000 Genomes, and TCGA, achieving 
        state-of-the-art performance on various genomics tasks. The model demonstrates strong interpretability 
        through attention visualization and ablation studies. We provide comprehensive cross-validation results 
        with statistical significance testing (p < 0.001). Our code is available on GitHub with detailed 
        methods and data sharing protocols. The model shows novel architectural innovations specifically 
        designed for genomic sequence analysis, including multi-head attention mechanisms adapted for 
        DNA sequence patterns.
        """,
        "keywords": ["transformer", "deep learning", "genomics", "dna sequence", "attention mechanism", "foundation model"],
        "mesh_terms": ["Machine Learning", "Genomics", "DNA", "Neural Networks", "Transformer"]
    }
    
    score = scorer.score_paper(dnabert_paper)
    logger.info(f"Title: {dnabert_paper['title']}")
    logger.info(f"Score: {score.total_score}/100")
    logger.info(f"Quality Level: {score.quality_level.value.replace('_', ' ').title()}")
    logger.info(f"AI/ML Detected: {'Yes' if score.ai_ml_detected else 'No'}")
    
    # Example 2: Enformer (High Quality)
    logger.info("\n📄 Example 2: Enformer Paper (Expected: High Quality)")
    enformer_paper = {
        "title": "Enformer: Predicting gene expression from DNA sequence using attention mechanisms",
        "abstract": """
        We present Enformer, a deep learning model that predicts gene expression from DNA sequence using 
        attention mechanisms to capture long-range interactions. The model is trained on ENCODE data and 
        validated using cross-validation and independent test sets. We compare against baseline methods 
        and demonstrate improved performance. Statistical significance testing shows p < 0.001. 
        The model architecture includes novel attention mechanisms for genomic sequences. 
        Code is available on GitHub with detailed methods.
        """,
        "keywords": ["deep learning", "attention mechanism", "gene expression", "dna sequence", "encode"],
        "mesh_terms": ["Deep Learning", "Gene Expression", "DNA", "Attention", "Neural Networks"]
    }
    
    score = scorer.score_paper(enformer_paper)
    logger.info(f"Title: {enformer_paper['title']}")
    logger.info(f"Score: {score.total_score}/100")
    logger.info(f"Quality Level: {score.quality_level.value.replace('_', ' ').title()}")
    
    # Example 3: Traditional Biology (Filtered Out)
    logger.info("\n📄 Example 3: Traditional Biology Paper (Expected: Filtered Out)")
    traditional_paper = {
        "title": "RRTF1 gene expression analysis using PCR and Western blot techniques",
        "abstract": """
        We investigated the expression of RRTF1 gene in various cell lines using polymerase chain reaction (PCR) 
        and Western blot analysis. Cell culture experiments were performed under standard conditions. 
        Immunohistochemistry was used to localize protein expression. Flow cytometry analysis revealed 
        cell cycle distribution. Statistical analysis was performed using Student's t-test. 
        Our findings demonstrate the role of RRTF1 in cellular processes through traditional molecular biology approaches.
        """,
        "keywords": ["pcr", "western blot", "cell culture", "immunohistochemistry", "flow cytometry"],
        "mesh_terms": ["Polymerase Chain Reaction", "Western Blotting", "Cell Culture", "Immunohistochemistry"]
    }
    
    score = scorer.score_paper(traditional_paper)
    logger.info(f"Title: {traditional_paper['title']}")
    logger.info(f"Score: {score.total_score}/100")
    logger.info(f"Quality Level: {score.quality_level.value.replace('_', ' ').title()}")
    logger.info(f"Reasoning: {score.reasoning[0] if score.reasoning else 'None'}")
    
    # Example 4: Medium Quality Paper
    logger.info("\n📄 Example 4: Medium Quality Paper (Expected: Medium Quality)")
    medium_paper = {
        "title": "Machine learning approach for variant calling in genomic data",
        "abstract": """
        We apply machine learning techniques to variant calling in genomic sequences. 
        The method uses supervised learning with cross-validation. We evaluate on a subset of 
        1000 Genomes data and show improved accuracy compared to existing methods. 
        Statistical testing confirms significance. Code is available upon request.
        """,
        "keywords": ["machine learning", "variant calling", "genomics", "supervised learning"],
        "mesh_terms": ["Machine Learning", "Genomics", "Variation"]
    }
    
    score = scorer.score_paper(medium_paper)
    logger.info(f"Title: {medium_paper['title']}")
    logger.info(f"Score: {score.total_score}/100")
    logger.info(f"Quality Level: {score.quality_level.value.replace('_', ' ').title()}")
    
    # Example 5: No AI/ML Content (Filtered Out)
    logger.info("\n📄 Example 5: No AI/ML Content (Expected: Filtered Out)")
    no_ai_paper = {
        "title": "Analysis of protein structure using X-ray crystallography",
        "abstract": """
        We determined the three-dimensional structure of a protein using X-ray crystallography. 
        Crystals were grown under various conditions and diffraction data was collected. 
        Structure determination was performed using molecular replacement. 
        The results provide insights into protein function and interactions.
        """,
        "keywords": ["x-ray crystallography", "protein structure", "crystallization"],
        "mesh_terms": ["X-Ray Crystallography", "Protein Structure", "Crystallization"]
    }
    
    score = scorer.score_paper(no_ai_paper)
    logger.info(f"Title: {no_ai_paper['title']}")
    logger.info(f"Score: {score.total_score}/100")
    logger.info(f"Quality Level: {score.quality_level.value.replace('_', ' ').title()}")
    logger.info(f"AI/ML Detected: {'Yes' if score.ai_ml_detected else 'No'}")
    
    # Detailed analysis of DNABERT
    logger.info("\n" + "=" * 60)
    logger.info("🔍 Detailed Analysis: DNABERT Paper")
    logger.info("=" * 60)
    
    score = scorer.score_paper(dnabert_paper)
    summary = scorer.get_quality_summary(score)
    logger.info(summary)
    
    # Quality level statistics
    logger.info("\n" + "=" * 60)
    logger.info("📊 Quality Level Summary")
    logger.info("=" * 60)
    
    quality_levels = {
        "Gold Standard (95-100)": "Papers with state-of-the-art methods, major benchmarks, and comprehensive validation",
        "High Quality (80-94)": "Papers with solid AI/ML approaches and good validation",
        "Medium Quality (60-79)": "Papers with basic AI/ML methods and limited validation",
        "Filtered Out (0)": "Papers without AI/ML content or traditional biology only"
    }
    
    for level, description in quality_levels.items():
        logger.info(f"{level}: {description}")
    
    logger.info("\n✅ Quality scoring example completed!")
    logger.info("Use this scorer to filter and rank papers for your genomics + AI/ML research.")


if __name__ == "__main__":
    main() 