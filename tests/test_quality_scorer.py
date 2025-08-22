"""
Tests for GenomicsAIQualityScorer.

This module contains tests for the quality scoring functionality,
including test cases for DNABERT and traditional biology papers.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.quality_scorer import GenomicsAIQualityScorer, QualityLevel, QualityScore


class TestGenomicsAIQualityScorer:
    """Test cases for GenomicsAIQualityScorer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = GenomicsAIQualityScorer()
    
    def test_dnabert_paper(self):
        """Test DNABERT paper (should score 95+ and be Gold Standard)."""
        # DNABERT paper example based on the analysis
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
        
        score = self.scorer.score_paper(dnabert_paper)
        
        # DNABERT should be Gold Standard (90+)
        assert score.total_score >= 90, f"DNABERT should score 90+, got {score.total_score}"
        assert score.quality_level == QualityLevel.GOLD_STANDARD, f"DNABERT should be Gold Standard, got {score.quality_level}"
        assert score.ai_ml_detected == True, "DNABERT should have AI/ML detected"
        
        # Check component scores
        assert score.component_scores['methodological_innovation'] >= 20, "Should have high innovation score"
        assert score.component_scores['benchmark_usage'] >= 20, "Should use major benchmarks"
        assert score.component_scores['validation_rigor'] >= 15, "Should have rigorous validation"
        assert score.component_scores['synergy_bonus'] >= 15, "Should have synergy bonus"
        
        # Check for specific keywords and benchmarks
        assert 'transformer' in score.keywords_found, "Should detect transformer"
        assert 'deep learning' in score.keywords_found, "Should detect deep learning"
        assert 'ENCODE' in score.benchmarks_used, "Should use ENCODE benchmark"
        assert '1000 Genomes' in score.benchmarks_used, "Should use 1000 Genomes benchmark"
        assert 'TCGA' in score.benchmarks_used, "Should use TCGA benchmark"
        
        print(f"✅ DNABERT scored {score.total_score}/100 - {score.quality_level.value}")
    
    def test_traditional_biology_paper(self):
        """Test traditional biology paper (should be filtered out)."""
        # Traditional biology paper example (RRTF1-like)
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
        
        score = self.scorer.score_paper(traditional_paper)
        
        # Traditional biology paper should be filtered out
        assert score.quality_level == QualityLevel.FILTERED_OUT, f"Traditional biology should be filtered out, got {score.quality_level}"
        assert score.total_score == 0, f"Traditional biology should score 0, got {score.total_score}"
        
        print(f"✅ Traditional biology paper correctly filtered out - {score.quality_level.value}")
    
    def test_enformer_paper(self):
        """Test Enformer paper (should score 80-94 and be High Quality)."""
        # Enformer paper example
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
        
        score = self.scorer.score_paper(enformer_paper)
        
        # Enformer should be High Quality or Gold Standard (70+)
        assert score.total_score >= 70, f"Enformer should score 70+, got {score.total_score}"
        assert score.quality_level in [QualityLevel.HIGH_QUALITY, QualityLevel.GOLD_STANDARD], f"Enformer should be High Quality or Gold Standard, got {score.quality_level}"
        assert score.ai_ml_detected == True, "Enformer should have AI/ML detected"
        
        print(f"✅ Enformer scored {score.total_score}/100 - {score.quality_level.value}")
    
    def test_medium_quality_paper(self):
        """Test medium quality paper (should score 60-79)."""
        # Medium quality paper example
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
        
        score = self.scorer.score_paper(medium_paper)
        
        # Should be Low Quality or higher (30+)
        assert score.total_score >= 30, f"Medium quality should score 30+, got {score.total_score}"
        assert score.quality_level in [QualityLevel.LOW_QUALITY, QualityLevel.MEDIUM_QUALITY, QualityLevel.HIGH_QUALITY, QualityLevel.GOLD_STANDARD], f"Should be Low Quality or higher, got {score.quality_level}"
        
        print(f"✅ Medium quality paper scored {score.total_score}/100 - {score.quality_level.value}")
    
    def test_no_ai_ml_paper(self):
        """Test paper without AI/ML content (should be filtered out)."""
        # Paper without AI/ML content
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
        
        score = self.scorer.score_paper(no_ai_paper)
        
        # Should be filtered out
        assert score.quality_level == QualityLevel.FILTERED_OUT, f"No AI/ML should be filtered out, got {score.quality_level}"
        assert score.ai_ml_detected == False, "Should not detect AI/ML"
        assert score.total_score == 0, f"Should score 0, got {score.total_score}"
        
        print(f"✅ No AI/ML paper correctly filtered out - {score.quality_level.value}")
    
    def test_component_scoring(self):
        """Test individual component scoring."""
        # Test paper with specific components
        test_paper = {
            "title": "Novel transformer architecture for genomics with interpretability",
            "abstract": """
            We propose a novel transformer architecture for genomic sequence analysis. 
            The model uses self-attention mechanisms and achieves state-of-the-art performance 
            on ENCODE and TCGA datasets. Cross-validation shows statistical significance (p < 0.001). 
            We provide ablation studies and attention visualizations for interpretability. 
            Code is available on GitHub with detailed methods and data sharing protocols.
            """,
            "keywords": ["transformer", "novel architecture", "genomics", "interpretability", "encode", "tcga"],
            "mesh_terms": ["Transformer", "Genomics", "Interpretability"]
        }
        
        score = self.scorer.score_paper(test_paper)
        
        # Check component scores
        assert score.component_scores['methodological_innovation'] > 0, "Should have innovation score"
        assert score.component_scores['benchmark_usage'] > 0, "Should have benchmark score"
        assert score.component_scores['validation_rigor'] > 0, "Should have validation score"
        assert score.component_scores['reproducibility'] > 0, "Should have reproducibility score"
        
        print(f"✅ Component scoring test passed - Total: {score.total_score}/100")
    
    def test_quality_summary(self):
        """Test quality summary generation."""
        test_paper = {
            "title": "Test paper for summary",
            "abstract": "This is a test paper with transformer and deep learning for genomics analysis using ENCODE data with cross-validation.",
            "keywords": ["transformer", "deep learning", "genomics", "encode"],
            "mesh_terms": ["Transformer", "Deep Learning", "Genomics"]
        }
        
        score = self.scorer.score_paper(test_paper)
        summary = self.scorer.get_quality_summary(score)
        
        # Check summary contains expected elements
        assert "Quality Score:" in summary, "Summary should contain quality score"
        assert "Quality Level:" in summary, "Summary should contain quality level"
        assert "Component Scores:" in summary, "Summary should contain component scores"
        assert "AI/ML Keywords Found:" in summary, "Summary should contain keywords"
        
        print("✅ Quality summary test passed")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty paper
        empty_paper = {
            "title": "",
            "abstract": "",
            "keywords": [],
            "mesh_terms": []
        }
        
        score = self.scorer.score_paper(empty_paper)
        assert score.quality_level == QualityLevel.FILTERED_OUT, "Empty paper should be filtered out"
        assert score.total_score == 0, "Empty paper should score 0"
        
        # Paper with only traditional biology
        traditional_only = {
            "title": "PCR analysis of gene expression",
            "abstract": "We used PCR and Western blot to analyze gene expression in cell culture.",
            "keywords": ["pcr", "western blot"],
            "mesh_terms": ["PCR", "Western Blotting"]
        }
        
        score = self.scorer.score_paper(traditional_only)
        assert score.quality_level == QualityLevel.FILTERED_OUT, "Traditional only should be filtered out"
        
        print("✅ Edge cases test passed")
    
    def test_keyword_detection(self):
        """Test keyword detection functionality."""
        # Test a robust AI/ML paper that should definitely be detected
        test_paper = {
            "title": "Deep learning and transformer models for genomics analysis",
            "abstract": "This paper uses deep learning, transformer, neural networks, and machine learning for genomic sequence analysis. We apply BERT and attention mechanisms to DNA data.",
            "keywords": ["deep learning", "transformer", "machine learning", "genomics"],
            "mesh_terms": ["Deep Learning", "Neural Networks", "Genomics"]
        }
        
        score = self.scorer.score_paper(test_paper)
        assert score.ai_ml_detected == True, "Should detect AI/ML in comprehensive test"
        assert score.total_score > 0, "Should have positive score for AI/ML paper"
        
        # Test specific keywords are found
        expected_keywords = ["deep learning", "transformer", "machine learning"]
        for keyword in expected_keywords:
            assert keyword in score.keywords_found, f"Should find keyword: {keyword}"
        
        print("✅ Keyword detection test passed")


class TestQualityLevel:
    """Test cases for QualityLevel enum."""
    
    def test_quality_level_values(self):
        """Test QualityLevel enum values."""
        assert QualityLevel.FILTERED_OUT.value == "filtered_out"
        assert QualityLevel.LOW_QUALITY.value == "low_quality"
        assert QualityLevel.MEDIUM_QUALITY.value == "medium_quality"
        assert QualityLevel.HIGH_QUALITY.value == "high_quality"
        assert QualityLevel.GOLD_STANDARD.value == "gold_standard"
    
    def test_quality_level_comparison(self):
        """Test QualityLevel comparisons."""
        assert QualityLevel.GOLD_STANDARD != QualityLevel.HIGH_QUALITY
        assert QualityLevel.HIGH_QUALITY != QualityLevel.MEDIUM_QUALITY
        assert QualityLevel.MEDIUM_QUALITY != QualityLevel.FILTERED_OUT


class TestQualityScore:
    """Test cases for QualityScore dataclass."""
    
    def test_quality_score_creation(self):
        """Test QualityScore creation."""
        score = QualityScore(
            total_score=85,
            quality_level=QualityLevel.HIGH_QUALITY,
            ai_ml_detected=True,
            component_scores={"ai_ml_filter": 100, "methodological_innovation": 20},
            reasoning=["Test reason"],
            keywords_found=["transformer"],
            benchmarks_used=["ENCODE"],
            validation_methods=["cross-validation"]
        )
        
        assert score.total_score == 85
        assert score.quality_level == QualityLevel.HIGH_QUALITY
        assert score.ai_ml_detected == True
        assert len(score.component_scores) == 2
        assert len(score.reasoning) == 1
        assert len(score.keywords_found) == 1
        assert len(score.benchmarks_used) == 1
        assert len(score.validation_methods) == 1


if __name__ == "__main__":
    pytest.main([__file__]) 