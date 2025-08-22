"""
Tests for scientific dataset module.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from src.data.scientific_dataset import (
    ScientificPaper,
    ScientificDataModule,
    QualityTier,
    ComponentScores
)


class TestQualityTier:
    """Test QualityTier enum functionality."""
    
    def test_from_string_valid(self):
        """Test converting valid strings to QualityTier."""
        assert QualityTier.from_string("gold_standard") == QualityTier.GOLD_STANDARD
        assert QualityTier.from_string("high_quality") == QualityTier.HIGH_QUALITY
        assert QualityTier.from_string("GOLD_STANDARD") == QualityTier.GOLD_STANDARD
    
    def test_from_string_invalid(self):
        """Test converting invalid strings defaults to LOW_QUALITY."""
        assert QualityTier.from_string("invalid_tier") == QualityTier.LOW_QUALITY
        assert QualityTier.from_string("") == QualityTier.LOW_QUALITY


class TestComponentScores:
    """Test ComponentScores functionality."""
    
    def test_from_dict_complete(self):
        """Test creating ComponentScores from complete dictionary."""
        scores_dict = {
            "methodological_innovation": 35,
            "benchmark_usage": 30,
            "validation_rigor": 25,
            "reproducibility": 12,
            "synergy_bonus": 20
        }
        
        scores = ComponentScores.from_dict(scores_dict)
        
        assert scores.methodological_innovation == 35
        assert scores.benchmark_usage == 30
        assert scores.validation_rigor == 25
        assert scores.reproducibility == 12
        assert scores.synergy_bonus == 20
    
    def test_from_dict_partial(self):
        """Test creating ComponentScores from partial dictionary."""
        scores_dict = {
            "methodological_innovation": 35,
            "benchmark_usage": 30
        }
        
        scores = ComponentScores.from_dict(scores_dict)
        
        assert scores.methodological_innovation == 35
        assert scores.benchmark_usage == 30
        assert scores.validation_rigor == 0  # Default value
        assert scores.reproducibility == 0  # Default value
        assert scores.synergy_bonus == 0  # Default value


class TestScientificPaper:
    """Test ScientificPaper functionality."""
    
    @pytest.fixture
    def sample_paper_dict(self):
        """Sample paper dictionary for testing."""
        return {
            "pmid": "12345678",
            "title": "DNABERT: pre-trained Bidirectional Encoder Representations",
            "authors": ["Zhang, L", "Wang, J"],
            "journal": "Nature Methods",
            "publication_date": "2023-06-15",
            "doi": "10.1038/s41592-023-01851-8",
            "score": 122,
            "tier": "gold_standard",
            "ai_ml_detected": True,
            "component_scores": {
                "methodological_innovation": 35,
                "benchmark_usage": 30,
                "validation_rigor": 25,
                "reproducibility": 12,
                "synergy_bonus": 20
            },
            "reasoning": [
                "Transformer Genomics: 20 pts",
                "Foundation Model: 15 pts"
            ],
            "keywords_found": ["transformer", "deep learning"],
            "benchmarks_used": ["ENCODE", "1000 Genomes"],
            "validation_methods": ["cross-validation"],
            "scored_at": "2025-08-21T23:04:14.712198",
            "abstract": "This paper presents DNABERT..."
        }
    
    def test_from_dict(self, sample_paper_dict):
        """Test creating ScientificPaper from dictionary."""
        paper = ScientificPaper.from_dict(sample_paper_dict)
        
        assert paper.pmid == "12345678"
        assert paper.title == "DNABERT: pre-trained Bidirectional Encoder Representations"
        assert paper.authors == ["Zhang, L", "Wang, J"]
        assert paper.journal == "Nature Methods"
        assert paper.score == 122
        assert paper.tier == QualityTier.GOLD_STANDARD
        assert paper.ai_ml_detected is True
        assert paper.benchmarks_used == ["ENCODE", "1000 Genomes"]
        assert paper.abstract == "This paper presents DNABERT..."
    
    def test_to_training_text_with_metadata(self, sample_paper_dict):
        """Test converting paper to training text with metadata."""
        paper = ScientificPaper.from_dict(sample_paper_dict)
        training_text = paper.to_training_text(include_metadata=True)
        
        assert "Title: DNABERT" in training_text
        assert "Authors: Zhang, L, Wang, J" in training_text
        assert "Journal: Nature Methods" in training_text
        assert "Benchmarks: ENCODE, 1000 Genomes" in training_text
        assert "Abstract: This paper presents DNABERT..." in training_text
    
    def test_to_training_text_without_metadata(self, sample_paper_dict):
        """Test converting paper to training text without metadata."""
        paper = ScientificPaper.from_dict(sample_paper_dict)
        training_text = paper.to_training_text(include_metadata=False)
        
        assert "Title:" not in training_text
        assert "Authors:" not in training_text
        assert "Abstract: This paper presents DNABERT..." in training_text
    
    def test_extract_benchmarks_from_reasoning(self, sample_paper_dict):
        """Test extracting benchmarks from reasoning text."""
        sample_paper_dict["reasoning"] = [
            "Uses ENCODE dataset for validation",
            "Compared against TCGA and GTEx benchmarks",
            "1000 Genomes data used for training"
        ]
        
        paper = ScientificPaper.from_dict(sample_paper_dict)
        extracted_benchmarks = paper.extract_benchmarks_from_reasoning()
        
        assert "ENCODE" in extracted_benchmarks
        assert "TCGA" in extracted_benchmarks
        assert "GTEX" in extracted_benchmarks
        assert "1000 GENOMES" in extracted_benchmarks


class TestScientificDataModule:
    """Test ScientificDataModule functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        return {
            "metadata": {
                "generated_at": "2025-08-21T23:04:14.713169",
                "total_papers_processed": 3,
                "high_quality_papers_count": 3,
                "quality_threshold": 70
            },
            "papers": [
                {
                    "pmid": "12345678",
                    "title": "DNABERT: pre-trained Bidirectional Encoder Representations",
                    "authors": ["Zhang, L", "Wang, J"],
                    "journal": "Nature Methods",
                    "publication_date": "2023-06-15",
                    "doi": "10.1038/s41592-023-01851-8",
                    "score": 122,
                    "tier": "gold_standard",
                    "ai_ml_detected": True,
                    "component_scores": {
                        "methodological_innovation": 35,
                        "benchmark_usage": 30
                    },
                    "reasoning": ["Transformer Genomics: 20 pts"],
                    "keywords_found": ["transformer", "deep learning"],
                    "benchmarks_used": ["ENCODE", "1000 Genomes"],
                    "validation_methods": ["cross-validation"],
                    "scored_at": "2025-08-21T23:04:14.712198"
                },
                {
                    "pmid": "12345679",
                    "title": "Enformer: Predicting gene expression",
                    "authors": ["Avsec, Z"],
                    "journal": "Nature",
                    "publication_date": "2023-09-20",
                    "doi": "10.1038/s41586-021-04020-1",
                    "score": 85,
                    "tier": "high_quality",
                    "ai_ml_detected": True,
                    "component_scores": {
                        "methodological_innovation": 25,
                        "benchmark_usage": 20
                    },
                    "reasoning": ["Novel Architecture: 8 pts"],
                    "keywords_found": ["attention"],
                    "benchmarks_used": ["ENCODE"],
                    "validation_methods": ["test set"],
                    "scored_at": "2025-08-21T23:04:14.712351"
                },
                {
                    "pmid": "12345680",
                    "title": "Low quality paper",
                    "authors": ["Smith, J"],
                    "journal": "Low Impact Journal",
                    "publication_date": "2023-01-01",
                    "doi": "10.1000/low-impact",
                    "score": 45,
                    "tier": "low_quality",
                    "ai_ml_detected": False,
                    "component_scores": {},
                    "reasoning": [],
                    "keywords_found": [],
                    "benchmarks_used": [],
                    "validation_methods": [],
                    "scored_at": "2025-08-21T23:04:14.712500"
                }
            ]
        }
    
    @pytest.fixture
    def temp_data_file(self, sample_data):
        """Create temporary data file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)
    
    def test_init(self, temp_data_file):
        """Test ScientificDataModule initialization."""
        module = ScientificDataModule(temp_data_file)
        
        assert module.data_file == Path(temp_data_file)
        assert module.papers == []
        assert module.metadata == {}
        assert module._loaded is False
    
    def test_load_papers_all(self, temp_data_file):
        """Test loading all papers."""
        module = ScientificDataModule(temp_data_file)
        papers = module.load_papers(min_quality_score=0)
        
        assert len(papers) == 3
        assert module._loaded is True
        assert len(module.metadata) > 0
    
    def test_load_papers_filtered_by_score(self, temp_data_file):
        """Test loading papers filtered by quality score."""
        module = ScientificDataModule(temp_data_file)
        papers = module.load_papers(min_quality_score=80)
        
        assert len(papers) == 2  # Only papers with score >= 80
        assert all(p.score >= 80 for p in papers)
    
    def test_load_papers_filtered_by_tier(self, temp_data_file):
        """Test loading papers filtered by quality tier."""
        module = ScientificDataModule(temp_data_file)
        papers = module.load_papers(
            min_quality_score=0,
            quality_tiers=[QualityTier.GOLD_STANDARD]
        )
        
        assert len(papers) == 1
        assert papers[0].tier == QualityTier.GOLD_STANDARD
    
    def test_validate_paper_format_valid(self):
        """Test validating valid paper format."""
        module = ScientificDataModule("dummy.json")
        
        valid_paper = {
            "pmid": "12345",
            "title": "Test Paper",
            "authors": ["Author, A"],
            "journal": "Test Journal",
            "publication_date": "2023-01-01",
            "doi": "10.1000/test",
            "score": 100,
            "tier": "gold_standard"
        }
        
        is_valid, errors = module.validate_paper_format(valid_paper)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_paper_format_missing_fields(self):
        """Test validating paper with missing required fields."""
        module = ScientificDataModule("dummy.json")
        
        invalid_paper = {
            "pmid": "12345",
            "title": "Test Paper"
            # Missing required fields
        }
        
        is_valid, errors = module.validate_paper_format(invalid_paper)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("Missing required field" in error for error in errors)
    
    def test_validate_paper_format_wrong_types(self):
        """Test validating paper with wrong field types."""
        module = ScientificDataModule("dummy.json")
        
        invalid_paper = {
            "pmid": 12345,  # Should be string
            "title": "Test Paper",
            "authors": "Author, A",  # Should be list
            "journal": "Test Journal",
            "publication_date": "2023-01-01",
            "doi": "10.1000/test",
            "score": "invalid",  # Should be number
            "tier": "gold_standard"
        }
        
        is_valid, errors = module.validate_paper_format(invalid_paper)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("must be a string" in error for error in errors)
        assert any("must be a list" in error for error in errors)
        assert any("must be a number" in error for error in errors)
    
    def test_get_statistics(self, temp_data_file):
        """Test getting dataset statistics."""
        module = ScientificDataModule(temp_data_file)
        module.load_papers(min_quality_score=0)
        
        stats = module.get_statistics()
        
        assert stats["total_papers"] == 3
        assert "metadata" in stats
        assert "quality_tiers" in stats
        assert "score_statistics" in stats
        assert "benchmark_usage" in stats
        assert "journal_distribution" in stats
        
        # Check quality tier distribution
        assert stats["quality_tiers"]["gold_standard"] == 1
        assert stats["quality_tiers"]["high_quality"] == 1
        assert stats["quality_tiers"]["low_quality"] == 1
        
        # Check benchmark usage
        assert "ENCODE" in stats["benchmark_usage"]
        assert stats["benchmark_usage"]["ENCODE"] == 2  # Used in 2 papers
    
    def test_filter_by_benchmarks(self, temp_data_file):
        """Test filtering papers by benchmarks."""
        module = ScientificDataModule(temp_data_file)
        module.load_papers(min_quality_score=0)
        
        # Filter by ENCODE benchmark
        encode_papers = module.filter_by_benchmarks(["ENCODE"])
        assert len(encode_papers) == 2
        
        # Filter by non-existent benchmark
        nonexistent_papers = module.filter_by_benchmarks(["NonExistent"])
        assert len(nonexistent_papers) == 0
        
        # Filter by multiple benchmarks
        multiple_papers = module.filter_by_benchmarks(["ENCODE", "1000 Genomes"])
        assert len(multiple_papers) == 2
    
    def test_export_training_data_jsonl(self, temp_data_file):
        """Test exporting training data in JSONL format."""
        module = ScientificDataModule(temp_data_file)
        papers = module.load_papers(min_quality_score=80)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            output_file = f.name
        
        try:
            module.export_training_data(output_file, papers, format_type="jsonl")
            
            # Verify the output
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) == 2  # 2 papers with score >= 80
            
            # Check first line is valid JSON
            first_record = json.loads(lines[0])
            assert "text" in first_record
            assert "pmid" in first_record
            assert "title" in first_record
            assert "score" in first_record
            
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def test_export_training_data_json(self, temp_data_file):
        """Test exporting training data in JSON format."""
        module = ScientificDataModule(temp_data_file)
        papers = module.load_papers(min_quality_score=80)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            module.export_training_data(output_file, papers, format_type="json")
            
            # Verify the output
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            assert len(data) == 2  # 2 papers with score >= 80
            assert "text" in data[0]
            assert "pmid" in data[0]
            
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def test_export_training_data_txt(self, temp_data_file):
        """Test exporting training data in TXT format."""
        module = ScientificDataModule(temp_data_file)
        papers = module.load_papers(min_quality_score=80)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_file = f.name
        
        try:
            module.export_training_data(output_file, papers, format_type="txt")
            
            # Verify the output
            with open(output_file, 'r') as f:
                content = f.read()
            
            assert "DNABERT" in content
            assert "Enformer" in content
            assert "=" * 80 in content  # Separator between papers
            
        finally:
            Path(output_file).unlink(missing_ok=True)
    
    def test_get_papers_by_tier(self, temp_data_file):
        """Test getting papers by quality tier."""
        module = ScientificDataModule(temp_data_file)
        module.load_papers(min_quality_score=0)
        
        gold_papers = module.get_papers_by_tier(QualityTier.GOLD_STANDARD)
        assert len(gold_papers) == 1
        assert gold_papers[0].tier == QualityTier.GOLD_STANDARD
        
        high_papers = module.get_papers_by_tier(QualityTier.HIGH_QUALITY)
        assert len(high_papers) == 1
        assert high_papers[0].tier == QualityTier.HIGH_QUALITY
    
    def test_get_papers_by_journal(self, temp_data_file):
        """Test getting papers by journal."""
        module = ScientificDataModule(temp_data_file)
        module.load_papers(min_quality_score=0)
        
        nature_papers = module.get_papers_by_journal("Nature")
        assert len(nature_papers) == 1
        assert nature_papers[0].journal == "Nature"
        
        # Test case insensitive
        nature_papers_lower = module.get_papers_by_journal("nature")
        assert len(nature_papers_lower) == 1
    
    def test_file_not_found(self):
        """Test handling of non-existent data file."""
        module = ScientificDataModule("nonexistent.json")
        
        with pytest.raises(FileNotFoundError):
            module.load_papers()
    
    def test_invalid_json(self):
        """Test handling of invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_file = f.name
        
        try:
            module = ScientificDataModule(temp_file)
            
            with pytest.raises(ValueError, match="Invalid JSON format"):
                module.load_papers()
        
        finally:
            Path(temp_file).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])