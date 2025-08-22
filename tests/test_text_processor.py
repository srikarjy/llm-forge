"""
Tests for scientific text processor module.
"""

import pytest
from datasets import Dataset

from src.data.text_processor import (
    ScientificTextProcessor,
    PreprocessingConfig
)
from src.data.scientific_dataset import ScientificPaper, QualityTier, ComponentScores


class TestPreprocessingConfig:
    """Test PreprocessingConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        
        assert config.remove_citations is True
        assert config.remove_urls is True
        assert config.normalize_gene_names is True
        assert config.max_chunk_length == 2048
        assert config.chunk_overlap == 100
        assert config.causal_lm_format is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PreprocessingConfig(
            remove_citations=False,
            max_chunk_length=1024,
            instruction_format=True
        )
        
        assert config.remove_citations is False
        assert config.max_chunk_length == 1024
        assert config.instruction_format is True


class TestScientificTextProcessor:
    """Test ScientificTextProcessor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create text processor for testing."""
        return ScientificTextProcessor()
    
    @pytest.fixture
    def sample_paper(self):
        """Create sample scientific paper for testing."""
        return ScientificPaper(
            pmid="12345678",
            title="DNABERT: pre-trained Bidirectional Encoder Representations from Transformers for DNA sequence analysis",
            authors=["Zhang, L", "Wang, J", "Chen, X", "Li, Y"],
            journal="Nature Methods",
            publication_date="2023-06-15",
            doi="10.1038/s41592-023-01851-8",
            score=122.0,
            tier=QualityTier.GOLD_STANDARD,
            ai_ml_detected=True,
            component_scores=ComponentScores(
                methodological_innovation=35,
                benchmark_usage=30,
                validation_rigor=25,
                reproducibility=12,
                synergy_bonus=20
            ),
            reasoning=["Transformer Genomics: 20 pts", "Foundation Model: 15 pts"],
            keywords_found=["transformer", "deep learning", "neural network", "BERT"],
            benchmarks_used=["ENCODE", "1000 Genomes", "TCGA"],
            validation_methods=["cross-validation", "state-of-the-art"],
            scored_at="2025-08-21T23:04:14.712198",
            abstract="This paper presents DNABERT, a pre-trained bidirectional encoder for DNA sequence analysis using transformer architecture."
        )
    
    def test_init(self):
        """Test processor initialization."""
        processor = ScientificTextProcessor()
        
        assert processor.config is not None
        assert len(processor.citation_patterns) > 0
        assert processor.doi_pattern is not None
        assert len(processor.genomics_terms) > 0
    
    def test_init_with_config(self):
        """Test processor initialization with custom config."""
        config = PreprocessingConfig(remove_citations=False)
        processor = ScientificTextProcessor(config)
        
        assert processor.config.remove_citations is False
    
    def test_remove_citations_numbered(self, processor):
        """Test removing numbered citations."""
        text = "This is a sentence [1]. Another sentence [2,3]. Final sentence [1-5]."
        result = processor.remove_citations(text)
        
        assert "[1]" not in result
        assert "[2,3]" not in result
        assert "[1-5]" not in result
        assert "This is a sentence" in result
    
    def test_remove_citations_author_year(self, processor):
        """Test removing author-year citations."""
        text = "Previous work (Smith et al., 2023) showed that (Jones & Brown, 2022) results."
        result = processor.remove_citations(text)
        
        assert "(Smith et al., 2023)" not in result
        assert "(Jones & Brown, 2022)" not in result
        assert "Previous work" in result
        assert "showed that" in result
    
    def test_remove_citations_multiple(self, processor):
        """Test removing multiple citations."""
        text = "Research shows (Smith, 2023; Jones, 2024; Brown et al., 2022) significant results."
        result = processor.remove_citations(text)
        
        assert "(Smith, 2023; Jones, 2024; Brown et al., 2022)" not in result
        assert "Research shows" in result
        assert "significant results" in result
    
    def test_remove_urls_and_dois(self, processor):
        """Test removing URLs and DOIs."""
        text = "Visit https://example.com for more info. DOI: 10.1038/s41592-023-01851-8 available."
        result = processor.remove_urls_and_dois(text)
        
        assert "https://example.com" not in result
        assert "10.1038/s41592-023-01851-8" not in result
        assert "Visit" in result
        assert "for more info" in result
    
    def test_normalize_statistical_notation(self, processor):
        """Test normalizing statistical notation."""
        text = "Results show P < 0.05 and p-value = 0.001 with 95% CI."
        result = processor.normalize_statistical_notation(text)
        
        assert "p<0.05" in result or "p < 0.05" in result
        assert "95% CI" in result
    
    def test_normalize_whitespace(self, processor):
        """Test normalizing whitespace."""
        text = "This  has   multiple    spaces.\n\n\nAnd multiple\n\n\nline breaks."
        result = processor.normalize_whitespace(text)
        
        assert "  " not in result  # No double spaces
        assert "\n\n\n" not in result  # No triple line breaks
        assert "This has multiple spaces." in result
    
    def test_preprocess_scientific_text_full(self, processor):
        """Test full preprocessing pipeline."""
        text = """
        This paper [1] describes ENCODE data analysis (Smith et al., 2023).
        Results show P < 0.05   with   multiple   spaces.
        Visit https://example.com for details.
        """
        
        result = processor.preprocess_scientific_text(text)
        
        assert "[1]" not in result
        assert "(Smith et al., 2023)" not in result
        assert "https://example.com" not in result
        assert "ENCODE" in result  # Should preserve genomics terms
        assert "   " not in result  # Should normalize whitespace
    
    def test_preprocess_empty_text(self, processor):
        """Test preprocessing empty or invalid text."""
        assert processor.preprocess_scientific_text("") == ""
        assert processor.preprocess_scientific_text(None) == ""
        assert processor.preprocess_scientific_text("   ") == ""
    
    def test_chunk_text_simple(self, processor):
        """Test simple text chunking."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = processor.chunk_text(text, max_length=10)  # Small chunks for testing
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks
    
    def test_chunk_text_no_chunking_needed(self, processor):
        """Test chunking when text is already short enough."""
        text = "Short text that doesn't need chunking."
        chunks = processor.chunk_text(text, max_length=100)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_empty(self, processor):
        """Test chunking empty text."""
        chunks = processor.chunk_text("")
        assert chunks == []
    
    def test_split_sentences(self, processor):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = processor._split_sentences(text)
        
        assert len(sentences) >= 3  # Should split into multiple sentences
        assert all(isinstance(s, str) for s in sentences)
    
    def test_get_overlap_text(self, processor):
        """Test getting overlap text."""
        text = "word1 word2 word3 word4 word5"
        overlap = processor._get_overlap_text(text, 2)
        
        assert overlap == "word4 word5"
    
    def test_get_overlap_text_short(self, processor):
        """Test getting overlap text when text is shorter than overlap."""
        text = "word1 word2"
        overlap = processor._get_overlap_text(text, 5)
        
        assert overlap == text
    
    def test_create_causal_lm_dataset(self, processor, sample_paper):
        """Test creating causal language modeling dataset."""
        papers = [sample_paper]
        dataset = processor.create_causal_lm_dataset(papers)
        
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= 1
        assert "text" in dataset.column_names
        assert "pmid" in dataset.column_names
        assert "title" in dataset.column_names
        
        # Check first record
        first_record = dataset[0]
        assert first_record["pmid"] == sample_paper.pmid
        assert first_record["title"] == sample_paper.title
        assert isinstance(first_record["text"], str)
        assert len(first_record["text"]) > 0
    
    def test_create_instruction_dataset(self, processor, sample_paper):
        """Test creating instruction-following dataset."""
        papers = [sample_paper]
        dataset = processor.create_instruction_dataset(papers)
        
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= 1
        assert "instruction" in dataset.column_names
        assert "input" in dataset.column_names
        assert "output" in dataset.column_names
        
        # Check first record
        first_record = dataset[0]
        assert isinstance(first_record["instruction"], str)
        assert isinstance(first_record["input"], str)
        assert isinstance(first_record["output"], str)
        assert len(first_record["instruction"]) > 0
    
    def test_create_instruction_dataset_custom_types(self, processor, sample_paper):
        """Test creating instruction dataset with custom instruction types."""
        papers = [sample_paper]
        instruction_types = ["summarize", "extract_benchmarks"]
        dataset = processor.create_instruction_dataset(papers, instruction_types)
        
        assert isinstance(dataset, Dataset)
        assert len(dataset) == len(papers) * len(instruction_types)
    
    def test_create_qa_dataset(self, processor, sample_paper):
        """Test creating question-answering dataset."""
        papers = [sample_paper]
        dataset = processor.create_qa_dataset(papers)
        
        assert isinstance(dataset, Dataset)
        assert len(dataset) >= 1
        assert "question" in dataset.column_names
        assert "context" in dataset.column_names
        assert "answer" in dataset.column_names
        
        # Check first record
        first_record = dataset[0]
        assert isinstance(first_record["question"], str)
        assert isinstance(first_record["context"], str)
        assert isinstance(first_record["answer"], str)
        assert len(first_record["question"]) > 0
    
    def test_create_instruction_record_summarize(self, processor, sample_paper):
        """Test creating summarize instruction record."""
        processed_text = "Sample processed text"
        record = processor._create_instruction_record(sample_paper, processed_text, "summarize")
        
        assert record is not None
        assert "instruction" in record
        assert "Summarize" in record["instruction"]
        assert record["pmid"] == sample_paper.pmid
    
    def test_create_instruction_record_extract_benchmarks(self, processor, sample_paper):
        """Test creating extract benchmarks instruction record."""
        processed_text = "Sample processed text"
        record = processor._create_instruction_record(sample_paper, processed_text, "extract_benchmarks")
        
        assert record is not None
        assert "benchmarks" in record["instruction"].lower()
        assert "ENCODE" in record["output"]  # Should include paper's benchmarks
    
    def test_create_instruction_record_invalid_type(self, processor, sample_paper):
        """Test creating instruction record with invalid type."""
        processed_text = "Sample processed text"
        record = processor._create_instruction_record(sample_paper, processed_text, "invalid_type")
        
        assert record is None
    
    def test_get_answer_from_paper(self, processor, sample_paper):
        """Test getting answers from paper."""
        assert processor._get_answer_from_paper(sample_paper, "title") == sample_paper.title
        assert processor._get_answer_from_paper(sample_paper, "journal") == sample_paper.journal
        
        benchmarks_answer = processor._get_answer_from_paper(sample_paper, "benchmarks_used")
        assert "ENCODE" in benchmarks_answer
        
        keywords_answer = processor._get_answer_from_paper(sample_paper, "keywords_found")
        assert "transformer" in keywords_answer
        
        invalid_answer = processor._get_answer_from_paper(sample_paper, "invalid_key")
        assert invalid_answer == "Not available"
    
    def test_process_papers_for_training_causal_lm(self, processor, sample_paper):
        """Test processing papers for causal LM training."""
        papers = [sample_paper]
        dataset = processor.process_papers_for_training(papers, "causal_lm")
        
        assert isinstance(dataset, Dataset)
        assert "text" in dataset.column_names
    
    def test_process_papers_for_training_instruction(self, processor, sample_paper):
        """Test processing papers for instruction training."""
        papers = [sample_paper]
        dataset = processor.process_papers_for_training(papers, "instruction")
        
        assert isinstance(dataset, Dataset)
        assert "instruction" in dataset.column_names
    
    def test_process_papers_for_training_qa(self, processor, sample_paper):
        """Test processing papers for QA training."""
        papers = [sample_paper]
        dataset = processor.process_papers_for_training(papers, "qa")
        
        assert isinstance(dataset, Dataset)
        assert "question" in dataset.column_names
    
    def test_process_papers_for_training_invalid_format(self, processor, sample_paper):
        """Test processing papers with invalid format."""
        papers = [sample_paper]
        
        with pytest.raises(ValueError, match="Unsupported format type"):
            processor.process_papers_for_training(papers, "invalid_format")
    
    def test_genomics_terms_preservation(self, processor):
        """Test that genomics terms are preserved during processing."""
        text = "This study uses ENCODE and TCGA datasets for GWAS analysis."
        result = processor.preprocess_scientific_text(text)
        
        assert "ENCODE" in result
        assert "TCGA" in result
        assert "GWAS" in result
    
    def test_multiple_papers_processing(self, processor, sample_paper):
        """Test processing multiple papers."""
        # Create second paper
        paper2 = ScientificPaper(
            pmid="87654321",
            title="Another genomics paper",
            authors=["Author, B"],
            journal="Nature",
            publication_date="2023-07-01",
            doi="10.1038/test",
            score=95.0,
            tier=QualityTier.HIGH_QUALITY,
            ai_ml_detected=True,
            component_scores=ComponentScores(),
            reasoning=["Test reasoning"],
            keywords_found=["genomics", "analysis"],
            benchmarks_used=["GTEx"],
            validation_methods=["validation"],
            scored_at="2025-08-21T23:04:14.712198"
        )
        
        papers = [sample_paper, paper2]
        dataset = processor.create_causal_lm_dataset(papers)
        
        assert len(dataset) >= 2
        pmids = [record["pmid"] for record in dataset]
        assert sample_paper.pmid in pmids
        assert paper2.pmid in pmids


class TestTextProcessorIntegration:
    """Integration tests for text processor."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end text processing pipeline."""
        # Create processor with custom config
        config = PreprocessingConfig(
            remove_citations=True,
            normalize_whitespace=True,
            max_chunk_length=100
        )
        processor = ScientificTextProcessor(config)
        
        # Create sample paper
        paper = ScientificPaper(
            pmid="test123",
            title="Test Paper [1]",
            authors=["Test Author"],
            journal="Test Journal",
            publication_date="2023-01-01",
            doi="10.1000/test",
            score=100.0,
            tier=QualityTier.GOLD_STANDARD,
            ai_ml_detected=True,
            component_scores=ComponentScores(),
            reasoning=["Test reasoning"],
            keywords_found=["test", "genomics"],
            benchmarks_used=["ENCODE"],
            validation_methods=["test"],
            scored_at="2025-08-21T23:04:14.712198",
            abstract="This is a test abstract with citations [1,2] and URLs https://example.com."
        )
        
        # Process for different formats
        causal_dataset = processor.process_papers_for_training([paper], "causal_lm")
        instruction_dataset = processor.process_papers_for_training([paper], "instruction")
        qa_dataset = processor.process_papers_for_training([paper], "qa")
        
        # Verify all datasets were created
        assert len(causal_dataset) >= 1
        assert len(instruction_dataset) >= 1
        assert len(qa_dataset) >= 1
        
        # Verify text processing worked
        causal_text = causal_dataset[0]["text"]
        assert "[1,2]" not in causal_text  # Citations removed
        assert "https://example.com" not in causal_text  # URLs removed
        assert "ENCODE" in causal_text  # Genomics terms preserved


if __name__ == "__main__":
    pytest.main([__file__])