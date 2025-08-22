"""
Scientific text processor for genomics-specific preprocessing.

This module provides functionality for preprocessing scientific papers
with genomics-specific text cleaning, normalization, and formatting
for LLM fine-tuning.
"""

import re
import string
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datasets import Dataset

try:
    from ..utils.logger import get_logger
    from .scientific_dataset import ScientificPaper
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger
    from data.scientific_dataset import ScientificPaper

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    
    # Citation removal
    remove_citations: bool = True
    remove_urls: bool = True
    remove_doi: bool = True
    
    # Text normalization
    normalize_gene_names: bool = True
    normalize_statistical_notation: bool = True
    normalize_whitespace: bool = True
    
    # Text chunking
    max_chunk_length: int = 2048
    chunk_overlap: int = 100
    respect_sentence_boundaries: bool = True
    
    # Special handling
    preserve_genomics_terms: bool = True
    handle_special_tokens: bool = True
    
    # Dataset formatting
    instruction_format: bool = False
    causal_lm_format: bool = True
    qa_format: bool = False


class ScientificTextProcessor:
    """Text processor for genomics-specific preprocessing."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the text processor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        logger.info("Initialized ScientificTextProcessor")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for text processing."""
        
        # Citation patterns
        self.citation_patterns = [
            # Numbered citations: [1], [1-3], [1,2,3]
            re.compile(r'\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]'),
            # Author-year citations: (Smith et al., 2023), (Smith & Jones, 2023)
            re.compile(r'\([A-Za-z][^()]*(?:et al\.?|&)[^()]*\d{4}[^()]*\)'),
            # Simple author citations: (Smith, 2023)
            re.compile(r'\([A-Za-z][^(),]*,\s*\d{4}\)'),
            # Multiple citations: (Smith, 2023; Jones, 2024)
            re.compile(r'\([^()]*\d{4}[^()]*(?:;\s*[^()]*\d{4}[^()]*)*\)'),
        ]
        
        # DOI and URL patterns
        self.doi_pattern = re.compile(r'doi:\s*10\.\d+/[^\s]+', re.IGNORECASE)
        self.url_pattern = re.compile(r'https?://[^\s]+')
        
        # Gene name patterns (common genomics nomenclature)
        self.gene_patterns = [
            # Human genes: BRCA1, TP53, etc.
            re.compile(r'\b[A-Z]{2,}[0-9]+[A-Z]*\b'),
            # Mouse genes: Brca1, Tp53, etc.
            re.compile(r'\b[A-Z][a-z]+[0-9]+[a-z]*\b'),
            # Protein IDs: NP_000001.1, XP_123456.2
            re.compile(r'\b[NX]P_\d+\.\d+\b'),
        ]
        
        # Statistical notation patterns
        self.stat_patterns = [
            # P-values: p < 0.05, P = 0.001, p-value < 0.01
            re.compile(r'[Pp](?:\s*-?\s*value)?\s*[<>=≤≥]\s*0?\.\d+(?:[eE][-+]?\d+)?'),
            # Confidence intervals: 95% CI, CI 95%
            re.compile(r'(?:95%?\s*CI|CI\s*95%?)'),
            # Statistical significance: *, **, ***
            re.compile(r'\*{1,3}(?=\s|$)'),
        ]
        
        # Genomics-specific terms to preserve
        self.genomics_terms = {
            'ENCODE', 'TCGA', 'GTEx', '1000 Genomes', 'COSMIC', 'ClinVar',
            'dbSNP', 'GWAS', 'UniProt', 'PDB', 'NCBI', 'Ensembl',
            'SNP', 'CNV', 'indel', 'CRISPR', 'ChIP-seq', 'RNA-seq',
            'scRNA-seq', 'ATAC-seq', 'Hi-C', 'eQTL', 'GWAS'
        }
        
        # Special tokens for genomics
        self.special_tokens = {
            '<GENE>': 'gene name',
            '<PROTEIN>': 'protein identifier',
            '<PATHWAY>': 'biological pathway',
            '<DISEASE>': 'disease or phenotype',
            '<DRUG>': 'drug or compound',
            '<ORGANISM>': 'organism or species'
        }
    
    def preprocess_scientific_text(self, text: str) -> str:
        """Main preprocessing function for scientific text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        processed_text = text
        
        # Remove citations if configured
        if self.config.remove_citations:
            processed_text = self.remove_citations(processed_text)
        
        # Remove URLs and DOIs if configured
        if self.config.remove_urls:
            processed_text = self.remove_urls_and_dois(processed_text)
        
        # Normalize gene names if configured
        if self.config.normalize_gene_names:
            processed_text = self.normalize_gene_names(processed_text)
        
        # Normalize statistical notation if configured
        if self.config.normalize_statistical_notation:
            processed_text = self.normalize_statistical_notation(processed_text)
        
        # Handle special tokens if configured
        if self.config.handle_special_tokens:
            processed_text = self.handle_special_tokens(processed_text)
        
        # Normalize whitespace if configured
        if self.config.normalize_whitespace:
            processed_text = self.normalize_whitespace(processed_text)
        
        return processed_text.strip()
    
    def remove_citations(self, text: str) -> str:
        """Remove citations from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with citations removed
        """
        processed_text = text
        
        for pattern in self.citation_patterns:
            processed_text = pattern.sub('', processed_text)
        
        return processed_text
    
    def remove_urls_and_dois(self, text: str) -> str:
        """Remove URLs and DOIs from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with URLs and DOIs removed
        """
        processed_text = text
        
        if self.config.remove_doi:
            processed_text = self.doi_pattern.sub('', processed_text)
        
        if self.config.remove_urls:
            processed_text = self.url_pattern.sub('', processed_text)
        
        return processed_text
    
    def normalize_gene_names(self, text: str) -> str:
        """Normalize gene names and identifiers.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized gene names
        """
        processed_text = text
        
        # Preserve known genomics terms
        if self.config.preserve_genomics_terms:
            # Create temporary placeholders for genomics terms
            placeholders = {}
            for i, term in enumerate(self.genomics_terms):
                if term in processed_text:
                    placeholder = f"__GENOMICS_TERM_{i}__"
                    placeholders[placeholder] = term
                    processed_text = processed_text.replace(term, placeholder)
        
        # Apply gene name normalization patterns
        for pattern in self.gene_patterns:
            # This is a placeholder - in practice, you might want to
            # standardize gene names using a genomics database
            pass
        
        # Restore genomics terms
        if self.config.preserve_genomics_terms:
            for placeholder, term in placeholders.items():
                processed_text = processed_text.replace(placeholder, term)
        
        return processed_text
    
    def normalize_statistical_notation(self, text: str) -> str:
        """Normalize statistical notation.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized statistical notation
        """
        processed_text = text
        
        # Normalize p-values
        processed_text = re.sub(
            r'[Pp](?:\s*-?\s*value)?\s*[<>=≤≥]\s*0?\.\d+(?:[eE][-+]?\d+)?',
            lambda m: m.group(0).replace('P', 'p').replace(' ', ''),
            processed_text
        )
        
        # Normalize confidence intervals
        processed_text = re.sub(
            r'(?:95%?\s*CI|CI\s*95%?)',
            '95% CI',
            processed_text,
            flags=re.IGNORECASE
        )
        
        return processed_text
    
    def handle_special_tokens(self, text: str) -> str:
        """Handle special tokens for genomics entities.
        
        Args:
            text: Input text
            
        Returns:
            Text with special tokens handled
        """
        # This is a placeholder for more sophisticated entity recognition
        # In practice, you might use NER models or genomics-specific tools
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple whitespace with single space
        processed_text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        processed_text = processed_text.strip()
        
        # Normalize line breaks
        processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
        
        return processed_text
    
    def chunk_text(
        self, 
        text: str, 
        max_length: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """Chunk text into smaller pieces respecting sentence boundaries.
        
        Args:
            text: Input text to chunk
            max_length: Maximum chunk length (uses config default if None)
            overlap: Overlap between chunks (uses config default if None)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        max_length = max_length or self.config.max_chunk_length
        overlap = overlap or self.config.chunk_overlap
        
        # Simple sentence splitting (could be improved with NLTK/spaCy)
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed max_length, start new chunk
            if current_length + sentence_length > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and chunks:
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk.split())
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add final chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - could be improved with NLTK
        sentence_endings = re.compile(r'[.!?]+\s+')
        sentences = sentence_endings.split(text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_text(self, text: str, overlap_words: int) -> str:
        """Get overlap text from the end of a chunk.
        
        Args:
            text: Input text
            overlap_words: Number of words to overlap
            
        Returns:
            Overlap text
        """
        words = text.split()
        if len(words) <= overlap_words:
            return text
        
        return " ".join(words[-overlap_words:])
    
    def create_instruction_dataset(
        self, 
        papers: List[ScientificPaper],
        instruction_types: Optional[List[str]] = None
    ) -> Dataset:
        """Create instruction-following dataset from papers.
        
        Args:
            papers: List of scientific papers
            instruction_types: Types of instructions to generate
            
        Returns:
            HuggingFace Dataset object
        """
        if instruction_types is None:
            instruction_types = ["summarize", "extract_benchmarks", "identify_methods"]
        
        dataset_records = []
        
        for paper in papers:
            # Get paper text
            paper_text = paper.to_training_text(include_metadata=True)
            processed_text = self.preprocess_scientific_text(paper_text)
            
            # Generate different instruction types
            for instruction_type in instruction_types:
                record = self._create_instruction_record(paper, processed_text, instruction_type)
                if record:
                    dataset_records.append(record)
        
        logger.info(f"Created instruction dataset with {len(dataset_records)} records")
        return Dataset.from_list(dataset_records)
    
    def _create_instruction_record(
        self, 
        paper: ScientificPaper, 
        processed_text: str, 
        instruction_type: str
    ) -> Optional[Dict[str, str]]:
        """Create a single instruction record.
        
        Args:
            paper: Scientific paper
            processed_text: Preprocessed paper text
            instruction_type: Type of instruction
            
        Returns:
            Instruction record or None
        """
        if instruction_type == "summarize":
            instruction = "Summarize this genomics research paper:"
            response = f"This paper titled '{paper.title}' presents research on {', '.join(paper.keywords_found[:3])} using {', '.join(paper.benchmarks_used)} datasets."
            
        elif instruction_type == "extract_benchmarks":
            instruction = "What benchmarks or datasets were used in this genomics paper?"
            response = ", ".join(paper.benchmarks_used) if paper.benchmarks_used else "No specific benchmarks mentioned."
            
        elif instruction_type == "identify_methods":
            instruction = "What computational methods are described in this paper?"
            response = f"The paper describes methods related to {', '.join(paper.keywords_found[:5])}."
            
        else:
            return None
        
        return {
            "instruction": instruction,
            "input": processed_text[:1000],  # Truncate for instruction format
            "output": response,
            "pmid": paper.pmid,
            "title": paper.title
        }
    
    def create_causal_lm_dataset(self, papers: List[ScientificPaper]) -> Dataset:
        """Create causal language modeling dataset from papers.
        
        Args:
            papers: List of scientific papers
            
        Returns:
            HuggingFace Dataset object
        """
        dataset_records = []
        
        for paper in papers:
            # Get paper text
            paper_text = paper.to_training_text(include_metadata=True)
            processed_text = self.preprocess_scientific_text(paper_text)
            
            # Chunk text if necessary
            if len(processed_text.split()) > self.config.max_chunk_length:
                chunks = self.chunk_text(processed_text)
                for i, chunk in enumerate(chunks):
                    record = {
                        "text": chunk,
                        "pmid": paper.pmid,
                        "chunk_id": i,
                        "title": paper.title,
                        "score": paper.score,
                        "tier": paper.tier.value
                    }
                    dataset_records.append(record)
            else:
                record = {
                    "text": processed_text,
                    "pmid": paper.pmid,
                    "chunk_id": 0,
                    "title": paper.title,
                    "score": paper.score,
                    "tier": paper.tier.value
                }
                dataset_records.append(record)
        
        logger.info(f"Created causal LM dataset with {len(dataset_records)} records")
        return Dataset.from_list(dataset_records)
    
    def create_qa_dataset(self, papers: List[ScientificPaper]) -> Dataset:
        """Create question-answering dataset from papers.
        
        Args:
            papers: List of scientific papers
            
        Returns:
            HuggingFace Dataset object
        """
        dataset_records = []
        
        qa_templates = [
            {
                "question": "What is the title of this paper?",
                "answer_key": "title"
            },
            {
                "question": "Which journal published this research?",
                "answer_key": "journal"
            },
            {
                "question": "What benchmarks or datasets were used?",
                "answer_key": "benchmarks_used"
            },
            {
                "question": "What are the main keywords or topics?",
                "answer_key": "keywords_found"
            },
            {
                "question": "What validation methods were employed?",
                "answer_key": "validation_methods"
            }
        ]
        
        for paper in papers:
            paper_text = paper.to_training_text(include_metadata=False)
            processed_text = self.preprocess_scientific_text(paper_text)
            
            for template in qa_templates:
                answer = self._get_answer_from_paper(paper, template["answer_key"])
                if answer:
                    record = {
                        "question": template["question"],
                        "context": processed_text[:1500],  # Truncate context
                        "answer": answer,
                        "pmid": paper.pmid,
                        "title": paper.title
                    }
                    dataset_records.append(record)
        
        logger.info(f"Created QA dataset with {len(dataset_records)} records")
        return Dataset.from_list(dataset_records)
    
    def _get_answer_from_paper(self, paper: ScientificPaper, answer_key: str) -> str:
        """Get answer from paper based on answer key.
        
        Args:
            paper: Scientific paper
            answer_key: Key indicating what answer to extract
            
        Returns:
            Answer string
        """
        if answer_key == "title":
            return paper.title
        elif answer_key == "journal":
            return paper.journal
        elif answer_key == "benchmarks_used":
            return ", ".join(paper.benchmarks_used) if paper.benchmarks_used else "Not specified"
        elif answer_key == "keywords_found":
            return ", ".join(paper.keywords_found[:5]) if paper.keywords_found else "Not specified"
        elif answer_key == "validation_methods":
            return ", ".join(paper.validation_methods) if paper.validation_methods else "Not specified"
        else:
            return "Not available"
    
    def process_papers_for_training(
        self, 
        papers: List[ScientificPaper],
        format_type: str = "causal_lm"
    ) -> Dataset:
        """Process papers for training in specified format.
        
        Args:
            papers: List of scientific papers
            format_type: Format type ("causal_lm", "instruction", "qa")
            
        Returns:
            HuggingFace Dataset object
        """
        if format_type == "causal_lm":
            return self.create_causal_lm_dataset(papers)
        elif format_type == "instruction":
            return self.create_instruction_dataset(papers)
        elif format_type == "qa":
            return self.create_qa_dataset(papers)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def get_tokenization_stats(self, dataset: Dataset, tokenizer_name: str = "meta-llama/Llama-2-7b-hf") -> Dict[str, Any]:
        """Get tokenization statistics for the dataset.
        
        Args:
            dataset: HuggingFace dataset
            tokenizer_name: Tokenizer to use for stats
            
        Returns:
            Dictionary with tokenization statistics
        """
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Get text field name
            text_field = "text" if "text" in dataset.column_names else "input"
            
            # Tokenize a sample of texts
            sample_size = min(100, len(dataset))
            sample_texts = dataset.select(range(sample_size))[text_field]
            
            token_lengths = []
            for text in sample_texts:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                token_lengths.append(len(tokens))
            
            stats = {
                "tokenizer": tokenizer_name,
                "sample_size": sample_size,
                "min_tokens": min(token_lengths),
                "max_tokens": max(token_lengths),
                "mean_tokens": sum(token_lengths) / len(token_lengths),
                "median_tokens": sorted(token_lengths)[len(token_lengths) // 2],
                "tokens_over_2048": sum(1 for length in token_lengths if length > 2048),
                "percentage_over_2048": (sum(1 for length in token_lengths if length > 2048) / len(token_lengths)) * 100
            }
            
            return stats
            
        except ImportError:
            logger.warning("Transformers not available for tokenization stats")
            return {"error": "Transformers library not available"}
        except Exception as e:
            logger.error(f"Error calculating tokenization stats: {e}")
            return {"error": str(e)}