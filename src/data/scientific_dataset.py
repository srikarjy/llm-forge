"""
Scientific dataset module for loading and processing genomics papers.

This module provides functionality for loading high-quality scientific papers
from the quality pipeline output and preparing them for LLM fine-tuning.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

try:
    from ..utils.logger import get_logger
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.logger import get_logger

logger = get_logger(__name__)


class QualityTier(Enum):
    """Quality tiers for scientific papers."""
    GOLD_STANDARD = "gold_standard"
    HIGH_QUALITY = "high_quality"
    MODERATE_QUALITY = "moderate_quality"
    LOW_QUALITY = "low_quality"
    
    @classmethod
    def from_string(cls, tier_str: str) -> "QualityTier":
        """Convert string to QualityTier enum."""
        tier_map = {
            "gold_standard": cls.GOLD_STANDARD,
            "high_quality": cls.HIGH_QUALITY,
            "moderate_quality": cls.MODERATE_QUALITY,
            "low_quality": cls.LOW_QUALITY,
        }
        return tier_map.get(tier_str.lower(), cls.LOW_QUALITY)


@dataclass
class ComponentScores:
    """Component scores for paper quality assessment."""
    methodological_innovation: int = 0
    benchmark_usage: int = 0
    validation_rigor: int = 0
    reproducibility: int = 0
    synergy_bonus: int = 0
    
    @classmethod
    def from_dict(cls, scores_dict: Dict[str, int]) -> "ComponentScores":
        """Create ComponentScores from dictionary."""
        return cls(
            methodological_innovation=scores_dict.get("methodological_innovation", 0),
            benchmark_usage=scores_dict.get("benchmark_usage", 0),
            validation_rigor=scores_dict.get("validation_rigor", 0),
            reproducibility=scores_dict.get("reproducibility", 0),
            synergy_bonus=scores_dict.get("synergy_bonus", 0),
        )


@dataclass
class ScientificPaper:
    """Data model for scientific papers from the quality pipeline."""
    
    # Core paper metadata
    pmid: str
    title: str
    authors: List[str]
    journal: str
    publication_date: str
    doi: str
    
    # Quality assessment
    score: float
    tier: QualityTier
    ai_ml_detected: bool
    component_scores: ComponentScores
    reasoning: List[str]
    
    # Content analysis
    keywords_found: List[str]
    benchmarks_used: List[str]
    validation_methods: List[str]
    
    # Processing metadata
    scored_at: str
    
    # Optional fields
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    
    @classmethod
    def from_dict(cls, paper_dict: Dict[str, Any]) -> "ScientificPaper":
        """Create ScientificPaper from dictionary."""
        return cls(
            pmid=paper_dict["pmid"],
            title=paper_dict["title"],
            authors=paper_dict["authors"],
            journal=paper_dict["journal"],
            publication_date=paper_dict["publication_date"],
            doi=paper_dict["doi"],
            score=float(paper_dict["score"]),
            tier=QualityTier.from_string(paper_dict["tier"]),
            ai_ml_detected=paper_dict.get("ai_ml_detected", False),
            component_scores=ComponentScores.from_dict(paper_dict.get("component_scores", {})),
            reasoning=paper_dict.get("reasoning", []),
            keywords_found=paper_dict.get("keywords_found", []),
            benchmarks_used=paper_dict.get("benchmarks_used", []),
            validation_methods=paper_dict.get("validation_methods", []),
            scored_at=paper_dict.get("scored_at", ""),
            abstract=paper_dict.get("abstract"),
            full_text=paper_dict.get("full_text"),
        )
    
    def to_training_text(self, include_metadata: bool = True) -> str:
        """Convert paper to training text format for LLM fine-tuning."""
        text_parts = []
        
        if include_metadata:
            text_parts.append(f"Title: {self.title}")
            text_parts.append(f"Authors: {', '.join(self.authors)}")
            text_parts.append(f"Journal: {self.journal}")
            text_parts.append(f"Publication Date: {self.publication_date}")
            
            if self.benchmarks_used:
                text_parts.append(f"Benchmarks: {', '.join(self.benchmarks_used)}")
            
            if self.keywords_found:
                text_parts.append(f"Keywords: {', '.join(self.keywords_found[:10])}")  # Limit keywords
        
        if self.abstract:
            text_parts.append(f"Abstract: {self.abstract}")
        
        if self.full_text:
            text_parts.append(f"Content: {self.full_text}")
        
        return "\n\n".join(text_parts)
    
    def extract_benchmarks_from_reasoning(self) -> List[str]:
        """Extract benchmark names from reasoning text using regex."""
        benchmarks = set()
        
        # Common genomics benchmarks patterns
        benchmark_patterns = [
            r'\b(ENCODE|encode)\b',
            r'\b(1000\s*Genomes?|1000G)\b',
            r'\b(TCGA|tcga)\b',
            r'\b(GTEx|gtex)\b',
            r'\b(COSMIC|cosmic)\b',
            r'\b(ClinVar|clinvar)\b',
            r'\b(dbSNP|dbsnp)\b',
            r'\b(GWAS|gwas)\b',
            r'\b(UniProt|uniprot)\b',
            r'\b(PDB|pdb)\b',
        ]
        
        reasoning_text = " ".join(self.reasoning)
        
        for pattern in benchmark_patterns:
            matches = re.findall(pattern, reasoning_text, re.IGNORECASE)
            for match in matches:
                benchmarks.add(match.upper())
        
        return list(benchmarks)


class ScientificDataModule:
    """Data module for loading and processing scientific papers."""
    
    def __init__(self, data_file: str):
        """Initialize the data module.
        
        Args:
            data_file: Path to the high-quality papers JSON file
        """
        self.data_file = Path(data_file)
        self.papers: List[ScientificPaper] = []
        self.metadata: Dict[str, Any] = {}
        self._loaded = False
        
        logger.info(f"Initialized ScientificDataModule with data file: {data_file}")
    
    def load_papers(
        self, 
        min_quality_score: float = 70.0,
        quality_tiers: Optional[List[QualityTier]] = None,
        reload: bool = False
    ) -> List[ScientificPaper]:
        """Load papers from the data file with optional filtering.
        
        Args:
            min_quality_score: Minimum quality score threshold
            quality_tiers: List of quality tiers to include (None = all)
            reload: Force reload from file even if already loaded
            
        Returns:
            List of filtered ScientificPaper objects
        """
        if not self._loaded or reload:
            self._load_from_file()
        
        filtered_papers = []
        
        for paper in self.papers:
            # Filter by quality score
            if paper.score < min_quality_score:
                continue
            
            # Filter by quality tiers
            if quality_tiers and paper.tier not in quality_tiers:
                continue
            
            filtered_papers.append(paper)
        
        logger.info(
            f"Loaded {len(filtered_papers)} papers "
            f"(min_score={min_quality_score}, tiers={quality_tiers})"
        )
        
        return filtered_papers
    
    def _load_from_file(self) -> None:
        """Load papers from the JSON file."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        logger.info(f"Loading papers from {self.data_file}")
        
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.metadata = data.get("metadata", {})
            papers_data = data.get("papers", [])
            
            self.papers = []
            validation_errors = []
            
            for i, paper_dict in enumerate(papers_data):
                is_valid, errors = self.validate_paper_format(paper_dict)
                
                if is_valid:
                    try:
                        paper = ScientificPaper.from_dict(paper_dict)
                        self.papers.append(paper)
                    except Exception as e:
                        validation_errors.append(f"Paper {i}: Failed to create ScientificPaper: {e}")
                else:
                    validation_errors.append(f"Paper {i}: {', '.join(errors)}")
            
            if validation_errors:
                logger.warning(f"Found {len(validation_errors)} validation errors:")
                for error in validation_errors[:5]:  # Show first 5 errors
                    logger.warning(f"  {error}")
                if len(validation_errors) > 5:
                    logger.warning(f"  ... and {len(validation_errors) - 5} more errors")
            
            self._loaded = True
            logger.info(f"Successfully loaded {len(self.papers)} valid papers")
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {self.data_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load papers from {self.data_file}: {e}")
    
    def validate_paper_format(self, paper_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate paper dictionary format.
        
        Args:
            paper_dict: Dictionary containing paper data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required_fields = [
            "pmid", "title", "authors", "journal", "publication_date", 
            "doi", "score", "tier"
        ]
        
        for field in required_fields:
            if field not in paper_dict:
                errors.append(f"Missing required field: {field}")
            elif paper_dict[field] is None:
                errors.append(f"Field {field} cannot be None")
        
        # Type validation
        if "pmid" in paper_dict and not isinstance(paper_dict["pmid"], str):
            errors.append("pmid must be a string")
        
        if "title" in paper_dict and not isinstance(paper_dict["title"], str):
            errors.append("title must be a string")
        
        if "authors" in paper_dict and not isinstance(paper_dict["authors"], list):
            errors.append("authors must be a list")
        
        if "score" in paper_dict:
            try:
                float(paper_dict["score"])
            except (ValueError, TypeError):
                errors.append("score must be a number")
        
        if "tier" in paper_dict and not isinstance(paper_dict["tier"], str):
            errors.append("tier must be a string")
        
        # Optional field validation
        optional_list_fields = ["reasoning", "keywords_found", "benchmarks_used", "validation_methods"]
        for field in optional_list_fields:
            if field in paper_dict and paper_dict[field] is not None:
                if not isinstance(paper_dict[field], list):
                    errors.append(f"{field} must be a list")
        
        return len(errors) == 0, errors
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded papers.
        
        Returns:
            Dictionary containing various statistics
        """
        if not self._loaded:
            self.load_papers()
        
        if not self.papers:
            return {"total_papers": 0}
        
        # Quality tier distribution
        tier_counts = {}
        for tier in QualityTier:
            tier_counts[tier.value] = sum(1 for p in self.papers if p.tier == tier)
        
        # Score statistics
        scores = [p.score for p in self.papers]
        score_stats = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "median": sorted(scores)[len(scores) // 2],
        }
        
        # Benchmark usage
        all_benchmarks = []
        for paper in self.papers:
            all_benchmarks.extend(paper.benchmarks_used)
        
        benchmark_counts = {}
        for benchmark in set(all_benchmarks):
            benchmark_counts[benchmark] = all_benchmarks.count(benchmark)
        
        # Journal distribution
        journal_counts = {}
        for paper in self.papers:
            journal_counts[paper.journal] = journal_counts.get(paper.journal, 0) + 1
        
        # AI/ML detection
        ai_ml_count = sum(1 for p in self.papers if p.ai_ml_detected)
        
        return {
            "total_papers": len(self.papers),
            "metadata": self.metadata,
            "quality_tiers": tier_counts,
            "score_statistics": score_stats,
            "benchmark_usage": dict(sorted(benchmark_counts.items(), key=lambda x: x[1], reverse=True)),
            "journal_distribution": dict(sorted(journal_counts.items(), key=lambda x: x[1], reverse=True)),
            "ai_ml_detected": ai_ml_count,
            "ai_ml_percentage": (ai_ml_count / len(self.papers)) * 100,
        }
    
    def filter_by_benchmarks(self, benchmarks: List[str]) -> List[ScientificPaper]:
        """Filter papers by benchmark usage.
        
        Args:
            benchmarks: List of benchmark names to filter by
            
        Returns:
            List of papers that use any of the specified benchmarks
        """
        if not self._loaded:
            self.load_papers()
        
        benchmarks_lower = [b.lower() for b in benchmarks]
        filtered_papers = []
        
        for paper in self.papers:
            paper_benchmarks_lower = [b.lower() for b in paper.benchmarks_used]
            
            # Check if any of the specified benchmarks are used
            if any(bench in paper_benchmarks_lower for bench in benchmarks_lower):
                filtered_papers.append(paper)
        
        logger.info(f"Filtered to {len(filtered_papers)} papers using benchmarks: {benchmarks}")
        return filtered_papers
    
    def export_training_data(
        self, 
        output_path: str,
        papers: Optional[List[ScientificPaper]] = None,
        format_type: str = "jsonl",
        include_metadata: bool = True
    ) -> None:
        """Export papers as training data for LLM fine-tuning.
        
        Args:
            output_path: Path to save the training data
            papers: List of papers to export (None = all loaded papers)
            format_type: Output format ('jsonl', 'json', 'txt')
            include_metadata: Whether to include paper metadata in training text
        """
        if papers is None:
            if not self._loaded:
                self.load_papers()
            papers = self.papers
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(papers)} papers to {output_path} (format: {format_type})")
        
        if format_type == "jsonl":
            with open(output_file, 'w', encoding='utf-8') as f:
                for paper in papers:
                    training_text = paper.to_training_text(include_metadata=include_metadata)
                    record = {
                        "text": training_text,
                        "pmid": paper.pmid,
                        "title": paper.title,
                        "score": paper.score,
                        "tier": paper.tier.value,
                        "benchmarks": paper.benchmarks_used,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        elif format_type == "json":
            training_data = []
            for paper in papers:
                training_text = paper.to_training_text(include_metadata=include_metadata)
                record = {
                    "text": training_text,
                    "pmid": paper.pmid,
                    "title": paper.title,
                    "score": paper.score,
                    "tier": paper.tier.value,
                    "benchmarks": paper.benchmarks_used,
                }
                training_data.append(record)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        elif format_type == "txt":
            with open(output_file, 'w', encoding='utf-8') as f:
                for i, paper in enumerate(papers):
                    if i > 0:
                        f.write("\n" + "="*80 + "\n\n")
                    
                    training_text = paper.to_training_text(include_metadata=include_metadata)
                    f.write(training_text)
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        logger.info(f"Successfully exported training data to {output_path}")
    
    def get_papers_by_tier(self, tier: QualityTier) -> List[ScientificPaper]:
        """Get papers filtered by quality tier.
        
        Args:
            tier: Quality tier to filter by
            
        Returns:
            List of papers with the specified tier
        """
        if not self._loaded:
            self.load_papers()
        
        return [paper for paper in self.papers if paper.tier == tier]
    
    def get_papers_by_journal(self, journal: str) -> List[ScientificPaper]:
        """Get papers from a specific journal.
        
        Args:
            journal: Journal name to filter by
            
        Returns:
            List of papers from the specified journal
        """
        if not self._loaded:
            self.load_papers()
        
        return [paper for paper in self.papers if paper.journal.lower() == journal.lower()]