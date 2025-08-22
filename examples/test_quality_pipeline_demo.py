#!/usr/bin/env python3
"""
Quality Pipeline Demo for ScientificLLM-Forge.

This script demonstrates the quality pipeline using mock data
so you can test the functionality without configuring PubMed API access.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pubmed_client import PubMedPaper
from data.quality_scorer import GenomicsAIQualityScorer, QualityLevel
from utils.logger import setup_logger


def create_mock_papers() -> List[PubMedPaper]:
    """Create mock PubMed papers for testing the pipeline."""
    
    mock_papers = [
        PubMedPaper(
            pmid="12345678",
            title="DNABERT: pre-trained Bidirectional Encoder Representations from Transformers for DNA sequence analysis",
            abstract="We present DNABERT, a novel foundation model for DNA sequence analysis using transformer architecture. Our model leverages self-attention mechanisms to capture long-range dependencies in genomic sequences. We evaluate DNABERT on multiple benchmark datasets including ENCODE, 1000 Genomes, and TCGA, achieving state-of-the-art performance on various genomics tasks. The model demonstrates strong interpretability through attention visualization and ablation studies. We provide comprehensive cross-validation results with statistical significance testing (p < 0.001). Our code is available on GitHub with detailed methods and data sharing protocols.",
            authors=["Zhang, L", "Wang, J", "Chen, X", "Li, Y"],
            journal="Nature Methods",
            publication_date="2023-06-15",
            citation_count=45,
            keywords=["transformer", "deep learning", "genomics", "dna sequence", "attention mechanism", "foundation model"],
            doi="10.1038/s41592-023-01851-8",
            mesh_terms=["Machine Learning", "Genomics", "DNA", "Neural Networks", "Transformer"],
            publication_type="research-article",
            language="en"
        ),
        
        PubMedPaper(
            pmid="12345679",
            title="Enformer: Predicting gene expression from DNA sequence using attention mechanisms",
            abstract="We present Enformer, a deep learning model that predicts gene expression from DNA sequence using attention mechanisms to capture long-range interactions. The model is trained on ENCODE data and validated using cross-validation and independent test sets. We compare against baseline methods and demonstrate improved performance. Statistical significance testing shows p < 0.001. The model architecture includes novel attention mechanisms for genomic sequences. Code is available on GitHub with detailed methods.",
            authors=["Avsec, Z", "Agarwal, V", "Visentin, D", "Ledsam, JR"],
            journal="Nature",
            publication_date="2023-09-20",
            citation_count=38,
            keywords=["deep learning", "attention mechanism", "gene expression", "dna sequence", "encode"],
            doi="10.1038/s41586-021-04020-1",
            mesh_terms=["Deep Learning", "Gene Expression", "DNA", "Attention", "Neural Networks"],
            publication_type="research-article",
            language="en"
        ),
        
        PubMedPaper(
            pmid="12345680",
            title="Machine learning approach for variant calling in genomic data",
            abstract="We apply machine learning techniques to variant calling in genomic sequences. The method uses supervised learning with cross-validation. We evaluate on a subset of 1000 Genomes data and show improved accuracy compared to existing methods. Statistical testing confirms significance. Code is available upon request.",
            authors=["Smith, A", "Johnson, B", "Brown, C"],
            journal="Bioinformatics",
            publication_date="2023-03-10",
            citation_count=12,
            keywords=["machine learning", "variant calling", "genomics", "supervised learning"],
            doi="10.1093/bioinformatics/btad123",
            mesh_terms=["Machine Learning", "Genomics", "Variation"],
            publication_type="research-article",
            language="en"
        ),
        
        PubMedPaper(
            pmid="12345681",
            title="RRTF1 gene expression analysis using PCR and Western blot techniques",
            abstract="We investigated the expression of RRTF1 gene in various cell lines using polymerase chain reaction (PCR) and Western blot analysis. Cell culture experiments were performed under standard conditions. Immunohistochemistry was used to localize protein expression. Flow cytometry analysis revealed cell cycle distribution. Statistical analysis was performed using Student's t-test. Our findings demonstrate the role of RRTF1 in cellular processes through traditional molecular biology approaches.",
            authors=["Davis, M", "Wilson, K", "Taylor, R"],
            journal="Molecular Biology Reports",
            publication_date="2023-01-15",
            citation_count=8,
            keywords=["pcr", "western blot", "cell culture", "immunohistochemistry", "flow cytometry"],
            doi="10.1007/s11033-023-08234-5",
            mesh_terms=["Polymerase Chain Reaction", "Western Blotting", "Cell Culture", "Immunohistochemistry"],
            publication_type="research-article",
            language="en"
        ),
        
        PubMedPaper(
            pmid="12345682",
            title="Multi-modal deep learning for cancer genomics and transcriptomics",
            abstract="We propose a novel multi-modal deep learning framework that integrates genomic and transcriptomic data for cancer prediction. Our transformer-based architecture processes both DNA sequence and RNA expression data simultaneously. We evaluate on TCGA and GTEx datasets with comprehensive cross-validation. The model achieves state-of-the-art performance with statistical significance (p < 0.001). Ablation studies demonstrate the importance of multi-modal integration. Code and data are available on GitHub with detailed implementation.",
            authors=["Garcia, P", "Martinez, L", "Rodriguez, A", "Lopez, M"],
            journal="Nature Communications",
            publication_date="2023-11-05",
            citation_count=25,
            keywords=["multi-modal", "deep learning", "cancer genomics", "transformer", "transcriptomics"],
            doi="10.1038/s41467-023-41234-7",
            mesh_terms=["Deep Learning", "Genomics", "Transcriptomics", "Neoplasms", "Multi-modal"],
            publication_type="research-article",
            language="en"
        )
    ]
    
    return mock_papers


class QualityPipelineDemo:
    """Demo pipeline for testing quality scoring with mock data."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize the demo pipeline.
        
        Args:
            output_dir: Directory to save results
        """
        self.logger = setup_logger("quality_pipeline_demo", level="INFO")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize quality scorer
        self.quality_scorer = GenomicsAIQualityScorer()
        
        # Statistics tracking
        self.stats = {
            "total_papers": 0,
            "papers_scored": 0,
            "high_quality_papers": 0,
            "tier_counts": {
                "gold_standard": 0,
                "high_quality": 0,
                "medium_quality": 0,
                "low_quality": 0,
                "filtered_out": 0
            }
        }
    
    def score_paper(self, paper: PubMedPaper) -> Dict[str, Any]:
        """Score a single paper for quality."""
        try:
            # Convert PubMedPaper to dictionary format expected by scorer
            paper_data = {
                "title": paper.title,
                "abstract": paper.abstract,
                "keywords": paper.keywords,
                "mesh_terms": paper.mesh_terms or [],
                "journal": paper.journal,
                "pmid": paper.pmid,
                "authors": paper.authors,
                "publication_date": paper.publication_date,
                "doi": paper.doi
            }
            
            # Score the paper
            score_result = self.quality_scorer.score_paper(paper_data)
            
            # Create result dictionary
            result = {
                "pmid": paper.pmid,
                "title": paper.title,
                "authors": paper.authors,
                "journal": paper.journal,
                "publication_date": paper.publication_date,
                "doi": paper.doi,
                "score": score_result.total_score,
                "tier": score_result.quality_level.value,
                "ai_ml_detected": score_result.ai_ml_detected,
                "component_scores": score_result.component_scores,
                "reasoning": score_result.reasoning[:3],  # Top 3 reasoning points
                "keywords_found": score_result.keywords_found,
                "benchmarks_used": score_result.benchmarks_used,
                "validation_methods": score_result.validation_methods,
                "scored_at": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error scoring paper {paper.pmid}: {str(e)}")
            return {
                "pmid": paper.pmid,
                "title": paper.title,
                "error": str(e),
                "score": 0,
                "tier": "error"
            }
    
    def process_papers(self, papers: List[PubMedPaper]) -> List[Dict[str, Any]]:
        """Process and score all papers."""
        self.logger.info(f"📊 Processing {len(papers)} mock papers...")
        
        scored_papers = []
        self.stats["total_papers"] = len(papers)
        
        for i, paper in enumerate(papers, 1):
            self.logger.info(f"Processing paper {i}/{len(papers)}: {paper.title[:60]}...")
            
            result = self.score_paper(paper)
            scored_papers.append(result)
            
            # Update statistics
            if "error" not in result:
                self.stats["papers_scored"] += 1
                tier = result["tier"]
                self.stats["tier_counts"][tier] += 1
                
                if result["score"] >= 70:
                    self.stats["high_quality_papers"] += 1
        
        return scored_papers
    
    def save_high_quality_papers(self, scored_papers: List[Dict[str, Any]]) -> None:
        """Save high-quality papers (score ≥ 70) to JSON file."""
        high_quality_papers = [
            paper for paper in scored_papers 
            if paper.get("score", 0) >= 70 and "error" not in paper
        ]
        
        if not high_quality_papers:
            self.logger.info("📝 No high-quality papers found to save")
            return
        
        output_file = self.output_dir / "high_quality_papers_demo.json"
        
        # Prepare data for saving
        data_to_save = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_papers_processed": len(scored_papers),
                "high_quality_papers_count": len(high_quality_papers),
                "quality_threshold": 70,
                "pipeline_version": "1.0",
                "demo_mode": True
            },
            "papers": high_quality_papers
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved {len(high_quality_papers)} high-quality papers to {output_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving high-quality papers: {str(e)}")
    
    def display_results(self, scored_papers: List[Dict[str, Any]]) -> None:
        """Display scoring results in a formatted table."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("📋 QUALITY SCORING RESULTS (DEMO)")
        self.logger.info("=" * 100)
        
        for i, paper in enumerate(scored_papers, 1):
            if "error" in paper:
                self.logger.info(f"{i:2d}. ❌ ERROR: {paper['title'][:60]}...")
                self.logger.info(f"    Error: {paper['error']}")
            else:
                # Format tier with emoji
                tier_emoji = {
                    "gold_standard": "🥇",
                    "high_quality": "🥈", 
                    "medium_quality": "🥉",
                    "low_quality": "📄",
                    "filtered_out": "🚫"
                }.get(paper["tier"], "❓")
                
                self.logger.info(f"{i:2d}. {tier_emoji} {paper['score']:3.0f}/100 - {paper['tier'].replace('_', ' ').title()}")
                self.logger.info(f"    📖 {paper['title'][:80]}...")
                self.logger.info(f"    👥 {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                self.logger.info(f"    📅 {paper['publication_date']} | {paper['journal']}")
                
                # Show top reasoning points
                if paper["reasoning"]:
                    self.logger.info(f"    💡 {' | '.join(paper['reasoning'][:2])}")
                
                self.logger.info("")
    
    def display_statistics(self) -> None:
        """Display pipeline statistics."""
        self.logger.info("\n" + "=" * 100)
        self.logger.info("📊 PIPELINE STATISTICS (DEMO)")
        self.logger.info("=" * 100)
        
        self.logger.info(f"📈 Total Papers Processed: {self.stats['total_papers']}")
        self.logger.info(f"✅ Successfully Scored: {self.stats['papers_scored']}")
        self.logger.info(f"🌟 High-Quality Papers (≥70): {self.stats['high_quality_papers']}")
        
        self.logger.info("\n📊 Quality Distribution:")
        tier_names = {
            "gold_standard": "🥇 Gold Standard",
            "high_quality": "🥈 High Quality", 
            "medium_quality": "🥉 Medium Quality",
            "low_quality": "📄 Low Quality",
            "filtered_out": "🚫 Filtered Out"
        }
        
        for tier, count in self.stats["tier_counts"].items():
            if self.stats["total_papers"] > 0:
                percentage = (count / self.stats["total_papers"]) * 100
                self.logger.info(f"   {tier_names[tier]}: {count:2d} papers ({percentage:5.1f}%)")
            else:
                self.logger.info(f"   {tier_names[tier]}: {count:2d} papers")
        
        # Success rate
        if self.stats["total_papers"] > 0:
            success_rate = (self.stats["papers_scored"] / self.stats["total_papers"]) * 100
            self.logger.info(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        # High-quality rate
        if self.stats["papers_scored"] > 0:
            hq_rate = (self.stats["high_quality_papers"] / self.stats["papers_scored"]) * 100
            self.logger.info(f"🌟 High-Quality Rate: {hq_rate:.1f}%")
    
    def run_demo(self) -> None:
        """Run the demo pipeline with mock data."""
        self.logger.info("🚀 Starting Quality Pipeline Demo")
        self.logger.info("=" * 50)
        
        # Create mock papers
        papers = create_mock_papers()
        self.logger.info(f"📚 Created {len(papers)} mock papers for testing")
        
        # Process papers
        scored_papers = self.process_papers(papers)
        
        # Display results
        self.display_results(scored_papers)
        
        # Save high-quality papers
        self.save_high_quality_papers(scored_papers)
        
        # Display statistics
        self.display_statistics()
        
        self.logger.info("\n✅ Demo completed successfully!")
        self.logger.info("💡 To test with real PubMed data, configure your email in test_quality_pipeline.py")


def main():
    """Main function to run the demo pipeline."""
    
    try:
        # Initialize and run demo pipeline
        pipeline = QualityPipelineDemo(output_dir="data")
        pipeline.run_demo()
        
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logging.error(f"Demo error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 