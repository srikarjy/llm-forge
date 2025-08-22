#!/usr/bin/env python3
"""
Quality Pipeline Test for ScientificLLM-Forge.

This script demonstrates a complete pipeline for:
1. Searching PubMed for genomics + AI/ML papers
2. Scoring each paper for quality
3. Filtering and saving high-quality papers
4. Generating statistics and reports
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pubmed_client import PubMedAPIClient, PubMedPaper
from data.quality_scorer import GenomicsAIQualityScorer, QualityLevel
from utils.logger import setup_logger


class QualityPipeline:
    """Pipeline for searching and scoring scientific papers."""
    
    def __init__(self, email: str, output_dir: str = "data"):
        """Initialize the quality pipeline.
        
        Args:
            email: Email address for NCBI API access
            output_dir: Directory to save results
        """
        self.logger = setup_logger("quality_pipeline", level="INFO")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.pubmed_client = PubMedAPIClient(
            email=email,
            api_key=None,  # Optional: Add your API key for higher rate limits
            tool="ScientificLLM-Forge-Quality-Pipeline"
        )
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
    
    def search_papers(self, max_papers: int = 10) -> List[PubMedPaper]:
        """Search for genomics + AI/ML papers using PubMed.
        
        Args:
            max_papers: Maximum number of papers to search for
            
        Returns:
            List of PubMedPaper objects
        """
        self.logger.info(f"🔍 Searching for {max_papers} genomics + AI/ML papers...")
        
        try:
            # Use the predefined genomics + AI/ML search terms
            papers = self.pubmed_client.search_genomics_ai_papers(
                max_results=max_papers,
                date_from="2023/01/01",  # Papers from 2023 onwards
                date_to="2024/12/31"
            )
            
            self.logger.info(f"✅ Found {len(papers)} papers")
            return papers
            
        except Exception as e:
            self.logger.error(f"❌ Error searching papers: {str(e)}")
            return []
    
    def score_paper(self, paper: PubMedPaper) -> Dict[str, Any]:
        """Score a single paper for quality.
        
        Args:
            paper: PubMedPaper object
            
        Returns:
            Dictionary with scoring results
        """
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
        """Process and score all papers.
        
        Args:
            papers: List of PubMedPaper objects
            
        Returns:
            List of scored paper results
        """
        self.logger.info(f"📊 Processing {len(papers)} papers...")
        
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
        """Save high-quality papers (score ≥ 70) to JSON file.
        
        Args:
            scored_papers: List of scored paper results
        """
        high_quality_papers = [
            paper for paper in scored_papers 
            if paper.get("score", 0) >= 70 and "error" not in paper
        ]
        
        if not high_quality_papers:
            self.logger.info("📝 No high-quality papers found to save")
            return
        
        output_file = self.output_dir / "high_quality_papers.json"
        
        # Prepare data for saving
        data_to_save = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_papers_processed": len(scored_papers),
                "high_quality_papers_count": len(high_quality_papers),
                "quality_threshold": 70,
                "pipeline_version": "1.0"
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
        """Display scoring results in a formatted table.
        
        Args:
            scored_papers: List of scored paper results
        """
        self.logger.info("\n" + "=" * 100)
        self.logger.info("📋 QUALITY SCORING RESULTS")
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
        self.logger.info("📊 PIPELINE STATISTICS")
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
    
    def run_pipeline(self, max_papers: int = 10) -> None:
        """Run the complete quality pipeline.
        
        Args:
            max_papers: Maximum number of papers to process
        """
        self.logger.info("🚀 Starting Quality Pipeline")
        self.logger.info("=" * 50)
        
        # Step 1: Search for papers
        papers = self.search_papers(max_papers)
        if not papers:
            self.logger.error("❌ No papers found. Exiting pipeline.")
            return
        
        # Step 2: Score papers
        scored_papers = self.process_papers(papers)
        
        # Step 3: Display results
        self.display_results(scored_papers)
        
        # Step 4: Save high-quality papers
        self.save_high_quality_papers(scored_papers)
        
        # Step 5: Display statistics
        self.display_statistics()
        
        self.logger.info("\n✅ Pipeline completed successfully!")


def main():
    """Main function to run the quality pipeline."""
    
    # Configuration
    EMAIL = "your-email@example.com"  # ⚠️ REPLACE WITH YOUR ACTUAL EMAIL
    MAX_PAPERS = 10
    OUTPUT_DIR = "data"
    
    # Check if email is configured
    if EMAIL == "your-email@example.com":
        print("❌ ERROR: Please configure your email address in the script!")
        print("   Edit the EMAIL variable in examples/test_quality_pipeline.py")
        print("   Replace 'your-email@example.com' with your actual email address")
        return
    
    try:
        # Initialize and run pipeline
        pipeline = QualityPipeline(email=EMAIL, output_dir=OUTPUT_DIR)
        pipeline.run_pipeline(max_papers=MAX_PAPERS)
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        logging.error(f"Pipeline error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main() 