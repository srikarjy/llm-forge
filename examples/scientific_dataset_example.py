#!/usr/bin/env python3
"""
Example script demonstrating ScientificDataModule usage.

This script shows how to load, filter, and process scientific papers
from the high-quality papers dataset for LLM fine-tuning.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.scientific_dataset import ScientificDataModule, QualityTier


def main():
    """Demonstrate ScientificDataModule usage."""
    
    print("=== Scientific Dataset Module Example ===\n")
    
    # Initialize the data module
    data_file = "data/high_quality_papers_demo.json"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the high-quality papers demo file exists.")
        return
    
    print(f"📁 Loading data from: {data_file}")
    module = ScientificDataModule(data_file)
    
    # 1. Load all papers
    print("\n1. Loading All Papers:")
    all_papers = module.load_papers(min_quality_score=0)
    print(f"   Total papers loaded: {len(all_papers)}")
    
    for paper in all_papers:
        print(f"   - {paper.title[:60]}... (Score: {paper.score}, Tier: {paper.tier.value})")
    
    # 2. Get dataset statistics
    print("\n2. Dataset Statistics:")
    stats = module.get_statistics()
    
    print(f"   Total papers: {stats['total_papers']}")
    print(f"   AI/ML papers: {stats['ai_ml_detected']} ({stats['ai_ml_percentage']:.1f}%)")
    
    print("\n   Quality tier distribution:")
    for tier, count in stats['quality_tiers'].items():
        if count > 0:
            print(f"     {tier}: {count}")
    
    print("\n   Score statistics:")
    score_stats = stats['score_statistics']
    print(f"     Min: {score_stats['min']}")
    print(f"     Max: {score_stats['max']}")
    print(f"     Mean: {score_stats['mean']:.1f}")
    print(f"     Median: {score_stats['median']}")
    
    print("\n   Top benchmarks used:")
    for benchmark, count in list(stats['benchmark_usage'].items())[:5]:
        print(f"     {benchmark}: {count} papers")
    
    print("\n   Journal distribution:")
    for journal, count in list(stats['journal_distribution'].items())[:3]:
        print(f"     {journal}: {count} papers")
    
    # 3. Filter papers by quality score
    print("\n3. Filtering by Quality Score:")
    high_quality_papers = module.load_papers(min_quality_score=100)
    print(f"   Papers with score >= 100: {len(high_quality_papers)}")
    
    for paper in high_quality_papers:
        print(f"   - {paper.title[:50]}... (Score: {paper.score})")
    
    # 4. Filter papers by quality tier
    print("\n4. Filtering by Quality Tier:")
    gold_papers = module.load_papers(
        min_quality_score=0,
        quality_tiers=[QualityTier.GOLD_STANDARD]
    )
    print(f"   Gold standard papers: {len(gold_papers)}")
    
    for paper in gold_papers:
        print(f"   - {paper.title[:50]}... (Tier: {paper.tier.value})")
    
    # 5. Filter by benchmarks
    print("\n5. Filtering by Benchmarks:")
    encode_papers = module.filter_by_benchmarks(["ENCODE"])
    print(f"   Papers using ENCODE benchmark: {len(encode_papers)}")
    
    tcga_papers = module.filter_by_benchmarks(["TCGA"])
    print(f"   Papers using TCGA benchmark: {len(tcga_papers)}")
    
    # 6. Show paper details
    print("\n6. Paper Details Example:")
    if all_papers:
        paper = all_papers[0]
        print(f"   Title: {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Journal: {paper.journal}")
        print(f"   Publication Date: {paper.publication_date}")
        print(f"   DOI: {paper.doi}")
        print(f"   Quality Score: {paper.score}")
        print(f"   Quality Tier: {paper.tier.value}")
        print(f"   AI/ML Detected: {paper.ai_ml_detected}")
        print(f"   Keywords: {', '.join(paper.keywords_found[:5])}...")
        print(f"   Benchmarks: {', '.join(paper.benchmarks_used)}")
        print(f"   Validation Methods: {', '.join(paper.validation_methods)}")
        
        # Show component scores
        scores = paper.component_scores
        print(f"   Component Scores:")
        print(f"     Methodological Innovation: {scores.methodological_innovation}")
        print(f"     Benchmark Usage: {scores.benchmark_usage}")
        print(f"     Validation Rigor: {scores.validation_rigor}")
        print(f"     Reproducibility: {scores.reproducibility}")
        print(f"     Synergy Bonus: {scores.synergy_bonus}")
    
    # 7. Generate training text
    print("\n7. Training Text Example:")
    if all_papers:
        paper = all_papers[0]
        training_text = paper.to_training_text(include_metadata=True)
        print("   Training text preview (first 300 chars):")
        print(f"   {training_text[:300]}...")
    
    # 8. Export training data
    print("\n8. Exporting Training Data:")
    
    # Export high-quality papers for training
    output_dir = Path("outputs/training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export in different formats
    formats_to_export = ["jsonl", "json", "txt"]
    
    for format_type in formats_to_export:
        output_file = output_dir / f"high_quality_papers.{format_type}"
        
        try:
            module.export_training_data(
                str(output_file),
                papers=high_quality_papers,
                format_type=format_type,
                include_metadata=True
            )
            print(f"   ✅ Exported {len(high_quality_papers)} papers to {output_file}")
            
            # Show file size
            file_size = output_file.stat().st_size
            print(f"      File size: {file_size:,} bytes")
            
        except Exception as e:
            print(f"   ❌ Failed to export {format_type}: {e}")
    
    # 9. Advanced filtering examples
    print("\n9. Advanced Filtering Examples:")
    
    # Papers from Nature journals
    nature_papers = module.get_papers_by_journal("Nature")
    print(f"   Papers from Nature: {len(nature_papers)}")
    
    nature_methods_papers = module.get_papers_by_journal("Nature Methods")
    print(f"   Papers from Nature Methods: {len(nature_methods_papers)}")
    
    # Papers using multiple benchmarks
    multi_benchmark_papers = module.filter_by_benchmarks(["ENCODE", "TCGA", "GTEx"])
    print(f"   Papers using ENCODE, TCGA, or GTEx: {len(multi_benchmark_papers)}")
    
    # Extract benchmarks from reasoning
    print("\n10. Benchmark Extraction from Reasoning:")
    for paper in all_papers[:2]:  # Show first 2 papers
        extracted = paper.extract_benchmarks_from_reasoning()
        if extracted:
            print(f"   {paper.title[:40]}...")
            print(f"     Reasoning: {paper.reasoning}")
            print(f"     Extracted benchmarks: {extracted}")
    
    print("\n=== Example Complete ===")
    print("\nNext steps:")
    print("1. Use the exported training data for LLM fine-tuning")
    print("2. Integrate with QLoRA configuration from Task 1")
    print("3. Implement the enhanced ModelTrainer in the next tasks")


if __name__ == "__main__":
    main()