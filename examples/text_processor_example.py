#!/usr/bin/env python3
"""
Example script demonstrating ScientificTextProcessor usage.

This script shows how to preprocess scientific papers for LLM fine-tuning
with genomics-specific text cleaning and dataset creation.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.scientific_dataset import ScientificDataModule
from data.text_processor import ScientificTextProcessor, PreprocessingConfig


def main():
    """Demonstrate ScientificTextProcessor usage."""
    
    print("=== Scientific Text Processor Example ===\n")
    
    # 1. Load scientific papers
    data_file = "data/high_quality_papers_demo.json"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure the high-quality papers demo file exists.")
        return
    
    print(f"📁 Loading papers from: {data_file}")
    data_module = ScientificDataModule(data_file)
    papers = data_module.load_papers(min_quality_score=100)
    print(f"   Loaded {len(papers)} high-quality papers\n")
    
    # 2. Create text processor with custom configuration
    print("2. Creating Text Processor:")
    config = PreprocessingConfig(
        remove_citations=True,
        remove_urls=True,
        normalize_gene_names=True,
        normalize_statistical_notation=True,
        max_chunk_length=1024,  # Smaller chunks for demo
        chunk_overlap=50,
        causal_lm_format=True
    )
    
    processor = ScientificTextProcessor(config)
    print(f"   ✅ Created processor with config:")
    print(f"      - Remove citations: {config.remove_citations}")
    print(f"      - Normalize gene names: {config.normalize_gene_names}")
    print(f"      - Max chunk length: {config.max_chunk_length}")
    print(f"      - Chunk overlap: {config.chunk_overlap}")
    
    # 3. Demonstrate text preprocessing
    print("\n3. Text Preprocessing Examples:")
    
    if papers:
        paper = papers[0]
        original_text = paper.to_training_text(include_metadata=True)
        
        print(f"   Original text (first 200 chars):")
        print(f"   {original_text[:200]}...")
        
        # Show individual preprocessing steps
        print(f"\n   Step-by-step preprocessing:")
        
        # Citations removal
        text_no_citations = processor.remove_citations(original_text)
        print(f"   After citation removal: {len(text_no_citations)} chars")
        
        # URL/DOI removal
        text_no_urls = processor.remove_urls_and_dois(text_no_citations)
        print(f"   After URL/DOI removal: {len(text_no_urls)} chars")
        
        # Statistical notation normalization
        text_normalized_stats = processor.normalize_statistical_notation(text_no_urls)
        print(f"   After stat normalization: {len(text_normalized_stats)} chars")
        
        # Whitespace normalization
        text_normalized_ws = processor.normalize_whitespace(text_normalized_stats)
        print(f"   After whitespace normalization: {len(text_normalized_ws)} chars")
        
        # Full preprocessing
        processed_text = processor.preprocess_scientific_text(original_text)
        print(f"\n   Fully processed text (first 200 chars):")
        print(f"   {processed_text[:200]}...")
        
        # Show text chunking
        print(f"\n   Text chunking:")
        chunks = processor.chunk_text(processed_text, max_length=300)
        print(f"   Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
            print(f"   Chunk {i+1} ({len(chunk.split())} words): {chunk[:100]}...")
    
    # 4. Create different dataset formats
    print("\n4. Creating Training Datasets:")
    
    # Causal Language Modeling Dataset
    print("   Creating Causal LM Dataset...")
    causal_dataset = processor.create_causal_lm_dataset(papers)
    print(f"   ✅ Created causal LM dataset with {len(causal_dataset)} records")
    
    # Show sample record
    if len(causal_dataset) > 0:
        sample_record = causal_dataset[0]
        print(f"      Sample record keys: {list(sample_record.keys())}")
        print(f"      Text length: {len(sample_record['text'])} chars")
        print(f"      PMID: {sample_record['pmid']}")
        print(f"      Title: {sample_record['title'][:50]}...")
    
    # Instruction-Following Dataset
    print("\n   Creating Instruction Dataset...")
    instruction_dataset = processor.create_instruction_dataset(
        papers, 
        instruction_types=["summarize", "extract_benchmarks", "identify_methods"]
    )
    print(f"   ✅ Created instruction dataset with {len(instruction_dataset)} records")
    
    # Show sample instruction
    if len(instruction_dataset) > 0:
        sample_instruction = instruction_dataset[0]
        print(f"      Sample instruction: {sample_instruction['instruction']}")
        print(f"      Input length: {len(sample_instruction['input'])} chars")
        print(f"      Output: {sample_instruction['output'][:100]}...")
    
    # Question-Answering Dataset
    print("\n   Creating QA Dataset...")
    qa_dataset = processor.create_qa_dataset(papers)
    print(f"   ✅ Created QA dataset with {len(qa_dataset)} records")
    
    # Show sample QA pair
    if len(qa_dataset) > 0:
        sample_qa = qa_dataset[0]
        print(f"      Sample question: {sample_qa['question']}")
        print(f"      Answer: {sample_qa['answer']}")
        print(f"      Context length: {len(sample_qa['context'])} chars")
    
    # 5. Export datasets
    print("\n5. Exporting Datasets:")
    
    output_dir = Path("outputs/processed_datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export causal LM dataset
    causal_file = output_dir / "causal_lm_dataset.json"
    causal_dataset.to_json(str(causal_file))
    print(f"   ✅ Exported causal LM dataset to {causal_file}")
    print(f"      File size: {causal_file.stat().st_size:,} bytes")
    
    # Export instruction dataset
    instruction_file = output_dir / "instruction_dataset.json"
    instruction_dataset.to_json(str(instruction_file))
    print(f"   ✅ Exported instruction dataset to {instruction_file}")
    print(f"      File size: {instruction_file.stat().st_size:,} bytes")
    
    # Export QA dataset
    qa_file = output_dir / "qa_dataset.json"
    qa_dataset.to_json(str(qa_file))
    print(f"   ✅ Exported QA dataset to {qa_file}")
    print(f"      File size: {qa_file.stat().st_size:,} bytes")
    
    # 6. Demonstrate different processing configurations
    print("\n6. Different Processing Configurations:")
    
    # Minimal processing
    minimal_config = PreprocessingConfig(
        remove_citations=False,
        remove_urls=False,
        normalize_gene_names=False,
        normalize_statistical_notation=False
    )
    minimal_processor = ScientificTextProcessor(minimal_config)
    
    if papers:
        original = papers[0].to_training_text()
        minimal_processed = minimal_processor.preprocess_scientific_text(original)
        full_processed = processor.preprocess_scientific_text(original)
        
        print(f"   Original text length: {len(original)} chars")
        print(f"   Minimal processing: {len(minimal_processed)} chars")
        print(f"   Full processing: {len(full_processed)} chars")
        print(f"   Reduction: {((len(original) - len(full_processed)) / len(original) * 100):.1f}%")
    
    # 7. Tokenization statistics (if transformers available)
    print("\n7. Tokenization Statistics:")
    
    try:
        stats = processor.get_tokenization_stats(causal_dataset)
        if "error" not in stats:
            print(f"   Tokenizer: {stats['tokenizer']}")
            print(f"   Sample size: {stats['sample_size']}")
            print(f"   Token length stats:")
            print(f"     Min: {stats['min_tokens']} tokens")
            print(f"     Max: {stats['max_tokens']} tokens")
            print(f"     Mean: {stats['mean_tokens']:.1f} tokens")
            print(f"     Median: {stats['median_tokens']} tokens")
            print(f"   Sequences over 2048 tokens: {stats['tokens_over_2048']} ({stats['percentage_over_2048']:.1f}%)")
        else:
            print(f"   ⚠️  Could not calculate tokenization stats: {stats['error']}")
    except Exception as e:
        print(f"   ⚠️  Error calculating tokenization stats: {e}")
    
    # 8. Advanced preprocessing examples
    print("\n8. Advanced Preprocessing Examples:")
    
    # Test citation removal
    citation_text = "This study [1,2,3] shows that previous work (Smith et al., 2023) and (Jones & Brown, 2022) found significant results."
    no_citations = processor.remove_citations(citation_text)
    print(f"   Citation removal:")
    print(f"     Before: {citation_text}")
    print(f"     After:  {no_citations}")
    
    # Test statistical notation
    stats_text = "Results show P < 0.05 and p-value = 0.001 with 95% CI and statistical significance ***."
    normalized_stats = processor.normalize_statistical_notation(stats_text)
    print(f"\n   Statistical notation:")
    print(f"     Before: {stats_text}")
    print(f"     After:  {normalized_stats}")
    
    # Test genomics terms preservation
    genomics_text = "This study uses ENCODE, TCGA, and GTEx datasets for GWAS analysis of SNPs and CNVs."
    processed_genomics = processor.preprocess_scientific_text(genomics_text)
    print(f"\n   Genomics terms preservation:")
    print(f"     Before: {genomics_text}")
    print(f"     After:  {processed_genomics}")
    
    print("\n=== Text Processing Complete ===")
    print("\nNext steps:")
    print("1. Use the exported datasets for LLM fine-tuning")
    print("2. Integrate with QLoRA configuration from Task 1")
    print("3. Implement the enhanced ModelTrainer in the next tasks")
    print("4. Consider using more advanced NLP libraries (spaCy, NLTK) for better preprocessing")


if __name__ == "__main__":
    main()