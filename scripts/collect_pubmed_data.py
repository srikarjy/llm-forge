#!/usr/bin/env python3
"""
PubMed data collection script for ScientificLLM-Forge.

This script demonstrates how to use the PubMed API client to collect
scientific papers for genomics + AI/ML research.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pubmed_client import PubMedAPIClient
from utils.config import ConfigManager
from utils.logger import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect PubMed data for scientific LLM training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pubmed_api.yaml",
        help="Path to PubMed API configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/pubmed",
        help="Output directory for collected data"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=1000,
        help="Maximum number of papers to collect"
    )
    parser.add_argument(
        "--date-from",
        type=str,
        help="Start date for paper collection (YYYY/MM/DD)"
    )
    parser.add_argument(
        "--date-to",
        type=str,
        help="End date for paper collection (YYYY/MM/DD)"
    )
    parser.add_argument(
        "--search-terms",
        type=str,
        nargs="+",
        help="Additional custom search terms"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for bulk downloads"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_manager = ConfigManager()
    return config_manager.load_config(config_path)


def setup_output_directory(output_dir: str) -> Path:
    """Set up output directory structure.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Path object for output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / "raw").mkdir(exist_ok=True)
    (output_path / "processed").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)
    
    return output_path


def collect_papers(client: PubMedAPIClient, config: dict, args) -> dict:
    """Collect papers using the PubMed API client.
    
    Args:
        client: PubMed API client
        config: Configuration dictionary
        args: Command line arguments
        
    Returns:
        Dictionary with collection statistics
    """
    logger = logging.getLogger(__name__)
    
    # Set default dates if not provided
    if not args.date_from:
        args.date_from = config.get("data", {}).get("pubmed", {}).get("date_from", "2020/01/01")
    if not args.date_to:
        args.date_to = config.get("data", {}).get("pubmed", {}).get("date_to", "2024/01/01")
    
    # Get custom search terms
    custom_terms = args.search_terms or config.get("data", {}).get("pubmed", {}).get("search_terms", [])
    
    logger.info(f"Starting paper collection with parameters:")
    logger.info(f"  Date range: {args.date_from} to {args.date_to}")
    logger.info(f"  Max papers: {args.max_papers}")
    logger.info(f"  Custom terms: {custom_terms}")
    
    # Search for genomics + AI/ML papers
    papers = client.search_genomics_ai_papers(
        max_results=args.max_papers,
        date_from=args.date_from,
        date_to=args.date_to,
        custom_terms=custom_terms
    )
    
    logger.info(f"Found {len(papers)} papers")
    
    # Convert to dictionaries for JSON serialization
    papers_data = []
    for paper in papers:
        paper_dict = {
            "pmid": paper.pmid,
            "title": paper.title,
            "abstract": paper.abstract,
            "authors": paper.authors,
            "journal": paper.journal,
            "publication_date": paper.publication_date,
            "citation_count": paper.citation_count,
            "keywords": paper.keywords,
            "doi": paper.doi,
            "mesh_terms": paper.mesh_terms,
            "publication_type": paper.publication_type,
            "language": paper.language,
            "last_updated": paper.last_updated
        }
        papers_data.append(paper_dict)
    
    return {
        "total_papers": len(papers_data),
        "papers": papers_data,
        "collection_date": datetime.now().isoformat(),
        "date_range": {
            "from": args.date_from,
            "to": args.date_to
        },
        "search_terms": custom_terms
    }


def save_collection_results(results: dict, output_dir: Path, config: dict):
    """Save collection results to files.
    
    Args:
        results: Collection results dictionary
        output_dir: Output directory path
        config: Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw data
    raw_file = output_dir / "raw" / f"pubmed_papers_{timestamp}.json"
    with open(raw_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved raw data to {raw_file}")
    
    # Save individual papers as separate files for easier processing
    papers_dir = output_dir / "raw" / "papers"
    papers_dir.mkdir(exist_ok=True)
    
    for paper in results["papers"]:
        pmid = paper["pmid"]
        paper_file = papers_dir / f"{pmid}.json"
        with open(paper_file, 'w', encoding='utf-8') as f:
            json.dump(paper, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(results['papers'])} individual paper files")
    
    # Create summary report
    summary = {
        "collection_summary": {
            "total_papers": results["total_papers"],
            "collection_date": results["collection_date"],
            "date_range": results["date_range"],
            "search_terms": results["search_terms"]
        },
        "usage_stats": {
            "daily_requests_used": 0,  # Will be updated below
            "requests_remaining": 10000
        },
        "file_locations": {
            "raw_data": str(raw_file),
            "individual_papers": str(papers_dir),
            "processed_data": str(output_dir / "processed")
        }
    }
    
    summary_file = output_dir / f"collection_summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved collection summary to {summary_file}")


def main():
    """Main function for PubMed data collection."""
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("pubmed_collection", log_level)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Setup output directory
        logger.info("Setting up output directory...")
        output_dir = setup_output_directory(args.output_dir)
        
        # Initialize PubMed API client
        logger.info("Initializing PubMed API client...")
        api_config = config.get("api", {}).get("pubmed", {})
        
        client = PubMedAPIClient(
            email=api_config.get("email", "your-email@example.com"),
            api_key=api_config.get("api_key"),
            base_url=api_config.get("base_url", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"),
            tool=api_config.get("tool", "ScientificLLM-Forge")
        )
        
        # Check if email is configured
        if client.email == "your-email@example.com":
            logger.error("Please configure your email in the PubMed API configuration file")
            logger.error("Edit configs/pubmed_api.yaml and set your email address")
            sys.exit(1)
        
        # Collect papers
        logger.info("Starting paper collection...")
        results = collect_papers(client, config, args)
        
        # Save results
        logger.info("Saving collection results...")
        save_collection_results(results, output_dir, config)
        
        # Print usage statistics
        usage_stats = client.get_usage_stats()
        logger.info("Collection completed successfully!")
        logger.info(f"Papers collected: {results['total_papers']}")
        logger.info(f"Daily requests used: {usage_stats['daily_requests']}")
        logger.info(f"Requests remaining: {usage_stats['requests_remaining']}")
        
        # Print next steps
        logger.info("\nNext steps:")
        logger.info("1. Review the collected papers in the raw data directory")
        logger.info("2. Process the data for training using the data preprocessing modules")
        logger.info("3. Use the processed data for model fine-tuning")
        
    except Exception as e:
        logger.error(f"Data collection failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 