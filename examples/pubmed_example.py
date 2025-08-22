#!/usr/bin/env python3
"""
Example usage of PubMed API client for ScientificLLM-Forge.

This script demonstrates how to use the PubMedAPIClient to collect
scientific papers for genomics + AI/ML research.
"""

import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pubmed_client import PubMedAPIClient
from utils.logger import setup_logger


def main():
    """Example usage of PubMed API client."""
    
    # Setup logging
    logger = setup_logger("pubmed_example", level="INFO")
    
    # Initialize the PubMed API client
    # Note: You need to provide your email address for NCBI API access
    client = PubMedAPIClient(
        email="your-email@example.com",  # Replace with your email
        api_key=None,  # Optional: Add your API key for higher rate limits
        tool="ScientificLLM-Forge-Example"
    )
    
    logger.info("PubMed API client initialized")
    
    # Example 1: Search for papers with a specific query
    logger.info("Example 1: Searching for papers with specific query")
    try:
        pmids = client.search_papers(
            query="genomics AND machine learning",
            max_results=10,
            date_from="2023/01/01",
            date_to="2024/01/01"
        )
        logger.info(f"Found {len(pmids)} papers")
        
        # Get details for the first paper
        if pmids:
            paper = client.get_paper_details(pmids[0])
            if paper:
                logger.info(f"First paper: {paper.title}")
                logger.info(f"Authors: {', '.join(paper.authors[:3])}...")
                logger.info(f"Journal: {paper.journal}")
                logger.info(f"Publication date: {paper.publication_date}")
                logger.info(f"Abstract preview: {paper.abstract[:100]}...")
        
    except Exception as e:
        logger.error(f"Error in example 1: {str(e)}")
    
    # Example 2: Search for genomics + AI/ML papers using predefined terms
    logger.info("\nExample 2: Searching for genomics + AI/ML papers")
    try:
        papers = client.search_genomics_ai_papers(
            max_results=5,
            date_from="2023/01/01",
            date_to="2024/01/01"
        )
        
        logger.info(f"Found {len(papers)} genomics + AI/ML papers")
        
        # Print details of found papers
        for i, paper in enumerate(papers[:3], 1):
            logger.info(f"\nPaper {i}:")
            logger.info(f"  PMID: {paper.pmid}")
            logger.info(f"  Title: {paper.title}")
            logger.info(f"  Journal: {paper.journal}")
            logger.info(f"  Date: {paper.publication_date}")
            logger.info(f"  Authors: {', '.join(paper.authors[:2])}...")
            logger.info(f"  Keywords: {', '.join(paper.keywords[:3])}...")
        
    except Exception as e:
        logger.error(f"Error in example 2: {str(e)}")
    
    # Example 3: Bulk download papers
    logger.info("\nExample 3: Bulk download papers")
    try:
        # Get some PMIDs first
        pmids = client.search_papers(
            query="genomics AND artificial intelligence",
            max_results=3,
            date_from="2023/01/01"
        )
        
        if pmids:
            # Save papers to JSON file
            output_file = "example_papers.json"
            successful, total = client.bulk_download(pmids, output_file, batch_size=2)
            
            logger.info(f"Bulk download completed: {successful}/{total} papers downloaded")
            logger.info(f"Results saved to: {output_file}")
            
            # Load and display the saved data
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Saved data contains {len(data['papers'])} papers")
            logger.info(f"Download date: {data['metadata']['download_date']}")
        
    except Exception as e:
        logger.error(f"Error in example 3: {str(e)}")
    
    # Example 4: Check usage statistics
    logger.info("\nExample 4: Usage statistics")
    try:
        stats = client.get_usage_stats()
        logger.info("Current usage statistics:")
        logger.info(f"  Daily requests used: {stats['daily_requests']}")
        logger.info(f"  Requests remaining: {stats['requests_remaining']}")
        logger.info(f"  Current requests per second: {stats['current_requests_per_second']}")
        
    except Exception as e:
        logger.error(f"Error in example 4: {str(e)}")
    
    logger.info("\nExample completed!")


if __name__ == "__main__":
    print("PubMed API Client Example")
    print("=" * 50)
    print("This example demonstrates how to use the PubMed API client")
    print("to collect scientific papers for genomics + AI/ML research.")
    print("\nNote: You need to provide your email address in the script")
    print("for NCBI API access. Edit the script and replace")
    print("'your-email@example.com' with your actual email address.")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
        main()
    except KeyboardInterrupt:
        print("\nExample cancelled by user.")
    except Exception as e:
        print(f"\nError running example: {str(e)}")
        print("Make sure you have configured your email address in the script.") 