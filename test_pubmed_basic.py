#!/usr/bin/env python3
"""
Basic test of PubMed API client functionality.
This script tests the client without making actual API calls.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data.pubmed_client import PubMedAPIClient, PubMedPaper


def test_pubmed_client_creation():
    """Test creating a PubMed client instance."""
    print("🔧 Testing PubMed client creation...")
    
    client = PubMedAPIClient(
        email="test@example.com",
        api_key=None,
        tool="ScientificLLM-Forge-Test"
    )
    
    print(f"✅ Client created successfully!")
    print(f"   Email: {client.email}")
    print(f"   Tool: {client.tool}")
    print(f"   Max requests per second: {client.max_requests_per_second}")
    print(f"   Max requests per day: {client.max_requests_per_day}")
    
    return client


def test_pubmed_paper_creation():
    """Test creating PubMed paper objects."""
    print("\n📄 Testing PubMed paper creation...")
    
    paper = PubMedPaper(
        pmid="12345",
        title="Test Paper: Genomics and Machine Learning",
        abstract="This is a test abstract about genomics and machine learning applications.",
        authors=["John Doe", "Jane Smith", "Bob Johnson"],
        journal="Test Journal of Bioinformatics",
        publication_date="2023-01-15",
        citation_count=42,
        keywords=["genomics", "machine learning", "bioinformatics"],
        doi="10.1234/test.2023.001",
        mesh_terms=["Genomics", "Machine Learning", "Computational Biology"],
        publication_type="research-article",
        language="en",
        last_updated="2023-01-20"
    )
    
    print(f"✅ Paper created successfully!")
    print(f"   PMID: {paper.pmid}")
    print(f"   Title: {paper.title}")
    print(f"   Authors: {', '.join(paper.authors)}")
    print(f"   Journal: {paper.journal}")
    print(f"   Date: {paper.publication_date}")
    print(f"   Citations: {paper.citation_count}")
    print(f"   Keywords: {', '.join(paper.keywords)}")
    print(f"   DOI: {paper.doi}")
    
    return paper


def test_search_terms():
    """Test the predefined search terms."""
    print("\n🔍 Testing search terms...")
    
    client = PubMedAPIClient(email="test@example.com")
    
    print(f"✅ Found {len(client.default_search_terms)} predefined search terms:")
    for i, term in enumerate(client.default_search_terms[:5], 1):
        print(f"   {i}. {term}")
    
    if len(client.default_search_terms) > 5:
        print(f"   ... and {len(client.default_search_terms) - 5} more terms")
    
    return client.default_search_terms


def test_usage_stats():
    """Test usage statistics functionality."""
    print("\n📊 Testing usage statistics...")
    
    client = PubMedAPIClient(email="test@example.com")
    stats = client.get_usage_stats()
    
    print(f"✅ Usage statistics retrieved:")
    print(f"   Daily requests used: {stats['daily_requests']}")
    print(f"   Max daily requests: {stats['max_daily_requests']}")
    print(f"   Requests remaining: {stats['requests_remaining']}")
    print(f"   Last request date: {stats['last_request_date']}")
    print(f"   Current requests per second: {stats['current_requests_per_second']}")


def test_data_structures():
    """Test data structure conversions."""
    print("\n🔄 Testing data structure conversions...")
    
    paper = PubMedPaper(
        pmid="67890",
        title="Another Test Paper",
        abstract="Another test abstract.",
        authors=["Alice Brown", "Charlie Davis"],
        journal="Another Test Journal",
        publication_date="2023-02-01",
        citation_count=15,
        keywords=["test", "example"]
    )
    
    # Test conversion to dictionary
    from dataclasses import asdict
    paper_dict = asdict(paper)
    
    print(f"✅ Paper converted to dictionary:")
    print(f"   Keys: {list(paper_dict.keys())}")
    print(f"   PMID: {paper_dict['pmid']}")
    print(f"   Title: {paper_dict['title']}")
    print(f"   Authors: {paper_dict['authors']}")


def main():
    """Run all basic tests."""
    print("🧪 PubMed API Client Basic Tests")
    print("=" * 50)
    print("This script tests the PubMed client functionality")
    print("without making actual API calls.")
    print()
    
    try:
        # Test client creation
        client = test_pubmed_client_creation()
        
        # Test paper creation
        paper = test_pubmed_paper_creation()
        
        # Test search terms
        search_terms = test_search_terms()
        
        # Test usage stats
        test_usage_stats()
        
        # Test data structures
        test_data_structures()
        
        print("\n" + "=" * 50)
        print("✅ All basic tests passed!")
        print("\n📝 Next steps:")
        print("1. To test with real API calls, update the email address in examples/pubmed_example.py")
        print("2. Run: python examples/pubmed_example.py")
        print("3. Or use the data collection script: python scripts/collect_pubmed_data.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 