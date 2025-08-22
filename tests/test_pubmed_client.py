"""
Tests for PubMed API client.

This module contains tests for the PubMedAPIClient class to verify
its functionality, rate limiting, and error handling.
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import timedelta
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.pubmed_client import PubMedAPIClient, PubMedPaper


class TestPubMedPaper:
    """Test cases for PubMedPaper dataclass."""
    
    def test_pubmed_paper_creation(self):
        """Test creating a PubMedPaper instance."""
        paper = PubMedPaper(
            pmid="12345",
            title="Test Paper",
            abstract="This is a test abstract",
            authors=["John Doe", "Jane Smith"],
            journal="Test Journal",
            publication_date="2023-01-01",
            citation_count=10,
            keywords=["test", "paper"]
        )
        
        assert paper.pmid == "12345"
        assert paper.title == "Test Paper"
        assert paper.abstract == "This is a test abstract"
        assert len(paper.authors) == 2
        assert paper.journal == "Test Journal"
        assert paper.publication_date == "2023-01-01"
        assert paper.citation_count == 10
        assert len(paper.keywords) == 2
    
    def test_pubmed_paper_defaults(self):
        """Test PubMedPaper with default values."""
        paper = PubMedPaper(
            pmid="12345",
            title="Test",
            abstract="Test",
            authors=[],
            journal="Test",
            publication_date="2023-01-01",
            citation_count=0,
            keywords=[]
        )
        
        assert paper.doi is None
        assert paper.mesh_terms == []
        assert paper.keywords == []
        assert paper.publication_type is None
        assert paper.language == "en"
        assert paper.last_updated is None


class TestPubMedAPIClient:
    """Test cases for PubMedAPIClient class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = PubMedAPIClient(
            email="test@example.com",
            api_key="test_key",
            tool="TestTool"
        )
    
    def test_client_initialization(self):
        """Test PubMedAPIClient initialization."""
        assert self.client.email == "test@example.com"
        assert self.client.api_key == "test_key"
        assert self.client.tool == "TestTool"
        assert self.client.max_requests_per_second == 3
        assert self.client.max_requests_per_day == 10000
        assert len(self.client.default_search_terms) > 0
    
    def test_rate_limiting_initialization(self):
        """Test rate limiting initialization."""
        assert len(self.client.request_timestamps) == 0
        assert self.client.daily_request_count == 0
        assert self.client.last_request_date is not None
    
    def test_default_search_terms(self):
        """Test that default search terms are properly configured."""
        terms = self.client.default_search_terms
        
        # Check that terms contain genomics and AI/ML keywords
        genomics_keywords = ["genomics", "genomic", "DNA", "genetic", "bioinformatics"]
        ai_keywords = ["machine learning", "artificial intelligence", "deep learning", "AI", "ML"]
        
        for term in terms:
            has_genomics = any(keyword in term.lower() for keyword in genomics_keywords)
            has_ai = any(keyword in term.lower() for keyword in ai_keywords)
            assert has_genomics and has_ai, f"Term '{term}' should contain both genomics and AI keywords"
    
    @patch('data.pubmed_client.requests.Session')
    def test_make_request_success(self, mock_session):
        """Test successful API request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "<xml>success</xml>"
        mock_response.content = b"<xml>success</xml>"
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        client = PubMedAPIClient(email="test@example.com")
        
        with patch.object(client, '_check_rate_limits'):
            response = client._make_request('esearch.fcgi', {'db': 'pubmed'})
            
            assert response == mock_response
            mock_session_instance.get.assert_called_once()
    
    @patch('data.pubmed_client.requests.Session')
    def test_make_request_with_retry(self, mock_session):
        """Test API request with retry logic."""
        # Mock failed then successful response
        mock_failed_response = Mock()
        mock_failed_response.raise_for_status.side_effect = Exception("Connection error")
        
        mock_success_response = Mock()
        mock_success_response.raise_for_status.return_value = None
        mock_success_response.text = "<xml>success</xml>"
        mock_success_response.content = b"<xml>success</xml>"
        
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = [mock_failed_response, mock_success_response]
        mock_session.return_value = mock_session_instance
        
        client = PubMedAPIClient(email="test@example.com")
        
        with patch.object(client, '_check_rate_limits'):
            with patch('time.sleep'):  # Mock sleep to speed up test
                response = client._make_request('esearch.fcgi', {'db': 'pubmed'})
                
                assert response == mock_success_response
                assert mock_session_instance.get.call_count == 2
    
    def test_parse_publication_date(self):
        """Test publication date parsing."""
        # Test with all components
        date_elem = Mock()
        date_elem.find.side_effect = lambda x: Mock(text="2023") if x == 'Year' else (
            Mock(text="Jan") if x == 'Month' else Mock(text="15") if x == 'Day' else None
        )
        
        result = self.client._parse_publication_date(date_elem)
        assert result == "2023-01-15"
        
        # Test with missing month and day
        date_elem.find.side_effect = lambda x: Mock(text="2023") if x == 'Year' else None
        result = self.client._parse_publication_date(date_elem)
        assert result == "2023-01-01"
        
        # Test with missing year
        date_elem.find.side_effect = lambda x: None
        result = self.client._parse_publication_date(date_elem)
        assert result == ""
    
    def test_parse_article_xml(self):
        """Test XML article parsing."""
        # Create a mock XML element
        article_elem = Mock()
        
        # Mock PMID
        pmid_elem = Mock()
        pmid_elem.text = "12345"
        article_elem.find.return_value = pmid_elem
        
        # Mock title
        title_elem = Mock()
        title_elem.itertext.return_value = ["Test Title"]
        article_elem.find.side_effect = lambda x: (
            pmid_elem if x == './/PMID' else
            title_elem if x == './/ArticleTitle' else
            None
        )
        
        # Mock other elements to return None
        def mock_find(xpath):
            if xpath == './/PMID':
                return pmid_elem
            elif xpath == './/ArticleTitle':
                return title_elem
            else:
                return None
        
        article_elem.find.side_effect = mock_find
        
        # Test parsing
        with patch.object(self.client, '_parse_publication_date', return_value="2023-01-01"):
            paper = self.client._parse_article_xml(article_elem)
            
            assert paper.pmid == "12345"
            assert paper.title == "Test Title"
            assert paper.abstract == ""
            assert paper.authors == []
            assert paper.journal == ""
            assert paper.publication_date == "2023-01-01"
    
    def test_get_usage_stats(self):
        """Test usage statistics retrieval."""
        stats = self.client.get_usage_stats()
        
        assert "daily_requests" in stats
        assert "max_daily_requests" in stats
        assert "requests_remaining" in stats
        assert "last_request_date" in stats
        assert "current_requests_per_second" in stats
        
        assert stats["max_daily_requests"] == 10000
        assert stats["requests_remaining"] == 10000 - stats["daily_requests"]
    
    def test_save_papers_to_json(self):
        """Test saving papers to JSON file."""
        papers = [
            {
                "pmid": "12345",
                "title": "Test Paper 1",
                "abstract": "Test abstract 1",
                "authors": ["Author 1"],
                "journal": "Test Journal",
                "publication_date": "2023-01-01",
                "citation_count": 10,
                "keywords": ["test"]
            },
            {
                "pmid": "67890",
                "title": "Test Paper 2",
                "abstract": "Test abstract 2",
                "authors": ["Author 2"],
                "journal": "Test Journal",
                "publication_date": "2023-01-02",
                "citation_count": 5,
                "keywords": ["test"]
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_papers.json"
            
            self.client._save_papers_to_json(papers, str(output_file))
            
            assert output_file.exists()
            
            # Verify JSON content
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "papers" in data
            assert len(data["papers"]) == 2
            assert data["papers"][0]["pmid"] == "12345"
            assert data["papers"][1]["pmid"] == "67890"
    
    @patch('data.pubmed_client.requests.Session')
    def test_search_papers_mock(self, mock_session):
        """Test paper search with mocked API."""
        # Mock successful search response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.content = b"""
        <eSearchResult>
            <Count>2</Count>
            <IdList>
                <Id>12345</Id>
                <Id>67890</Id>
            </IdList>
        </eSearchResult>
        """
        
        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance
        
        client = PubMedAPIClient(email="test@example.com")
        
        with patch.object(client, '_check_rate_limits'):
            with patch('time.sleep'):
                pmids = client.search_papers("genomics AND machine learning", max_results=10)
                
                assert len(pmids) == 2
                assert "12345" in pmids
                assert "67890" in pmids
    
    def test_search_genomics_ai_papers(self):
        """Test genomics + AI/ML paper search."""
        client = PubMedAPIClient(email="test@example.com")
        
        # Mock the search_papers method
        with patch.object(client, 'search_papers', return_value=["12345", "67890"]):
            with patch.object(client, 'get_paper_details') as mock_get_details:
                # Mock paper details
                mock_paper = Mock()
                mock_paper.pmid = "12345"
                mock_paper.title = "Test Paper"
                mock_get_details.return_value = mock_paper
                
                papers = client.search_genomics_ai_papers(max_results=10)
                
                assert len(papers) == 2
                assert papers[0].pmid == "12345"
                assert papers[1].pmid == "12345"  # Same mock paper returned twice


class TestPubMedClientIntegration:
    """Integration tests for PubMed API client."""
    
    def test_client_with_real_config(self):
        """Test client with real configuration structure."""
        client = PubMedAPIClient(
            email="test@example.com",
            api_key="test_key",
            base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            tool="ScientificLLM-Forge"
        )
        
        # Verify all required attributes are set
        assert client.email == "test@example.com"
        assert client.api_key == "test_key"
        assert "eutils.ncbi.nlm.nih.gov" in client.base_url
        assert client.tool == "ScientificLLM-Forge"
        
        # Verify search terms are properly formatted
        for term in client.default_search_terms:
            assert "AND" in term, f"Search term should contain AND operator: {term}"
            assert "(" in term and ")" in term, f"Search term should contain parentheses: {term}"
    
    def test_rate_limiting_logic(self):
        """Test rate limiting logic."""
        client = PubMedAPIClient(email="test@example.com")
        
        # Test daily counter reset
        client.daily_request_count = 100
        client.last_request_date = (client.last_request_date - timedelta(days=1))
        
        client._reset_daily_counter()
        assert client.daily_request_count == 0
        
        # Test rate limiting check
        client.request_timestamps = [time.time() - 0.5, time.time() - 0.3, time.time() - 0.1]
        
        # Should not raise exception for 3 requests in 1 second
        try:
            client._check_rate_limits()
        except Exception:
            pytest.fail("Rate limiting should not trigger for 3 requests in 1 second")


if __name__ == "__main__":
    pytest.main([__file__]) 