"""
PubMed API client for ScientificLLM-Forge.

This module provides a robust client for interacting with the PubMed API,
including rate limiting, error handling, and data extraction for scientific papers.
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
import random


@dataclass
class PubMedPaper:
    """Data class for PubMed paper information."""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    publication_date: str
    citation_count: int
    keywords: List[str]
    doi: Optional[str] = None
    mesh_terms: List[str] = None
    publication_type: str = None
    language: str = "en"
    last_updated: str = None
    
    def __post_init__(self):
        if self.mesh_terms is None:
            self.mesh_terms = []
        if self.keywords is None:
            self.keywords = []


class PubMedAPIClient:
    """Robust PubMed API client with rate limiting and error handling."""
    
    def __init__(self, email: str, api_key: Optional[str] = None, 
                 base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
                 tool: str = "ScientificLLM-Forge"):
        """Initialize the PubMed API client.
        
        Args:
            email: Email address (required for NCBI API)
            api_key: Optional API key for higher rate limits
            base_url: Base URL for NCBI E-utilities
            tool: Tool name for API requests
        """
        self.email = email
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.tool = tool
        
        # Rate limiting settings
        self.max_requests_per_second = 3
        self.max_requests_per_day = 10000
        self.request_timestamps = []
        self.daily_request_count = 0
        self.last_request_date = datetime.now().date()
        
        # Retry settings
        self.max_retries = 5
        self.base_delay = 1.0
        self.max_delay = 60.0
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': f'{tool}/1.0 ({email})'
        })
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Default search terms for genomics + AI/ML
        self.default_search_terms = [
            "genomics AND (machine learning OR artificial intelligence OR deep learning)",
            "genomics AND (AI OR ML OR neural network)",
            "genomic data AND (machine learning OR AI)",
            "DNA sequencing AND (artificial intelligence OR ML)",
            "genetic analysis AND (deep learning OR machine learning)",
            "bioinformatics AND (AI OR machine learning)",
            "precision medicine AND (artificial intelligence OR ML)",
            "genomic medicine AND (machine learning OR AI)",
            "genetic variants AND (machine learning OR deep learning)",
            "genomic prediction AND (artificial intelligence OR ML)"
        ]
    
    def _reset_daily_counter(self):
        """Reset daily request counter if it's a new day."""
        current_date = datetime.now().date()
        if current_date != self.last_request_date:
            self.daily_request_count = 0
            self.last_request_date = current_date
    
    def _check_rate_limits(self):
        """Check and enforce rate limits."""
        self._reset_daily_counter()
        
        # Check daily limit
        if self.daily_request_count >= self.max_requests_per_day:
            raise Exception(f"Daily request limit ({self.max_requests_per_day}) exceeded")
        
        # Check per-second limit
        current_time = time.time()
        # Remove timestamps older than 1 second
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if current_time - ts < 1.0]
        
        if len(self.request_timestamps) >= self.max_requests_per_second:
            sleep_time = 1.0 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                self.logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> requests.Response:
        """Make a request to the PubMed API with retry logic.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            API response
        """
        # Add common parameters
        params.update({
            'email': self.email,
            'tool': self.tool,
            'retmode': 'xml'
        })
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        for attempt in range(self.max_retries):
            try:
                self._check_rate_limits()
                
                # Record request timestamp
                self.request_timestamps.append(time.time())
                self.daily_request_count += 1
                
                self.logger.debug(f"Making request to {endpoint} (attempt {attempt + 1})")
                response = self.session.get(url, params=params, timeout=30)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Check for API errors in XML response
                if 'error' in response.text.lower():
                    self.logger.warning(f"API error in response: {response.text[:200]}")
                    raise Exception(f"API error: {response.text}")
                
                return response
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff with jitter
                delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), 
                           self.max_delay)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
    
    def search_papers(self, query: str, max_results: int = 100, 
                     date_from: Optional[str] = None, 
                     date_to: Optional[str] = None,
                     article_types: Optional[List[str]] = None) -> List[str]:
        """Search for papers and return PMIDs.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            date_from: Start date (YYYY/MM/DD format)
            date_to: End date (YYYY/MM/DD format)
            article_types: List of article types to filter by
            
        Returns:
            List of PMIDs
        """
        self.logger.info(f"Searching for papers with query: {query}")
        
        # Build date range filter
        date_filter = ""
        if date_from or date_to:
            date_filter = " AND "
            if date_from and date_to:
                date_filter += f"({date_from}[PDAT]:{date_to}[PDAT])"
            elif date_from:
                date_filter += f"({date_from}[PDAT]:3000[PDAT])"
            elif date_to:
                date_filter += f"(1900[PDAT]:{date_to}[PDAT])"
        
        # Build article type filter
        type_filter = ""
        if article_types:
            type_filter = " AND (" + " OR ".join([f'"{at}"[PTYP]' for at in article_types]) + ")"
        
        # Combine query with filters
        full_query = query + date_filter + type_filter
        
        params = {
            'db': 'pubmed',
            'term': full_query,
            'retmax': min(max_results, 100),  # API limit per request
            'sort': 'relevance'
        }
        
        pmids = []
        retstart = 0
        
        while len(pmids) < max_results:
            params['retstart'] = retstart
            
            try:
                response = self._make_request('esearch.fcgi', params)
                root = ET.fromstring(response.content)
                
                # Extract PMIDs
                id_list = root.find('.//IdList')
                if id_list is None:
                    break
                
                batch_pmids = [id_elem.text for id_elem in id_list.findall('Id')]
                if not batch_pmids:
                    break
                
                pmids.extend(batch_pmids)
                retstart += len(batch_pmids)
                
                # Check if we've reached the total count
                count_elem = root.find('.//Count')
                if count_elem is not None:
                    total_count = int(count_elem.text)
                    if retstart >= total_count:
                        break
                
                # Small delay between batches
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in search_papers: {str(e)}")
                break
        
        pmids = pmids[:max_results]
        self.logger.info(f"Found {len(pmids)} papers")
        return pmids
    
    def get_paper_details(self, pmid: str) -> Optional[PubMedPaper]:
        """Get detailed information for a single paper.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PubMedPaper object or None if not found
        """
        self.logger.debug(f"Fetching details for PMID: {pmid}")
        
        params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        
        try:
            response = self._make_request('efetch.fcgi', params)
            root = ET.fromstring(response.content)
            
            # Find the article
            article = root.find('.//PubmedArticle')
            if article is None:
                self.logger.warning(f"No article found for PMID: {pmid}")
                return None
            
            return self._parse_article_xml(article)
            
        except Exception as e:
            self.logger.error(f"Error fetching details for PMID {pmid}: {str(e)}")
            return None
    
    def _parse_article_xml(self, article_elem: ET.Element) -> PubMedPaper:
        """Parse article XML and extract paper details.
        
        Args:
            article_elem: XML element containing article data
            
        Returns:
            PubMedPaper object
        """
        # Extract basic information
        pmid = article_elem.find('.//PMID').text if article_elem.find('.//PMID') is not None else ""
        
        # Extract title
        title_elem = article_elem.find('.//ArticleTitle')
        title = " ".join(title_elem.itertext()) if title_elem is not None else ""
        
        # Extract abstract
        abstract_elem = article_elem.find('.//AbstractText')
        abstract = " ".join(abstract_elem.itertext()) if abstract_elem is not None else ""
        
        # Extract authors
        authors = []
        author_list = article_elem.find('.//AuthorList')
        if author_list is not None:
            for author in author_list.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
                elif last_name is not None:
                    authors.append(last_name.text)
        
        # Extract journal information
        journal_elem = article_elem.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ""
        
        # Extract publication date
        pub_date_elem = article_elem.find('.//PubDate')
        publication_date = self._parse_publication_date(pub_date_elem) if pub_date_elem is not None else ""
        
        # Extract DOI
        doi_elem = article_elem.find('.//ELocationID[@EIdType="doi"]')
        doi = doi_elem.text if doi_elem is not None else None
        
        # Extract MeSH terms
        mesh_terms = []
        mesh_heading_list = article_elem.find('.//MeshHeadingList')
        if mesh_heading_list is not None:
            for mesh_heading in mesh_heading_list.findall('.//MeshHeading'):
                descriptor = mesh_heading.find('DescriptorName')
                if descriptor is not None:
                    mesh_terms.append(descriptor.text)
        
        # Extract keywords
        keywords = []
        keyword_list = article_elem.find('.//KeywordList')
        if keyword_list is not None:
            for keyword in keyword_list.findall('.//Keyword'):
                keywords.append(keyword.text)
        
        # Extract publication type
        pub_type_elem = article_elem.find('.//PublicationType')
        publication_type = pub_type_elem.text if pub_type_elem is not None else ""
        
        # Extract language
        language_elem = article_elem.find('.//Language')
        language = language_elem.text if language_elem is not None else "en"
        
        # Extract last updated date
        last_updated_elem = article_elem.find('.//MedlineCitation/DateRevised')
        if last_updated_elem is not None:
            year = last_updated_elem.find('Year')
            month = last_updated_elem.find('Month')
            day = last_updated_elem.find('Day')
            if year is not None:
                last_updated = f"{year.text}-{month.text if month is not None else '01'}-{day.text if day is not None else '01'}"
            else:
                last_updated = None
        else:
            last_updated = None
        
        # Note: Citation count is not available in the basic PubMed API
        # Would need to use additional APIs like PubMed Central or external services
        citation_count = 0
        
        return PubMedPaper(
            pmid=pmid,
            title=title,
            abstract=abstract,
            authors=authors,
            journal=journal,
            publication_date=publication_date,
            citation_count=citation_count,
            keywords=keywords,
            doi=doi,
            mesh_terms=mesh_terms,
            publication_type=publication_type,
            language=language,
            last_updated=last_updated
        )
    
    def _parse_publication_date(self, pub_date_elem: ET.Element) -> str:
        """Parse publication date from XML element.
        
        Args:
            pub_date_elem: XML element containing publication date
            
        Returns:
            Formatted date string
        """
        year_elem = pub_date_elem.find('Year')
        month_elem = pub_date_elem.find('Month')
        day_elem = pub_date_elem.find('Day')
        
        if year_elem is None:
            return ""
        
        year = year_elem.text
        month = month_elem.text if month_elem is not None else "01"
        day = day_elem.text if day_elem is not None else "01"
        
        # Handle month names
        if month.isalpha():
            month_map = {
                'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
            }
            month = month_map.get(month.lower()[:3], '01')
        
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    
    def bulk_download(self, pmids: List[str], output_file: str,
                     batch_size: int = 50) -> Tuple[int, int]:
        """Download details for multiple papers in batches.
        
        Args:
            pmids: List of PMIDs to download
            output_file: Output JSON file path
            batch_size: Number of papers to process in each batch
            
        Returns:
            Tuple of (successful_downloads, total_attempts)
        """
        self.logger.info(f"Starting bulk download of {len(pmids)} papers")
        
        papers = []
        successful = 0
        total_attempts = len(pmids)
        
        # Process in batches
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(pmids) + batch_size - 1)//batch_size}")
            
            for pmid in batch_pmids:
                try:
                    paper = self.get_paper_details(pmid)
                    if paper is not None:
                        papers.append(asdict(paper))
                        successful += 1
                        self.logger.debug(f"Successfully downloaded PMID: {pmid}")
                    else:
                        self.logger.warning(f"Failed to download PMID: {pmid}")
                except Exception as e:
                    self.logger.error(f"Error downloading PMID {pmid}: {str(e)}")
            
            # Save intermediate results
            if papers:
                self._save_papers_to_json(papers, output_file)
            
            # Small delay between batches
            time.sleep(0.5)
        
        self.logger.info(f"Bulk download completed: {successful}/{total_attempts} papers downloaded")
        return successful, total_attempts
    
    def _save_papers_to_json(self, papers: List[Dict[str, Any]], output_file: str):
        """Save papers to JSON file.
        
        Args:
            papers: List of paper dictionaries
            output_file: Output file path
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with metadata
        data = {
            "metadata": {
                "download_date": datetime.now().isoformat(),
                "total_papers": len(papers),
                "source": "PubMed API",
                "tool": self.tool
            },
            "papers": papers
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(papers)} papers to {output_file}")
    
    def search_genomics_ai_papers(self, max_results: int = 100,
                                 date_from: Optional[str] = None,
                                 date_to: Optional[str] = None,
                                 custom_terms: Optional[List[str]] = None) -> List[PubMedPaper]:
        """Search for genomics + AI/ML papers using predefined terms.
        
        Args:
            max_results: Maximum number of results to return
            date_from: Start date (YYYY/MM/DD format)
            date_to: End date (YYYY/MM/DD format)
            custom_terms: Additional custom search terms
            
        Returns:
            List of PubMedPaper objects
        """
        self.logger.info("Searching for genomics + AI/ML papers")
        
        # Combine default and custom terms
        search_terms = self.default_search_terms.copy()
        if custom_terms:
            search_terms.extend(custom_terms)
        
        all_pmids = set()
        
        # Search with each term
        for term in search_terms:
            try:
                pmids = self.search_papers(
                    query=term,
                    max_results=max_results // len(search_terms),
                    date_from=date_from,
                    date_to=date_to,
                    article_types=["research-article", "review-article"]
                )
                all_pmids.update(pmids)
                
                if len(all_pmids) >= max_results:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error searching with term '{term}': {str(e)}")
                continue
        
        # Convert to list and limit results
        pmids_list = list(all_pmids)[:max_results]
        
        # Download paper details
        papers = []
        for pmid in pmids_list:
            try:
                paper = self.get_paper_details(pmid)
                if paper is not None:
                    papers.append(paper)
            except Exception as e:
                self.logger.error(f"Error downloading PMID {pmid}: {str(e)}")
        
        self.logger.info(f"Found {len(papers)} genomics + AI/ML papers")
        return papers
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            "daily_requests": self.daily_request_count,
            "max_daily_requests": self.max_requests_per_day,
            "requests_remaining": self.max_requests_per_day - self.daily_request_count,
            "last_request_date": self.last_request_date.isoformat(),
            "current_requests_per_second": len(self.request_timestamps)
        } 