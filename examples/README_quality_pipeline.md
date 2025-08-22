# Quality Pipeline for ScientificLLM-Forge

This directory contains scripts for testing the complete quality pipeline that combines PubMed data collection with quality scoring.

## 📋 Overview

The quality pipeline provides a complete workflow for:
1. **Searching PubMed** for genomics + AI/ML papers
2. **Scoring papers** for quality using the GenomicsAIQualityScorer
3. **Filtering and saving** high-quality papers (score ≥ 70)
4. **Generating statistics** and reports

## 🚀 Quick Start

### Option 1: Demo Mode (No API Access Required)

Test the pipeline with mock data:

```bash
python examples/test_quality_pipeline_demo.py
```

This will:
- Create 5 mock papers (including DNABERT, Enformer, etc.)
- Score each paper for quality
- Display results with emojis and formatting
- Save high-quality papers to `data/high_quality_papers_demo.json`
- Show comprehensive statistics

### Option 2: Real PubMed Data (Requires Email Configuration)

To use real PubMed data:

1. **Configure your email** in `examples/test_quality_pipeline.py`:
   ```python
   EMAIL = "your-actual-email@example.com"  # Replace with your email
   ```

2. **Run the pipeline**:
   ```bash
   python examples/test_quality_pipeline.py
   ```

## 📊 Expected Results

### Demo Results
- **DNABERT**: 122/100 → 🥇 Gold Standard
- **Enformer**: 108/100 → 🥇 Gold Standard  
- **Multi-modal Paper**: 110/100 → 🥇 Gold Standard
- **Machine Learning Paper**: 54/100 → 🥉 Medium Quality
- **Traditional Biology**: 0/100 → 🚫 Filtered Out

### Quality Distribution
- 🥇 **Gold Standard**: 60% (3 papers)
- 🥈 **High Quality**: 0% (0 papers)
- 🥉 **Medium Quality**: 20% (1 paper)
- 📄 **Low Quality**: 0% (0 papers)
- 🚫 **Filtered Out**: 20% (1 paper)

## 📁 Output Files

### High-Quality Papers JSON
Saved to `data/high_quality_papers_demo.json` (demo) or `data/high_quality_papers.json` (real):

```json
{
  "metadata": {
    "generated_at": "2025-08-21T23:04:14.713169",
    "total_papers_processed": 5,
    "high_quality_papers_count": 3,
    "quality_threshold": 70,
    "pipeline_version": "1.0"
  },
  "papers": [
    {
      "pmid": "12345678",
      "title": "DNABERT: pre-trained Bidirectional Encoder...",
      "score": 122,
      "tier": "gold_standard",
      "component_scores": {
        "methodological_innovation": 35,
        "benchmark_usage": 30,
        "validation_rigor": 25,
        "reproducibility": 12,
        "synergy_bonus": 20
      },
      "reasoning": [
        "Transformer Genomics: 20 pts",
        "Foundation Model: 15 pts",
        "Long Range: 12 pts"
      ]
    }
  ]
}
```

## 🔧 Configuration

### Pipeline Settings
- **Max Papers**: Default 10 (configurable)
- **Quality Threshold**: 70+ for high-quality papers
- **Date Range**: 2023-2024 (configurable)
- **Output Directory**: `data/` (configurable)

### Quality Scoring Components
- **Methodological Innovation** (0-35 pts): Novel architectures, foundation models
- **Benchmark Usage** (0-30 pts): ENCODE, 1000 Genomes, TCGA, etc.
- **Validation Rigor** (0-25 pts): Cross-validation, statistical significance
- **Reproducibility** (0-20 pts): Code availability, data sharing
- **Synergy Bonus** (0-20 pts): AI/ML + Genomics + Benchmarks

## 🎯 Quality Tiers

- **🥇 Gold Standard** (90+): State-of-the-art methods, major benchmarks
- **🥈 High Quality** (70-89): Solid AI/ML approaches, good validation
- **🥉 Medium Quality** (50-69): Basic AI/ML methods, limited validation
- **📄 Low Quality** (30-49): Minimal AI/ML content
- **🚫 Filtered Out** (<30): No AI/ML content or traditional biology only

## 🛠️ Error Handling

The pipeline includes comprehensive error handling:
- **API Errors**: Graceful handling of PubMed API failures
- **Scoring Errors**: Individual paper scoring failures don't stop the pipeline
- **File I/O Errors**: Safe saving with error reporting
- **User Interruption**: Clean shutdown on Ctrl+C

## 📈 Statistics

The pipeline provides detailed statistics:
- **Success Rate**: Percentage of papers successfully scored
- **High-Quality Rate**: Percentage of scored papers that are high-quality
- **Quality Distribution**: Breakdown by tier with percentages
- **Processing Summary**: Total papers, scored papers, high-quality papers

## 🔍 Debugging

### Common Issues

1. **Email Not Configured**:
   ```
   ❌ ERROR: Please configure your email address in the script!
   ```
   **Solution**: Edit the EMAIL variable in the script

2. **API Rate Limits**:
   ```
   ❌ Error searching papers: Rate limit exceeded
   ```
   **Solution**: Wait and retry, or add API key for higher limits

3. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'data.pubmed_client'
   ```
   **Solution**: Ensure you're running from the project root directory

### Verbose Logging
Enable debug logging by modifying the logger setup:
```python
self.logger = setup_logger("quality_pipeline", level="DEBUG")
```

## 🚀 Next Steps

1. **Configure your email** for real PubMed access
2. **Adjust quality thresholds** if needed
3. **Customize search terms** for specific research areas
4. **Extend the pipeline** with additional data sources
5. **Integrate with your research workflow**

## 📚 Related Files

- `src/data/pubmed_client.py`: PubMed API client
- `src/data/quality_scorer.py`: Quality scoring logic
- `src/utils/logger.py`: Logging utilities
- `tests/test_quality_scorer.py`: Unit tests for quality scorer
- `examples/quality_scorer_example.py`: Basic quality scorer demo 