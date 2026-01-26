#!/usr/bin/env python3
"""
Integration tests for the full entropy-comovement replication pipeline.

Tests end-to-end functionality including:
1. Data loading -> processing -> output
2. Report generation with actual table data
3. Pipeline orchestration
4. Output file validation

Usage:
    pytest tests/test_integration.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestReportGenerator:
    """Tests for report generation functionality."""
    
    def test_table_formatter_initialization(self):
        """Test TableFormatter initialization."""
        from src.python.report_generator import TableFormatter
        
        formatter = TableFormatter(decimal_places=4)
        assert formatter.decimal_places == 4
        assert formatter.significance_stars is True
        
        formatter_no_stars = TableFormatter(significance_stars=False)
        assert formatter_no_stars.significance_stars is False
    
    def test_format_value_with_significance(self):
        """Test value formatting with significance stars."""
        from src.python.report_generator import TableFormatter
        
        formatter = TableFormatter(decimal_places=3)
        
        # Test significance levels
        assert "***" in formatter.format_value(0.05, 0.001)  # p < 0.01
        assert "**" in formatter.format_value(0.05, 0.03)    # p < 0.05
        assert "*" in formatter.format_value(0.05, 0.08)     # p < 0.10
        assert "*" not in formatter.format_value(0.05, 0.15) # p >= 0.10
    
    def test_format_value_types(self):
        """Test formatting of different value types."""
        from src.python.report_generator import TableFormatter
        
        formatter = TableFormatter(decimal_places=2)
        
        # Integer
        assert formatter.format_value(42) == "42"
        
        # Float
        assert formatter.format_value(3.14159) == "3.14"
        
        # String
        assert formatter.format_value("test") == "test"
        
        # NaN
        assert formatter.format_value(np.nan) == ""
    
    def test_format_coefficient_table(self):
        """Test coefficient table formatting."""
        from src.python.report_generator import TableFormatter
        
        formatter = TableFormatter(decimal_places=3)
        
        coefficients = pd.DataFrame({
            'Model 1': [0.042, -1.25],
            'Model 2': [0.038, -1.10],
        }, index=['Intercept', 'BETA'])
        
        t_stats = pd.DataFrame({
            'Model 1': [2.85, -3.21],
            'Model 2': [2.54, -2.89],
        }, index=['Intercept', 'BETA'])
        
        p_values = pd.DataFrame({
            'Model 1': [0.004, 0.001],
            'Model 2': [0.011, 0.004],
        }, index=['Intercept', 'BETA'])
        
        formatted = formatter.format_coefficient_table(coefficients, t_stats, p_values)
        
        # Should have 4 rows (2 coef + 2 t-stat)
        assert len(formatted) == 4
        
        # Check t-stats in parentheses
        t_row = formatted.iloc[1]
        assert any('(' in str(v) and ')' in str(v) for v in t_row.values)
    
    def test_to_markdown(self):
        """Test markdown table generation."""
        from src.python.report_generator import TableFormatter
        
        formatter = TableFormatter()
        
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
        }, index=['X', 'Y', 'Z'])
        
        md = formatter.to_markdown(df, title="Test Table", notes="Test notes")
        
        assert "### Test Table" in md
        assert "| X | 1 | 4 |" in md
        assert "*Notes: Test notes*" in md
    
    def test_to_latex(self):
        """Test LaTeX table generation."""
        from src.python.report_generator import TableFormatter
        
        formatter = TableFormatter()
        
        df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4],
        }, index=['X', 'Y'])
        
        latex = formatter.to_latex(df, caption="Test", label="tab:test")
        
        assert "\\begin{table}" in latex
        assert "\\caption{Test}" in latex
        assert "\\label{tab:test}" in latex
        assert "\\end{table}" in latex


class TestReportGeneratorClass:
    """Tests for ReportGenerator class."""
    
    def test_initialization(self, tmp_path):
        """Test ReportGenerator initialization."""
        from src.python.report_generator import ReportGenerator
        
        generator = ReportGenerator(
            output_dir=str(tmp_path / "reports"),
            tables_dir=str(tmp_path / "tables"),
            figures_dir=str(tmp_path / "figures")
        )
        
        assert generator.output_dir.exists()
        assert generator.formatter is not None
    
    def test_table_specs_defined(self):
        """Test that all table specs are properly defined."""
        from src.python.report_generator import ReportGenerator
        
        # All 7 tables should be defined
        assert len(ReportGenerator.TABLE_SPECS) == 7
        
        for i in range(1, 8):
            spec = ReportGenerator.TABLE_SPECS[i]
            assert spec.number == i
            assert len(spec.title) > 0
            assert len(spec.description) > 0
    
    def test_figure_specs_defined(self):
        """Test that all figure specs are properly defined."""
        from src.python.report_generator import ReportGenerator
        
        # 2 figures should be defined
        assert len(ReportGenerator.FIGURE_SPECS) == 2
        
        for i in range(1, 3):
            spec = ReportGenerator.FIGURE_SPECS[i]
            assert spec.number == i
            assert len(spec.title) > 0
    
    def test_generate_table_section(self, tmp_path):
        """Test table section generation."""
        from src.python.report_generator import ReportGenerator
        
        # Create test table
        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        
        test_table = pd.DataFrame({
            'Quintile': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
            'Return': [0.85, 0.92, 1.01, 1.15, 1.27],
        })
        test_table.to_csv(tables_dir / "Table_5_Returns.csv", index=False)
        
        generator = ReportGenerator(
            output_dir=str(tmp_path / "reports"),
            tables_dir=str(tables_dir)
        )
        
        section = generator.generate_table_section(5, format_type="markdown")
        
        assert "Table 5" in section
        assert "Returns" in section or "Alphas" in section
    
    def test_generate_replication_report(self, tmp_path):
        """Test full report generation."""
        from src.python.report_generator import ReportGenerator
        
        generator = ReportGenerator(
            output_dir=str(tmp_path / "reports"),
            tables_dir=str(tmp_path / "tables"),
            figures_dir=str(tmp_path / "figures")
        )
        
        report = generator.generate_replication_report()
        
        # Check required sections
        assert "# Replication Report" in report
        assert "Executive Summary" in report
        assert "Methodology" in report
        assert "Data" in report
        assert "Conclusion" in report
        assert "Appendix" in report
    
    def test_save_report(self, tmp_path):
        """Test report saving."""
        from src.python.report_generator import ReportGenerator
        
        reports_dir = tmp_path / "reports"
        generator = ReportGenerator(output_dir=str(reports_dir))
        
        report_text = "# Test Report\n\nThis is a test."
        filepath = generator.save_report(report_text, "test_report.md")
        
        assert Path(filepath).exists()
        
        with open(filepath) as f:
            content = f.read()
        
        assert content == report_text


class TestResultsAggregator:
    """Tests for ResultsAggregator class."""
    
    def test_initialization(self, tmp_path):
        """Test ResultsAggregator initialization."""
        from src.python.report_generator import ResultsAggregator
        
        aggregator = ResultsAggregator(str(tmp_path))
        
        assert aggregator.base_dir == tmp_path
    
    def test_collect_tables(self, tmp_path):
        """Test table collection."""
        from src.python.report_generator import ResultsAggregator
        
        # Create test tables
        tables_dir = tmp_path / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        
        for i in range(3):
            df = pd.DataFrame({'A': [i], 'B': [i*2]})
            df.to_csv(tables_dir / f"test_table_{i}.csv", index=False)
        
        aggregator = ResultsAggregator(str(tmp_path))
        tables = aggregator.collect_tables()
        
        assert len(tables) == 3
        assert all(isinstance(df, pd.DataFrame) for df in tables.values())
    
    def test_collect_figures(self, tmp_path):
        """Test figure collection."""
        from src.python.report_generator import ResultsAggregator
        
        # Create test figure files
        figures_dir = tmp_path / "outputs" / "figures"
        figures_dir.mkdir(parents=True)
        
        for name in ["figure_1.png", "figure_2.png", "plot.pdf"]:
            (figures_dir / name).touch()
        
        aggregator = ResultsAggregator(str(tmp_path))
        figures = aggregator.collect_figures()
        
        assert len(figures) == 3
    
    def test_get_summary_statistics(self, tmp_path):
        """Test summary statistics collection."""
        from src.python.report_generator import ResultsAggregator
        
        # Create directories and files
        tables_dir = tmp_path / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        (tables_dir / "test.csv").touch()
        
        figures_dir = tmp_path / "outputs" / "figures"
        figures_dir.mkdir(parents=True)
        (figures_dir / "test.png").touch()
        
        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)
        (data_dir / "test.parquet").touch()
        
        aggregator = ResultsAggregator(str(tmp_path))
        stats = aggregator.get_summary_statistics()
        
        assert stats['n_tables'] >= 0
        assert stats['n_figures'] >= 0
        assert 'tables_available' in stats
        assert 'figures_available' in stats
        assert 'data_files' in stats
    
    def test_validate_replication(self, tmp_path):
        """Test replication validation."""
        from src.python.report_generator import ResultsAggregator
        
        # Create some expected files
        tables_dir = tmp_path / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        (tables_dir / "Table_5_test.csv").touch()
        
        data_dir = tmp_path / "data" / "processed"
        data_dir.mkdir(parents=True)
        (data_dir / "down_asy_scores.parquet").touch()
        
        aggregator = ResultsAggregator(str(tmp_path))
        validation = aggregator.validate_replication()
        
        assert isinstance(validation, dict)
        assert 'Table 5' in validation
        assert validation['Table 5'] is True
        assert 'down_asy_scores.parquet' in validation


class TestReplicationPipeline:
    """Tests for the full replication pipeline."""
    
    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline initialization."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        assert pipeline.demo_mode is True
        assert pipeline.processed_dir.exists()
        assert pipeline.tables_dir.exists()
    
    def test_pipeline_phases_defined(self):
        """Test that all phases are defined."""
        from scripts.run_full_replication import ReplicationPipeline
        
        assert len(ReplicationPipeline.PHASES) == 6
        
        for i in range(1, 7):
            assert i in ReplicationPipeline.PHASES
    
    def test_demo_data_generation(self, tmp_path):
        """Test demo data generation."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        result = pipeline._generate_demo_data()
        
        assert result['n_stocks'] > 0
        assert result['n_observations'] > 0
        assert 'demo' in result
        
        # Check file was created
        assert (pipeline.processed_dir / 'demo_returns.parquet').exists()
    
    def test_phase_1_demo(self, tmp_path):
        """Test Phase 1 in demo mode."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        result = pipeline.run_phase_1_data_loading()
        
        assert result['success'] is True
        assert result['n_observations'] > 0
    
    def test_phase_3_demo(self, tmp_path):
        """Test Phase 3 (factor regressions) in demo mode."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        # Run prerequisite phase
        pipeline.run_phase_1_data_loading()
        
        result = pipeline.run_phase_3_factor_regressions()
        
        assert result['success'] is True
        assert 'high_low_return' in result
        
        # Check table was created
        assert (pipeline.tables_dir / 'Table_5_Returns_Alphas.csv').exists()
    
    def test_phase_6_demo(self, tmp_path):
        """Test Phase 6 (report generation) in demo mode."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        # Run prerequisite phases
        pipeline.run_phase_1_data_loading()
        pipeline.run_phase_3_factor_regressions()
        pipeline.run_phase_4_firm_characteristics()
        
        result = pipeline.run_phase_6_report()
        
        assert result['success'] is True
        assert 'report_path' in result
        assert Path(result['report_path']).exists()
    
    def test_full_pipeline_demo(self, tmp_path):
        """Test running full pipeline in demo mode."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        results = pipeline.run()
        
        # All phases should succeed
        assert all(r['success'] for r in results.values())
        
        # Check outputs exist
        assert (pipeline.tables_dir / 'Table_5_Returns_Alphas.csv').exists()
        assert (pipeline.tables_dir / 'Table_3_Correlations.csv').exists()
        assert (pipeline.reports_dir / 'replication_report.md').exists()
    
    def test_selective_phases(self, tmp_path):
        """Test running selective phases."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        # Run only phases 1 and 3
        results = pipeline.run(phases=[1, 3])
        
        assert len(results) == 2
        assert 1 in results
        assert 3 in results
        assert results[1]['success'] is True
        assert results[3]['success'] is True


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_demo_script_runs(self, tmp_path, monkeypatch):
        """Test that demo_phase7.py runs without errors."""
        # Change to tmp directory
        monkeypatch.chdir(project_root)
        
        # Import and run the main demos
        from src.python.report_generator import TableFormatter, ReportGenerator
        
        # Table formatter
        formatter = TableFormatter()
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['X', 'Y'])
        md = formatter.to_markdown(df)
        assert len(md) > 0
        
        # Report generator
        generator = ReportGenerator(
            output_dir=str(tmp_path / "reports"),
            tables_dir=str(tmp_path / "tables")
        )
        report = generator.generate_replication_report()
        assert len(report) > 1000  # Should be substantial
    
    def test_report_contains_all_sections(self, tmp_path):
        """Test that generated report has all required sections."""
        from src.python.report_generator import ReportGenerator
        
        generator = ReportGenerator(
            output_dir=str(tmp_path / "reports"),
            tables_dir=str(tmp_path / "tables"),
            figures_dir=str(tmp_path / "figures")
        )
        report = generator.generate_replication_report()
        
        required_sections = [
            "Replication Report",
            "Executive Summary",
            "Introduction",
            "Methodology",
            "Data",
            "Replication Results",  # Changed from "Results" 
            "Robustness",
            "Conclusion",
            "Appendix",
        ]
        
        for section in required_sections:
            assert section in report, f"Missing section: {section}"
    
    def test_pipeline_outputs_consistent(self, tmp_path):
        """Test that pipeline outputs are internally consistent."""
        from scripts.run_full_replication import ReplicationPipeline
        
        pipeline = ReplicationPipeline(
            base_dir=str(tmp_path),
            demo_mode=True,
            verbose=False
        )
        
        results = pipeline.run()
        
        # Load generated tables
        table5 = pd.read_csv(pipeline.tables_dir / 'Table_5_Returns_Alphas.csv')
        table3 = pd.read_csv(pipeline.tables_dir / 'Table_3_Correlations.csv')
        
        # Basic consistency checks
        assert len(table5) > 0
        assert len(table3) > 0
        
        # Check report references tables correctly
        with open(pipeline.reports_dir / 'replication_report.md') as f:
            report = f.read()
        
        assert "Table 5" in report
        assert "Table 3" in report


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
