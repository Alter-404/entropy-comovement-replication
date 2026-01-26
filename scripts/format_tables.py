#!/usr/bin/env python3
"""
Format replication tables to match Jiang, Wu, and Zhou (2018) paper style.

This script reads the CSV tables and generates:
1. Publication-quality LaTeX tables
2. Formatted text tables for console viewing
3. Updated CSVs with proper formatting

Tables 1-7 follow the paper's exact structure and formatting conventions.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'python'))


class PaperTableFormatter:
    """
    Format tables to match the academic paper style from Jiang, Wu, Zhou (2018).
    
    Formatting conventions:
    - 2-4 decimal places for coefficients/returns
    - t-statistics in parentheses on separate rows
    - Significance stars: * p<0.10, ** p<0.05, *** p<0.01
    - Proper column alignment
    - Panel headers for multi-panel tables
    """
    
    def __init__(self, tables_dir: str = "outputs/tables"):
        self.tables_dir = Path(tables_dir)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
    def format_number(self, value, decimals=2, percentage=False):
        """Format a number with specified decimal places."""
        if pd.isna(value) or value is None:
            return ""
        try:
            val = float(value)
            if percentage:
                return f"{val:.{decimals}f}%"
            return f"{val:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    def add_significance_stars(self, coef, t_stat=None, p_val=None):
        """Add significance stars based on t-stat or p-value."""
        if pd.isna(coef) or coef == "":
            return ""
        
        coef_str = self.format_number(coef, decimals=2)
        
        # Determine significance
        stars = ""
        if p_val is not None and not pd.isna(p_val):
            if abs(p_val) < 0.01:
                stars = "***"
            elif abs(p_val) < 0.05:
                stars = "**"
            elif abs(p_val) < 0.10:
                stars = "*"
        elif t_stat is not None and not pd.isna(t_stat):
            t = abs(float(t_stat))
            if t > 2.576:  # ~1% significance
                stars = "***"
            elif t > 1.96:  # ~5% significance
                stars = "**"
            elif t > 1.645:  # ~10% significance
                stars = "*"
        
        return coef_str + stars
    
    def format_table_3(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 3: Cross-Sectional Correlations.
        
        Paper format - Full correlation matrix:
        - All 15 variables as rows and columns
        - Lower triangular matrix with significance stars
        - Diagonal = 1.00
        
        Handles both old format (Characteristic, Correlation, t-stat)
        and new format (full correlation matrix).
        """
        # Check if this is the new correlation matrix format
        if 'Characteristic' not in df.columns and df.index.name is not None or \
           (len(df.columns) > 3 and 'LQP' in df.columns):
            # New format: full correlation matrix
            return self._format_correlation_matrix(df)
        
        # Old format: single column correlations
        formatted = pd.DataFrame()
        formatted['Characteristic'] = df['Characteristic']
        
        # Format correlations with significance stars
        correlations = []
        for _, row in df.iterrows():
            corr = row.get('Correlation', row.get('correlation', 0))
            t_stat = row.get('t-statistic', row.get('t_statistic', None))
            correlations.append(self.add_significance_stars(corr, t_stat=t_stat))
        formatted['Correlation with DOWN_ASY'] = correlations
        
        # Format t-statistics
        t_stats = []
        for _, row in df.iterrows():
            t = row.get('t-statistic', row.get('t_statistic', None))
            if pd.notna(t):
                t_stats.append(f"({self.format_number(t, 2)})")
            else:
                t_stats.append("")
        formatted['t-statistic'] = t_stats
        
        # Generate LaTeX
        latex = self._generate_latex_table3(formatted)
        
        return formatted, latex
    
    def _format_correlation_matrix(self, df: pd.DataFrame) -> tuple:
        """
        Format the full correlation matrix for Table 3.
        
        Paper format:
        - Lower triangular matrix
        - Correlations rounded to 2 decimals
        - 1.00 on diagonal
        """
        # Round to 2 decimal places
        formatted = df.round(2)
        
        # Generate LaTeX for correlation matrix
        latex = self._generate_latex_correlation_matrix(formatted)
        
        return formatted, latex
    
    def _generate_latex_correlation_matrix(self, df: pd.DataFrame) -> str:
        """Generate LaTeX table for correlation matrix."""
        lines = []
        
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\caption{Cross-Sectional Correlations}")
        lines.append(r"\label{tab:correlations}")
        lines.append(r"\footnotesize")
        
        # Column format
        n_cols = len(df.columns)
        col_format = "l" + "r" * n_cols
        lines.append(r"\begin{tabular}{" + col_format + "}")
        lines.append(r"\hline\hline")
        
        # Short labels for header
        short_labels = {
            'LQP': 'LQP', 'UQP': 'UQP', 'DOWN_ASY': 'D\\_ASY',
            'BETA': r'$\beta$', 'DOWNSIDE_BETA': r'$\beta^-$', 'UPSIDE_BETA': r'$\beta^+$',
            'SIZE': 'SIZE', 'BM': 'BM', 'TURN': 'TURN', 'ILLIQ': 'ILLIQ',
            'MOM': 'MOM', 'IVOL': 'IVOL', 'COSKEW': 'CSK', 'COKURT': 'CKU', 'MAX': 'MAX'
        }
        
        header_labels = [short_labels.get(c, c) for c in df.columns]
        header = " & " + " & ".join(header_labels) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        
        # Data rows - lower triangular
        for i, (idx, row) in enumerate(df.iterrows()):
            row_label = short_labels.get(idx, idx)
            row_data = [row_label]
            
            for j, col in enumerate(df.columns):
                if j > i:
                    row_data.append("")
                elif i == j:
                    row_data.append("1.00")
                else:
                    val = row[col]
                    row_data.append(f"{val:.2f}")
            
            lines.append(" & ".join(row_data) + r" \\")
        
        lines.append(r"\hline\hline")
        lines.append(r"\end{tabular}")
        
        # Notes
        lines.append(r"\begin{tablenotes}")
        lines.append(r"\footnotesize")
        lines.append(r"\item This table reports time-series averages of cross-sectional correlations.")
        lines.append(r"\end{tablenotes}")
        lines.append(r"\end{table}")
        
        return "\n".join(lines)
    
    def format_table_4(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 4: Summary Statistics of Asymmetry Portfolios.
        
        Paper format (Jiang, Wu, Zhou 2018):
        - Rows: Deciles 1-10 sorted by DOWN_ASY
        - Columns: 13 characteristics (DOWN_ASY, β, β−, β+, SIZE, B/M, 
                   TURN, ILLIQ, MOM, IVOL, COSKEW, COKURT, MAX)
        """
        formatted = df.copy()
        
        # Define formatting rules for each column type
        format_rules = {
            'DOWN_ASY': 3,
            'β': 2,
            'β−': 2,
            'β+': 2,
            'SIZE': 2,
            'B/M': 2,
            'TURN': 4,
            'ILLIQ': 4,
            'MOM': 3,
            'IVOL': 4,
            'COSKEW': 3,
            'COKURT': 2,
            'MAX': 4
        }
        
        # Format each column
        for col in formatted.columns:
            if col in format_rules:
                decimals = format_rules[col]
                formatted[col] = formatted[col].apply(
                    lambda x: self.format_number(x, decimals) if pd.notna(x) else ""
                )
            elif col not in ['Decile']:
                # Default formatting for any other numeric column
                formatted[col] = formatted[col].apply(
                    lambda x: self.format_number(x, 2) if pd.notna(x) else ""
                )
        
        # Generate LaTeX
        latex = self._generate_latex_table4(formatted)
        
        return formatted, latex
    
    def format_table_5(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 5: Portfolio Returns and Alphas.
        
        Paper format (2 panels):
        - Panel A: DOWN_ASY, DOWN_CORR, β−, COSKEW
        - Panel B: BBC Co-Entropy, CC Co-Entropy
        Each with Return and Alpha columns
        """
        # The new Table 5 has a complex structure with panels
        # Just pass through since replicate_table5.py handles formatting
        formatted = df.copy()
        
        # Generate LaTeX (already generated by replicate_table5.py)
        latex = self._generate_latex_table5_panels(formatted)
        
        return formatted, latex
    
    def _generate_latex_table5_panels(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 5 with 2 panels (stub - actual generated in replicate_table5.py)."""
        # The LaTeX is already generated by replicate_table5.py
        # This just returns a placeholder if needed
        return "% LaTeX already generated by replicate_table5.py"
    
    def format_table_6(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 6: Time-Series Regressions.
        
        Paper format:
        - Variable names in rows
        - Multiple model columns
        - Coefficients with t-stats below
        """
        # Get model columns
        model_cols = [c for c in df.columns if c.startswith('Model') or c == 'Variable']
        
        formatted = df[model_cols].copy()
        
        # Format each cell
        for col in formatted.columns:
            if col != 'Variable':
                formatted[col] = formatted[col].apply(
                    lambda x: self.format_number(x, 2) if pd.notna(x) else ""
                )
        
        # Generate LaTeX
        latex = self._generate_latex_table6(formatted)
        
        return formatted, latex
    
    def format_table_1(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 1: Size and Power of Entropy-Based Tests.
        
        Paper format:
        - Panel column (κ values)
        - Sample size columns (n=50, 100, 200, 500)
        - Rejection rates as decimals
        """
        formatted = df.copy()
        
        # Reset index if Panel is index
        if 'Panel' not in formatted.columns:
            formatted = formatted.reset_index()
        
        # Format numeric columns
        for col in formatted.columns:
            if col.startswith('n='):
                formatted[col] = formatted[col].apply(lambda x: self.format_number(x, 3))
        
        latex = self._generate_latex_table1(formatted)
        return formatted, latex
    
    def format_table_2(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 2: Asymmetry Tests for Portfolio Returns.
        
        Paper format:
        - Portfolio name
        - S_rho statistic
        - p-value
        """
        formatted = pd.DataFrame()
        
        # Portfolio column
        if 'Portfolio' in df.columns:
            formatted['Portfolio'] = df['Portfolio']
        elif 'portfolio' in df.columns:
            formatted['Portfolio'] = df['portfolio']
        
        # S_rho with formatting
        s_rho_col = None
        for col in ['S_rho', 's_rho', 'S_RHO', 'entropy']:
            if col in df.columns:
                s_rho_col = col
                break
        
        if s_rho_col:
            formatted['S_rho'] = df[s_rho_col].apply(lambda x: self.format_number(x, 4))
        
        # p-value with significance
        p_col = None
        for col in ['p_value', 'p-value', 'pvalue', 'p_S_rho']:
            if col in df.columns:
                p_col = col
                break
        
        if p_col:
            p_values = []
            for _, row in df.iterrows():
                p = row[p_col]
                s = row[s_rho_col] if s_rho_col else 0
                p_values.append(self.add_significance_stars(p, p_val=p))
            formatted['p-value'] = p_values
        
        latex = self._generate_latex_table2(formatted)
        return formatted, latex
    
    def format_table_7(self, df: pd.DataFrame) -> tuple:
        """
        Format Table 7: Double-Sorted Portfolios.
        
        Paper format:
        - Control variable quintiles in rows
        - DOWN_ASY quintiles in columns
        - Returns as percentages
        """
        formatted = df.copy()
        
        # Format numeric columns
        for col in formatted.columns:
            if col not in ['Control', 'control', 'Control_Quintile']:
                formatted[col] = formatted[col].apply(lambda x: self.format_number(x, 2))
        
        latex = self._generate_latex_table7(formatted)
        return formatted, latex
    
    def _generate_latex_table1(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 1."""
        cols = [c for c in df.columns if c != 'Panel']
        col_spec = "l" + "c" * len(cols)
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Size and Power of Entropy-Based Asymmetry Tests}",
            r"\label{tab:size_power}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\hline\hline",
        ]
        
        # Header
        header = "Panel & " + " & ".join(cols) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        
        # Data rows
        for _, row in df.iterrows():
            panel = row['Panel'] if 'Panel' in row else row.name
            vals = [str(row[c]) for c in cols]
            lines.append(f"{panel} & " + " & ".join(vals) + r" \\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"This table reports rejection rates at the 5\% significance level from",
            r"1,000 Monte Carlo simulations. Panel A tests size under the null (Gaussian",
            r"copula, $\kappa=1$). Panels B-F test power under alternatives with varying",
            r"degrees of asymmetry ($\kappa < 1$).",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_latex_table2(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 2."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Asymmetry Tests for Portfolio Returns}",
            r"\label{tab:asymmetry_tests}",
            r"\begin{tabular}{lcc}",
            r"\hline\hline",
            r"Portfolio & $S_{\rho}$ & $p$-value \\",
            r"\hline",
        ]
        
        for _, row in df.iterrows():
            port = row.get('Portfolio', '')
            s_rho = row.get('S_rho', '')
            p_val = row.get('p-value', '')
            lines.append(f"{port} & {s_rho} & {p_val} \\\\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"This table reports entropy-based asymmetry test statistics and",
            r"bootstrap $p$-values for 30 portfolios. $S_{\rho}$ measures the",
            r"degree of asymmetry in comovements with the market.",
            r"***, **, * denote significance at the 1\%, 5\%, and 10\% levels.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_latex_table7(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 7."""
        cols = [c for c in df.columns if c not in ['Control', 'control', 'Control_Quintile']]
        col_spec = "l" + "c" * len(cols)
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Double-Sorted Portfolios Controlling for Risk Factors}",
            r"\label{tab:double_sorts}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\hline\hline",
        ]
        
        # Header
        header = "Control & " + " & ".join(cols) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        
        # Data rows
        control_col = 'Control' if 'Control' in df.columns else 'control' if 'control' in df.columns else 'Control_Quintile'
        for _, row in df.iterrows():
            control = row.get(control_col, '')
            vals = [str(row[c]) for c in cols]
            lines.append(f"{control} & " + " & ".join(vals) + r" \\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"This table reports average returns for portfolios double-sorted",
            r"first by control variable, then by DOWN\_ASY within each control",
            r"quintile. Returns are averaged across control quintiles.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_latex_table3(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 3."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Cross-Sectional Correlations of DOWN\_ASY with Firm Characteristics}",
            r"\label{tab:correlations}",
            r"\begin{tabular}{lcc}",
            r"\hline\hline",
            r"Characteristic & Correlation & $t$-statistic \\",
            r"\hline",
        ]
        
        for _, row in df.iterrows():
            char = row['Characteristic'].replace('_', r'\_')
            corr = row['Correlation with DOWN_ASY']
            t = row['t-statistic']
            lines.append(f"{char} & {corr} & {t} \\\\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"This table reports time-series averages of monthly cross-sectional",
            r"correlations between DOWN\_ASY and firm characteristics.",
            r"$t$-statistics are computed using Newey-West standard errors.",
            r"***, **, * denote significance at the 1\%, 5\%, and 10\% levels.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_latex_table4(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 4: Summary Statistics of Asymmetry Portfolios."""
        # Get all columns except index
        cols = list(df.columns)
        n_cols = len(cols)
        col_spec = "l" + "r" * n_cols
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Summary Statistics of Asymmetry Portfolios}",
            r"\label{tab:summary_stats}",
            r"\small",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\hline\hline",
        ]
        
        # Header
        header = "Decile & " + " & ".join(cols) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        
        # Data rows (use index as decile number)
        for idx, row in df.iterrows():
            row_vals = [str(idx)] + [str(row[c]) for c in cols]
            lines.append(" & ".join(row_vals) + r" \\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"\item This table reports various risk measures and firm characteristics of deciles sorted based on their realized DOWN\_ASY each month.",
            r"\item Variables: DOWN\_ASY = downside asymmetric comovement; $\beta$ = CAPM beta; $\beta^-$ = downside beta; $\beta^+$ = upside beta;",
            r"\item SIZE = log market capitalization; B/M = log book-to-market; TURN = turnover; ILLIQ = Amihud illiquidity;",
            r"\item MOM = past 6-month return; IVOL = idiosyncratic volatility; COSKEW = coskewness; COKURT = cokurtosis; MAX = maximum daily return.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
        
        return "\n".join(lines)
    
    def _generate_latex_table5(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 5."""
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Portfolio Returns and Alphas Sorted by DOWN\_ASY}",
            r"\label{tab:returns_alphas}",
            r"\begin{tabular}{lccc}",
            r"\hline\hline",
            r"Portfolio & Return (\%) & Alpha (\%) & $[t$-stat$]$ \\",
            r"\hline",
        ]
        
        for _, row in df.iterrows():
            port = row['Portfolio']
            ret = row['Return (%)']
            alpha = row['Alpha (%)']
            t = row['[t-stat]']
            lines.append(f"{port} & {ret} & {alpha} & {t} \\\\")
            
            # Add spacing before High-Low row
            if 'Q5' in str(port):
                lines.append(r"\hline")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"This table reports average monthly returns and Carhart four-factor",
            r"alphas for portfolios sorted by DOWN\_ASY. High-Low is the difference",
            r"between the highest and lowest quintile portfolios.",
            r"$t$-statistics are computed using Newey-West standard errors (12 lags).",
            r"***, **, * denote significance at the 1\%, 5\%, and 10\% levels.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def _generate_latex_table6(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 6."""
        model_cols = [c for c in df.columns if c.startswith('Model')]
        n_models = len(model_cols)
        col_spec = "l" + "c" * n_models
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Time-Series Determinants of the Asymmetry Premium}",
            r"\label{tab:time_series}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\hline\hline",
        ]
        
        # Header
        header = "Variable & " + " & ".join([f"({i+1})" for i in range(n_models)]) + r" \\"
        lines.append(header)
        lines.append(r"\hline")
        
        # Data rows
        for _, row in df.iterrows():
            var = row['Variable'].replace('_', ' ')
            vals = [str(row[c]) if row[c] != "" else "" for c in model_cols]
            lines.append(f"{var} & " + " & ".join(vals) + r" \\")
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            r"This table reports time-series regressions of the High-Low asymmetry",
            r"spread on market conditions. Market Volatility is the monthly variance",
            r"of daily market returns. Newey-West standard errors with 12 lags.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
    
    def generate_text_table(self, df: pd.DataFrame, title: str = "") -> str:
        """Generate nicely formatted text table for console output."""
        lines = []
        
        if title:
            lines.append("=" * 70)
            lines.append(title.center(70))
            lines.append("=" * 70)
        
        # Calculate column widths
        col_widths = {}
        for col in df.columns:
            max_width = len(str(col))
            for val in df[col]:
                max_width = max(max_width, len(str(val)))
            col_widths[col] = max_width + 2
        
        # Header
        header = ""
        for col in df.columns:
            header += str(col).center(col_widths[col])
        lines.append(header)
        lines.append("-" * len(header))
        
        # Data rows
        for _, row in df.iterrows():
            row_str = ""
            for col in df.columns:
                row_str += str(row[col]).center(col_widths[col])
            lines.append(row_str)
        
        lines.append("-" * len(header))
        
        return "\n".join(lines)
    
    def format_all_tables(self):
        """Format all tables and save in multiple formats."""
        print("=" * 70)
        print("  Formatting Tables to Match Paper Style")
        print("  Jiang, Wu, and Zhou (2018) - JFQA")
        print("=" * 70)
        
        results = {}
        
        # Table 1: Size and Power (if exists)
        table1_path = self.tables_dir / 'Table_1_Size_Power.csv'
        if table1_path.exists():
            print("\nFormatting Table 1: Size and Power Tests...")
            df = pd.read_csv(table1_path)
            formatted, latex = self.format_table_1(df)
            
            with open(self.tables_dir / 'Table_1_Size_Power.tex', 'w', encoding='utf-8') as f:
                f.write(latex)
            
            print(self.generate_text_table(formatted, "Table 1: Size and Power of Entropy Tests"))
            results['table1'] = True
        
        # Table 2: Asymmetry Tests (if exists)
        table2_path = self.tables_dir / 'Table_2_Asymmetry_Tests.csv'
        if table2_path.exists():
            print("\nFormatting Table 2: Asymmetry Tests...")
            df = pd.read_csv(table2_path)
            formatted, latex = self.format_table_2(df)
            
            with open(self.tables_dir / 'Table_2_Asymmetry_Tests.tex', 'w', encoding='utf-8') as f:
                f.write(latex)
            
            print(self.generate_text_table(formatted, "Table 2: Asymmetry Tests for Portfolios"))
            results['table2'] = True
        
        # Table 3: Correlations
        table3_path = self.tables_dir / 'Table_3_Correlations.csv'
        if table3_path.exists():
            print("\nFormatting Table 3: Cross-Sectional Correlations...")
            # Check if it's a correlation matrix (has index column)
            df = pd.read_csv(table3_path, index_col=0)
            formatted, latex = self.format_table_3(df)
            
            # Save LaTeX
            with open(self.tables_dir / 'Table_3_Correlations.tex', 'w', encoding='utf-8') as f:
                f.write(latex)
            
            # Print correlation matrix summary instead of full table
            print("  Table 3 formatted: 15x15 correlation matrix")
            results['table3'] = True
        
        # Table 4: Summary Stats of Asymmetry Portfolios
        table4_path = self.tables_dir / 'Table_4_Summary_Stats.csv'
        if table4_path.exists():
            print("\nFormatting Table 4: Summary Statistics of Asymmetry Portfolios...")
            # Read with index (Decile column)
            df = pd.read_csv(table4_path, index_col=0)
            formatted, latex = self.format_table_4(df)
            
            with open(self.tables_dir / 'Table_4_Summary_Stats.tex', 'w', encoding='utf-8') as f:
                f.write(latex)
            
            # Print a summary of columns
            print(f"  Table 4 formatted: 10 deciles x {len(df.columns)} characteristics")
            print(f"  Characteristics: {', '.join(df.columns[:5])}...")
            results['table4'] = True
        
        # Table 5: Returns and Alphas (2 panels - already formatted by replicate_table5.py)
        table5_path = self.tables_dir / 'Table_5_Returns_Alphas.csv'
        table5_tex = self.tables_dir / 'Table_5_Returns_Alphas.tex'
        if table5_path.exists():
            print("\nFormatting Table 5: Portfolio Returns and Alphas...")
            df = pd.read_csv(table5_path)
            
            # Count measures from column names
            ret_cols = [c for c in df.columns if c.endswith('_Return')]
            n_measures = len(ret_cols)
            
            # Display panel info
            print(self._format_table5_display(df))
            results['table5'] = True
    
    def _format_table5_display(self, df: pd.DataFrame) -> str:
        """Generate display text for Table 5."""
        lines = [
            "=" * 70,
            "                Table 5: Portfolio Returns and Alphas",
            "=" * 70
        ]
        
        # Find panel markers
        in_panel_a = True
        for _, row in df.iterrows():
            port = str(row.get('Portfolio', ''))
            
            if 'Panel A' in port:
                lines.append("\nPanel A: Downside Asymmetry and Conventional Measures")
                lines.append("-" * 60)
                continue
            elif 'Panel B' in port:
                lines.append("\nPanel B: Alternative Entropy Measures")
                lines.append("-" * 60)
                continue
            elif port == '':
                continue
            
            # Format row - skip if empty portfolio or nan
            port_str = str(port).strip()
            if port_str in ['', 'nan', 'NaN', 'None']:
                continue
            
            row_str = f"  {port_str:12s}"
            for col in df.columns:
                if col != 'Portfolio' and pd.notna(row.get(col, '')):
                    val = str(row[col])
                    if val and val != 'nan':
                        row_str += f"  {val:>10s}"
            lines.append(row_str)
        
        lines.append("-" * 60)
        return "\n".join(lines)
        
        # Note: Table 6 and 7 are now generated directly by replicate_table6.py and replicate_table7.py
        # with proper LaTeX output. No additional formatting needed here.
        
        # Table 7: Double-Sorted Portfolios (if exists)
        table7_path = self.tables_dir / 'Table_7_Double_Sorts.csv'
        if table7_path.exists():
            print("\nFormatting Table 7: Double-Sorted Portfolios...")
            df = pd.read_csv(table7_path)
            formatted, latex = self.format_table_7(df)
            
            with open(self.tables_dir / 'Table_7_Double_Sorts.tex', 'w', encoding='utf-8') as f:
                f.write(latex)
            
            print(self.generate_text_table(formatted, "Table 7: Double-Sorted Portfolios"))
            results['table7'] = True
        
        print("\n" + "=" * 70)
        print("  Table Formatting Complete!")
        print("=" * 70)
        print(f"\nOutput directory: {self.tables_dir}")
        print("\nFiles created:")
        print("  LaTeX files:")
        for table in ['1', '2', '3', '4', '5', '6', '7']:
            if results.get(f'table{table}'):
                print(f"    - Table_{table}_*.tex")
        
        return results


def main():
    """Main entry point."""
    tables_dir = Path(__file__).parent.parent / 'outputs' / 'tables'
    formatter = PaperTableFormatter(str(tables_dir))
    formatter.format_all_tables()


if __name__ == '__main__':
    main()
