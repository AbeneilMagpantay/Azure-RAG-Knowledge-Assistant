"""A/B Test analysis with statistical significance."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ABTestResult:
    """Result of an A/B test analysis."""
    control_conversion: float
    treatment_conversion: float
    relative_uplift: float
    p_value: float
    is_significant: bool
    confidence_level: float
    sample_size_control: int
    sample_size_treatment: int
    recommendation: str


class ABTestAnalyzer:
    """Analyze A/B test results with statistical significance."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analyzer.
        
        Args:
            confidence_level: Confidence level for significance (default 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze(
        self,
        control_conversions: int,
        control_total: int,
        treatment_conversions: int,
        treatment_total: int
    ) -> ABTestResult:
        """
        Analyze A/B test results.
        
        Args:
            control_conversions: Number of conversions in control group (A)
            control_total: Total samples in control group
            treatment_conversions: Number of conversions in treatment group (B)
            treatment_total: Total samples in treatment group
            
        Returns:
            ABTestResult with analysis
        """
        # Calculate conversion rates
        control_rate = control_conversions / control_total if control_total > 0 else 0
        treatment_rate = treatment_conversions / treatment_total if treatment_total > 0 else 0
        
        # Calculate relative uplift
        if control_rate > 0:
            relative_uplift = ((treatment_rate - control_rate) / control_rate) * 100
        else:
            relative_uplift = 0
        
        # Perform chi-square test for significance
        contingency_table = [
            [control_conversions, control_total - control_conversions],
            [treatment_conversions, treatment_total - treatment_conversions]
        ]
        
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        except ValueError:
            p_value = 1.0
        
        is_significant = p_value < self.alpha
        
        # Generate recommendation
        if not is_significant:
            recommendation = "No significant difference detected. Consider running the test longer or with more samples."
        elif treatment_rate > control_rate:
            recommendation = f"Treatment (B) is the winner with {relative_uplift:.1f}% uplift. Recommend implementing treatment."
        else:
            recommendation = f"Control (A) performs better. Treatment shows {abs(relative_uplift):.1f}% decrease."
        
        return ABTestResult(
            control_conversion=round(control_rate * 100, 2),
            treatment_conversion=round(treatment_rate * 100, 2),
            relative_uplift=round(relative_uplift, 2),
            p_value=round(p_value, 4),
            is_significant=is_significant,
            confidence_level=self.confidence_level * 100,
            sample_size_control=control_total,
            sample_size_treatment=treatment_total,
            recommendation=recommendation
        )
    
    def analyze_from_dataframe(
        self,
        df: pd.DataFrame,
        variant_col: str = "variant",
        conversion_col: str = "converted",
        control_label: str = "A",
        treatment_label: str = "B"
    ) -> ABTestResult:
        """
        Analyze A/B test from a DataFrame.
        
        Args:
            df: DataFrame with test data
            variant_col: Column name for variant (A/B)
            conversion_col: Column name for conversion (1/0 or True/False)
            control_label: Label for control group
            treatment_label: Label for treatment group
            
        Returns:
            ABTestResult with analysis
        """
        control = df[df[variant_col] == control_label]
        treatment = df[df[variant_col] == treatment_label]
        
        control_conversions = control[conversion_col].sum()
        control_total = len(control)
        treatment_conversions = treatment[conversion_col].sum()
        treatment_total = len(treatment)
        
        return self.analyze(
            control_conversions=int(control_conversions),
            control_total=control_total,
            treatment_conversions=int(treatment_conversions),
            treatment_total=treatment_total
        )
    
    def analyze_revenue_test(
        self,
        control_revenue: float,
        control_total: int,
        treatment_revenue: float,
        treatment_total: int
    ) -> Dict[str, Any]:
        """
        Analyze A/B test for revenue metrics.
        
        Args:
            control_revenue: Total revenue from control group
            control_total: Number of users in control
            treatment_revenue: Total revenue from treatment group
            treatment_total: Number of users in treatment
            
        Returns:
            Dict with revenue analysis
        """
        control_avg = control_revenue / control_total if control_total > 0 else 0
        treatment_avg = treatment_revenue / treatment_total if treatment_total > 0 else 0
        
        if control_avg > 0:
            uplift = ((treatment_avg - control_avg) / control_avg) * 100
        else:
            uplift = 0
        
        return {
            "control_avg_revenue": round(control_avg, 2),
            "treatment_avg_revenue": round(treatment_avg, 2),
            "revenue_uplift_percent": round(uplift, 2),
            "control_total_revenue": round(control_revenue, 2),
            "treatment_total_revenue": round(treatment_revenue, 2),
            "winner": "Treatment (B)" if treatment_avg > control_avg else "Control (A)"
        }
