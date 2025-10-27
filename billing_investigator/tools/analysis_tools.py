# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analysis tools for detecting usage patterns, anomalies, and correlations.

This module provides statistical analysis functions for:
- Period-over-period comparisons
- Appliance-level anomaly detection
- Time-of-use pattern analysis
- Weather correlation analysis
- Confidence scoring
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

# Note: Using numpy for statistics instead of scipy for lighter dependencies

# Configure logging
logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    SUDDEN_INCREASE = "sudden_increase"
    SUDDEN_DECREASE = "sudden_decrease"
    GRADUAL_INCREASE = "gradual_increase"
    PATTERN_SHIFT = "pattern_shift"
    OUTLIER = "outlier"


@dataclass
class ThresholdConfig:
    """Configuration for anomaly detection thresholds.

    Attributes:
        z_score_threshold: Number of standard deviations for z-score method
        iqr_multiplier: Multiplier for IQR method (typically 1.5 or 3.0)
        min_change_percent: Minimum percentage change to consider significant
        min_samples: Minimum number of samples required for reliable analysis
        seasonal_window_days: Days to use for seasonal baseline
        appliance_sensitivity: Sensitivity multipliers per appliance type
    """
    z_score_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    min_change_percent: float = 10.0
    min_samples: int = 30
    seasonal_window_days: int = 30
    appliance_sensitivity: Dict[str, float] = field(default_factory=lambda: {
        "hvac_heating_kwh": 1.0,
        "hvac_cooling_kwh": 1.0,
        "water_heater_kwh": 1.2,
        "ev_charging_kwh": 0.8,  # More tolerance for EV variation
        "pool_equipment_kwh": 1.0,
        "major_appliances_kwh": 1.3,
        "other_loads_kwh": 1.5,
    })

    def validate(self) -> None:
        """Validate threshold values are in reasonable ranges."""
        if self.z_score_threshold < 1.0 or self.z_score_threshold > 5.0:
            raise ValueError("z_score_threshold should be between 1.0 and 5.0")
        if self.iqr_multiplier < 0.5 or self.iqr_multiplier > 5.0:
            raise ValueError("iqr_multiplier should be between 0.5 and 5.0")
        if self.min_change_percent < 0 or self.min_change_percent > 100:
            raise ValueError("min_change_percent should be between 0 and 100")
        if self.min_samples < 5:
            raise ValueError("min_samples should be at least 5")


# Statistical Utilities (6.6)

def moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate moving average with specified window size.

    Args:
        data: Time series data
        window: Window size for moving average

    Returns:
        Series with moving average values
    """
    return data.rolling(window=window, min_periods=1).mean()


def detect_outliers_zscore(
    data: pd.Series,
    threshold: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Detect outliers using z-score method.

    Args:
        data: Data to analyze
        threshold: Z-score threshold (default 3.0)

    Returns:
        Tuple of (outlier_mask, z_scores)
    """
    mean = data.mean()
    std = data.std()

    if std == 0:
        return pd.Series([False] * len(data), index=data.index), pd.Series([0.0] * len(data), index=data.index)

    z_scores = np.abs((data - mean) / std)
    outliers = z_scores > threshold

    return outliers, z_scores


def detect_outliers_iqr(
    data: pd.Series,
    multiplier: float = 1.5
) -> Tuple[pd.Series, Tuple[float, float]]:
    """Detect outliers using IQR method.

    Args:
        data: Data to analyze
        multiplier: IQR multiplier (default 1.5)

    Returns:
        Tuple of (outlier_mask, (lower_bound, upper_bound))
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers, (lower_bound, upper_bound)


def seasonal_decomposition_simple(
    data: pd.Series,
    period: int = 24
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Simple seasonal decomposition into trend, seasonal, and residual.

    Args:
        data: Time series data
        period: Seasonal period (e.g., 24 for hourly with daily pattern)

    Returns:
        Tuple of (trend, seasonal, residual)
    """
    # Trend: moving average
    trend = data.rolling(window=period, center=True).mean()

    # Detrend
    detrended = data - trend

    # Seasonal: average by period
    seasonal = detrended.groupby(data.index % period).transform('mean')

    # Residual
    residual = data - trend - seasonal

    return trend, seasonal, residual


# Period Comparison (6.1)

@dataclass
class PeriodComparison:
    """Results of period-over-period comparison.

    Attributes:
        current_total: Total usage in current period
        baseline_total: Total usage in baseline period
        absolute_change: Absolute difference in kWh
        percent_change: Percentage change
        current_avg_daily: Average daily usage in current period
        baseline_avg_daily: Average daily usage in baseline period
        current_peak: Peak usage in current period
        baseline_peak: Peak usage in baseline period
        appliance_changes: Per-appliance changes
    """
    current_total: float
    baseline_total: float
    absolute_change: float
    percent_change: float
    current_avg_daily: float
    baseline_avg_daily: float
    current_peak: float
    baseline_peak: float
    appliance_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)


def calculate_period_comparison(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    total_column: str = "total_usage_kwh",
    appliance_columns: Optional[List[str]] = None
) -> PeriodComparison:
    """Compare usage between current and baseline periods.

    Args:
        current_df: DataFrame with current period data
        baseline_df: DataFrame with baseline period data
        total_column: Column name for total usage
        appliance_columns: List of appliance column names to analyze

    Returns:
        PeriodComparison object with detailed metrics
    """
    logger.info(f"Comparing periods: {len(current_df)} vs {len(baseline_df)} records")

    if appliance_columns is None:
        appliance_columns = [
            "hvac_heating_kwh", "hvac_cooling_kwh", "water_heater_kwh",
            "ev_charging_kwh", "pool_equipment_kwh", "major_appliances_kwh",
            "other_loads_kwh"
        ]

    # Total usage
    current_total = current_df[total_column].sum()
    baseline_total = baseline_df[total_column].sum()
    absolute_change = current_total - baseline_total
    percent_change = (absolute_change / baseline_total * 100) if baseline_total > 0 else 0

    # Daily averages
    current_days = (current_df['timestamp'].max() - current_df['timestamp'].min()).days + 1
    baseline_days = (baseline_df['timestamp'].max() - baseline_df['timestamp'].min()).days + 1

    current_avg_daily = current_total / current_days if current_days > 0 else 0
    baseline_avg_daily = baseline_total / baseline_days if baseline_days > 0 else 0

    # Peak usage
    current_peak = current_df[total_column].max()
    baseline_peak = baseline_df[total_column].max()

    # Appliance-level changes
    appliance_changes = {}
    for col in appliance_columns:
        if col in current_df.columns and col in baseline_df.columns:
            curr_sum = current_df[col].sum()
            base_sum = baseline_df[col].sum()
            change = curr_sum - base_sum
            pct_change = (change / base_sum * 100) if base_sum > 0 else 0

            appliance_changes[col] = {
                "current": curr_sum,
                "baseline": base_sum,
                "absolute_change": change,
                "percent_change": pct_change
            }

    logger.info(
        f"Total change: {absolute_change:.1f} kWh ({percent_change:+.1f}%), "
        f"Peak: {current_peak:.1f} vs {baseline_peak:.1f}"
    )

    return PeriodComparison(
        current_total=current_total,
        baseline_total=baseline_total,
        absolute_change=absolute_change,
        percent_change=percent_change,
        current_avg_daily=current_avg_daily,
        baseline_avg_daily=baseline_avg_daily,
        current_peak=current_peak,
        baseline_peak=baseline_peak,
        appliance_changes=appliance_changes
    )


# Appliance Anomaly Detection (6.2)

@dataclass
class ApplianceAnomaly:
    """Detected appliance anomaly.

    Attributes:
        appliance: Appliance column name
        anomaly_type: Type of anomaly detected
        timestamp_start: When anomaly started (if identifiable)
        timestamp_end: When anomaly ended (if identifiable)
        severity_score: Severity from 0-100
        baseline_avg: Baseline average usage
        anomaly_avg: Average usage during anomaly
        confidence: Confidence score 0-1
        description: Natural language description
    """
    appliance: str
    anomaly_type: AnomalyType
    timestamp_start: Optional[pd.Timestamp] = None
    timestamp_end: Optional[pd.Timestamp] = None
    severity_score: float = 0.0
    baseline_avg: float = 0.0
    anomaly_avg: float = 0.0
    confidence: float = 0.0
    description: str = ""


def detect_appliance_anomalies(
    df: pd.DataFrame,
    appliance_columns: Optional[List[str]] = None,
    config: Optional[ThresholdConfig] = None
) -> List[ApplianceAnomaly]:
    """Detect anomalies in appliance-level consumption.

    Args:
        df: DataFrame with usage data (must have timestamp and appliance columns)
        appliance_columns: List of appliance columns to analyze
        config: Threshold configuration

    Returns:
        List of detected anomalies
    """
    if config is None:
        config = ThresholdConfig()
    config.validate()

    if appliance_columns is None:
        appliance_columns = [
            "hvac_heating_kwh", "hvac_cooling_kwh", "water_heater_kwh",
            "ev_charging_kwh", "pool_equipment_kwh", "major_appliances_kwh"
        ]

    anomalies = []

    logger.info(f"Detecting anomalies across {len(appliance_columns)} appliances")

    for col in appliance_columns:
        if col not in df.columns:
            continue

        data = df[col].copy()

        if len(data) < config.min_samples:
            logger.warning(f"Insufficient samples for {col}: {len(data)} < {config.min_samples}")
            continue

        # Get appliance-specific sensitivity
        sensitivity = config.appliance_sensitivity.get(col, 1.0)
        adjusted_threshold = config.z_score_threshold / sensitivity

        # Z-score based detection
        outliers_z, z_scores = detect_outliers_zscore(data, adjusted_threshold)

        if outliers_z.sum() > 0:
            # Find contiguous outlier periods
            outlier_groups = (outliers_z != outliers_z.shift()).cumsum()

            for group_id in outlier_groups[outliers_z].unique():
                group_mask = (outlier_groups == group_id) & outliers_z
                group_data = data[group_mask]

                if len(group_data) < 3:  # Ignore single-point outliers
                    continue

                # Calculate baseline (non-outlier) average
                baseline_avg = data[~outliers_z].mean()
                anomaly_avg = group_data.mean()

                # Determine anomaly type and severity
                change_pct = ((anomaly_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0

                if abs(change_pct) < config.min_change_percent:
                    continue

                if change_pct > 0:
                    anomaly_type = AnomalyType.SUDDEN_INCREASE
                else:
                    anomaly_type = AnomalyType.SUDDEN_DECREASE

                severity = min(abs(change_pct), 100)
                confidence = min(z_scores[group_mask].mean() / 5.0, 1.0)

                anomaly = ApplianceAnomaly(
                    appliance=col,
                    anomaly_type=anomaly_type,
                    timestamp_start=df.loc[group_mask, 'timestamp'].min() if 'timestamp' in df.columns else None,
                    timestamp_end=df.loc[group_mask, 'timestamp'].max() if 'timestamp' in df.columns else None,
                    severity_score=severity,
                    baseline_avg=baseline_avg,
                    anomaly_avg=anomaly_avg,
                    confidence=confidence,
                    description=f"{col.replace('_kwh', '').replace('_', ' ').title()} usage {anomaly_type.value}: {change_pct:+.1f}% ({anomaly_avg:.1f} vs {baseline_avg:.1f} kWh)"
                )

                anomalies.append(anomaly)
                logger.info(f"Detected anomaly: {anomaly.description}")

    return anomalies


# Time Pattern Analysis (6.3)

@dataclass
class TimePatternShift:
    """Detected shift in time-of-use patterns.

    Attributes:
        shift_type: Description of shift type
        hours_affected: List of hours where pattern changed
        magnitude: Magnitude of shift (kWh or percentage)
        confidence: Confidence score 0-1
        description: Natural language description
    """
    shift_type: str
    hours_affected: List[int]
    magnitude: float
    confidence: float
    description: str


def analyze_time_patterns(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    usage_column: str = "total_usage_kwh",
    threshold_pct: float = 20.0
) -> List[TimePatternShift]:
    """Detect changes in time-of-use patterns.

    Args:
        current_df: Current period data
        baseline_df: Baseline period data
        usage_column: Column to analyze
        threshold_pct: Minimum percentage change to report

    Returns:
        List of detected pattern shifts
    """
    logger.info("Analyzing time-of-use pattern shifts")

    # Extract hour from timestamp
    current_df = current_df.copy()
    baseline_df = baseline_df.copy()

    current_df['hour'] = pd.to_datetime(current_df['timestamp']).dt.hour
    baseline_df['hour'] = pd.to_datetime(baseline_df['timestamp']).dt.hour

    # Calculate hourly averages
    current_hourly = current_df.groupby('hour')[usage_column].mean()
    baseline_hourly = baseline_df.groupby('hour')[usage_column].mean()

    # Find hours with significant changes
    changes = {}
    for hour in range(24):
        if hour in current_hourly.index and hour in baseline_hourly.index:
            curr = current_hourly[hour]
            base = baseline_hourly[hour]
            pct_change = ((curr - base) / base * 100) if base > 0 else 0

            if abs(pct_change) >= threshold_pct:
                changes[hour] = {
                    "current": curr,
                    "baseline": base,
                    "percent_change": pct_change
                }

    shifts = []

    if not changes:
        logger.info("No significant time pattern shifts detected")
        return shifts

    # Identify peak shifts
    current_peak_hour = current_hourly.idxmax()
    baseline_peak_hour = baseline_hourly.idxmax()

    if current_peak_hour != baseline_peak_hour:
        shift = TimePatternShift(
            shift_type="peak_hour_shift",
            hours_affected=[baseline_peak_hour, current_peak_hour],
            magnitude=current_hourly[current_peak_hour] - baseline_hourly[baseline_peak_hour],
            confidence=0.8,
            description=f"Peak usage hour shifted from {baseline_peak_hour}:00 to {current_peak_hour}:00"
        )
        shifts.append(shift)
        logger.info(shift.description)

    # Group consecutive hours with increases
    increase_hours = [h for h, c in changes.items() if c["percent_change"] > 0]
    if len(increase_hours) >= 3:
        shift = TimePatternShift(
            shift_type="usage_increase_period",
            hours_affected=increase_hours,
            magnitude=sum(changes[h]["current"] - changes[h]["baseline"] for h in increase_hours),
            confidence=0.7,
            description=f"Increased usage during hours {min(increase_hours)}-{max(increase_hours)}"
        )
        shifts.append(shift)

    return shifts


# Weather Correlation (6.4)

@dataclass
class WeatherCorrelation:
    """Results of weather correlation analysis.

    Attributes:
        correlation_coefficient: Pearson correlation coefficient (-1 to 1)
        p_value: Statistical significance
        temperature_sensitivity: kWh change per degree
        r_squared: Coefficient of determination
        description: Natural language description
    """
    correlation_coefficient: float
    p_value: float
    temperature_sensitivity: float
    r_squared: float
    description: str


def correlate_with_weather(
    df: pd.DataFrame,
    usage_column: str = "hvac_cooling_kwh",
    temp_column: str = "outdoor_temperature_f",
    degree_hours_column: Optional[str] = "cooling_degree_hours"
) -> WeatherCorrelation:
    """Analyze correlation between energy usage and temperature.

    Args:
        df: DataFrame with usage and temperature data
        usage_column: Column name for usage metric
        temp_column: Column name for temperature
        degree_hours_column: Optional degree-hours column for better correlation

    Returns:
        WeatherCorrelation results
    """
    logger.info(f"Analyzing correlation between {usage_column} and {temp_column}")

    # Use degree-hours if available, otherwise temperature
    if degree_hours_column and degree_hours_column in df.columns:
        x = df[degree_hours_column].values
        x_label = degree_hours_column
    else:
        x = df[temp_column].values
        x_label = temp_column

    y = df[usage_column].values

    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        logger.warning(f"Insufficient data for correlation analysis: {len(x)} points")
        return WeatherCorrelation(
            correlation_coefficient=0.0,
            p_value=1.0,
            temperature_sensitivity=0.0,
            r_squared=0.0,
            description="Insufficient data for correlation"
        )

    # Calculate Pearson correlation using numpy
    corr_coef = np.corrcoef(x, y)[0, 1]

    # Calculate p-value approximation for correlation
    # Using t-statistic: t = r * sqrt(n-2) / sqrt(1-r^2)
    n = len(x)
    if abs(corr_coef) < 1.0:
        t_stat = corr_coef * np.sqrt(n - 2) / np.sqrt(1 - corr_coef**2)
        # Simplified p-value estimation (two-tailed)
        p_value = 2 * (1 - 0.5 * (1 + np.tanh(t_stat / np.sqrt(n))))
    else:
        p_value = 0.0

    # Linear regression for sensitivity using numpy polyfit
    slope, intercept = np.polyfit(x, y, 1)

    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Determine strength and direction
    if abs(corr_coef) >= 0.7:
        strength = "strong"
    elif abs(corr_coef) >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if corr_coef > 0 else "negative"

    description = (
        f"{strength.title()} {direction} correlation (r={corr_coef:.2f}, p={p_value:.4f}): "
        f"{slope:.3f} kWh per unit {x_label}"
    )

    logger.info(description)

    return WeatherCorrelation(
        correlation_coefficient=corr_coef,
        p_value=p_value,
        temperature_sensitivity=slope,
        r_squared=r_squared,
        description=description
    )


# Confidence Scoring (6.5)

def calculate_confidence_score(
    sample_size: int,
    data_completeness: float,
    statistical_significance: float,
    consistency_score: float = 1.0
) -> Tuple[float, str]:
    """Calculate confidence score for analysis findings.

    Args:
        sample_size: Number of data points
        data_completeness: Fraction of complete data (0-1)
        statistical_significance: P-value or similar (lower is better)
        consistency_score: Consistency across multiple methods (0-1)

    Returns:
        Tuple of (confidence_score 0-1, justification_text)
    """
    # Sample size factor (sigmoid)
    size_factor = 1 / (1 + np.exp(-0.05 * (sample_size - 50)))

    # Data completeness factor (linear)
    completeness_factor = data_completeness

    # Statistical significance factor (inverse of p-value, capped)
    sig_factor = max(0, 1 - min(statistical_significance, 1.0))

    # Weighted combination
    confidence = (
        0.3 * size_factor +
        0.2 * completeness_factor +
        0.3 * sig_factor +
        0.2 * consistency_score
    )

    # Generate justification
    reasons = []
    if sample_size < 30:
        reasons.append("limited sample size")
    if data_completeness < 0.8:
        reasons.append("incomplete data")
    if statistical_significance > 0.05:
        reasons.append("low statistical significance")
    if consistency_score < 0.7:
        reasons.append("inconsistent across methods")

    if confidence >= 0.8:
        level = "High confidence"
    elif confidence >= 0.6:
        level = "Moderate confidence"
    else:
        level = "Low confidence"

    if reasons:
        justification = f"{level}: {', '.join(reasons)}"
    else:
        justification = f"{level}: sufficient data and strong statistical support"

    return confidence, justification


if __name__ == "__main__":
    # Test the analysis tools
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Analysis Tools Test")
    print("=" * 70)
    print()

    # Create sample data
    dates = pd.date_range("2024-07-01", periods=720, freq="H")

    np.random.seed(42)
    baseline_df = pd.DataFrame({
        "timestamp": dates[:360],
        "total_usage_kwh": np.random.normal(5, 1, 360),
        "hvac_cooling_kwh": np.random.normal(2, 0.5, 360),
        "outdoor_temperature_f": np.random.normal(75, 5, 360),
        "cooling_degree_hours": np.random.normal(10, 3, 360)
    })

    # Current period with anomaly
    current_df = pd.DataFrame({
        "timestamp": dates[360:],
        "total_usage_kwh": np.random.normal(6.5, 1.2, 360),  # Higher usage
        "hvac_cooling_kwh": np.random.normal(3.2, 0.6, 360),  # HVAC increase
        "outdoor_temperature_f": np.random.normal(78, 5, 360),
        "cooling_degree_hours": np.random.normal(12, 3, 360)
    })

    # Test 1: Period Comparison
    print("Test 1: Period Comparison")
    print("-" * 70)
    comparison = calculate_period_comparison(
        current_df,
        baseline_df,
        appliance_columns=["hvac_cooling_kwh"]
    )
    print(f"Total change: {comparison.absolute_change:.1f} kWh ({comparison.percent_change:+.1f}%)")
    print(f"Daily average: {comparison.current_avg_daily:.1f} vs {comparison.baseline_avg_daily:.1f}")
    print()

    # Test 2: Weather Correlation
    print("Test 2: Weather Correlation")
    print("-" * 70)
    correlation = correlate_with_weather(current_df)
    print(correlation.description)
    print()

    # Test 3: Confidence Scoring
    print("Test 3: Confidence Scoring")
    print("-" * 70)
    score, justification = calculate_confidence_score(
        sample_size=360,
        data_completeness=0.95,
        statistical_significance=0.01,
        consistency_score=0.85
    )
    print(f"Confidence: {score:.2f} - {justification}")
    print()

    print("=" * 70)
    print("✓ Analysis tools tested successfully!")
    print("=" * 70)
