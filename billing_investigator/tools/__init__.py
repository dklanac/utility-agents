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

"""Tools module for Bill Investigator agent."""

from .bigquery_tools import (
    get_database_settings,
    get_customer_usage_data,
    query_usage_patterns,
    serialize_dataframe,
    deserialize_dataframe,
)

from .analysis_tools import (
    ThresholdConfig,
    AnomalyType,
    PeriodComparison,
    ApplianceAnomaly,
    TimePatternShift,
    WeatherCorrelation,
    calculate_period_comparison,
    detect_appliance_anomalies,
    analyze_time_patterns,
    correlate_with_weather,
    calculate_confidence_score,
    moving_average,
    detect_outliers_zscore,
    detect_outliers_iqr,
)

__all__ = [
    # BigQuery tools
    "get_database_settings",
    "get_customer_usage_data",
    "query_usage_patterns",
    "serialize_dataframe",
    "deserialize_dataframe",
    # Analysis tools
    "ThresholdConfig",
    "AnomalyType",
    "PeriodComparison",
    "ApplianceAnomaly",
    "TimePatternShift",
    "WeatherCorrelation",
    "calculate_period_comparison",
    "detect_appliance_anomalies",
    "analyze_time_patterns",
    "correlate_with_weather",
    "calculate_confidence_score",
    "moving_average",
    "detect_outliers_zscore",
    "detect_outliers_iqr",
]
