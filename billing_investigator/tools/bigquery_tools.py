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

"""BigQuery tools for querying customer utility usage data."""

import datetime
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from google.adk.tools import ToolContext
from google.adk.tools.bigquery.client import get_bigquery_client
from google.cloud import bigquery
from google.cloud.exceptions import NotFound

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# BigQuery configuration from environment variables
BIGQUERY_PROJECT_ID = os.getenv("BIGQUERY_PROJECT_ID", "gen-ai-416617")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "utility_usage")
BIGQUERY_TABLE = os.getenv("BIGQUERY_TABLE", "energy_consumption")

# Fully qualified table name
FULL_TABLE_NAME = f"{BIGQUERY_PROJECT_ID}.{BIGQUERY_DATASET}.{BIGQUERY_TABLE}"

# Cache for database settings
_database_settings_cache: Optional[Dict] = None


def _serialize_value_for_sql(value):
    """Serializes a Python value from a pandas DataFrame into a BigQuery SQL literal.

    Args:
        value: Value to serialize

    Returns:
        String representation suitable for SQL
    """
    if pd.isna(value):
        return "NULL"
    if isinstance(value, str):
        # Escape single quotes and backslashes for SQL strings
        return f"'{value.replace(chr(92), chr(92)*2).replace(chr(39), chr(39)*2)}'"
    if isinstance(value, bytes):
        return f"b'{value.decode('utf-8', 'replace').replace(chr(92), chr(92)*2).replace(chr(39), chr(39)*2)}'"
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
        # Timestamps and datetimes need to be quoted
        return f"'{value}'"
    if isinstance(value, (list, np.ndarray)):
        # Format arrays
        return f"[{', '.join(_serialize_value_for_sql(v) for v in value)}]"
    if isinstance(value, dict):
        # For STRUCT, BQ expects ('val1', 'val2', ...)
        return f"({', '.join(_serialize_value_for_sql(v) for v in value.values())})"
    return str(value)


def get_database_settings(context: ToolContext) -> Dict:
    """Get database configuration and schema information.

    Retrieves BigQuery dataset metadata, table schemas, and sample data.
    Results are cached to avoid redundant queries.

    Args:
        context: Tool context containing BigQuery client

    Returns:
        Dictionary with project_id, dataset_id, table_schemas, and sample data

    Raises:
        NotFound: If dataset or table doesn't exist
        Exception: For other BigQuery errors
    """
    global _database_settings_cache

    if _database_settings_cache is not None:
        logger.info("Returning cached database settings")
        return _database_settings_cache

    try:
        client = get_bigquery_client(project=BIGQUERY_PROJECT_ID, credentials=None)
        dataset_ref = bigquery.DatasetReference(BIGQUERY_PROJECT_ID, BIGQUERY_DATASET)

        tables_info = {}
        for table in client.list_tables(dataset_ref):
            table_ref = bigquery.TableReference(dataset_ref, table.table_id)
            table_info = client.get_table(table_ref)

            # Get schema
            table_schema = [
                (field.name, field.field_type)
                for field in table_info.schema
            ]

            # Get sample values
            sample_query = f"SELECT * FROM `{table_ref}` LIMIT 5"
            sample_df = client.query(sample_query).to_dataframe()
            sample_values = sample_df.to_dict(orient="list")

            # Serialize sample values for SQL representation
            for key in sample_values:
                sample_values[key] = [
                    _serialize_value_for_sql(v) for v in sample_values[key]
                ]

            tables_info[str(table_ref)] = {
                "table_schema": table_schema,
                "example_values": sample_values,
                "row_count": table_info.num_rows
            }

        _database_settings_cache = {
            "project_id": BIGQUERY_PROJECT_ID,
            "dataset_id": BIGQUERY_DATASET,
            "table_name": BIGQUERY_TABLE,
            "full_table_name": FULL_TABLE_NAME,
            "tables": tables_info
        }

        logger.info(f"Retrieved database settings for {len(tables_info)} tables")
        return _database_settings_cache

    except NotFound as e:
        logger.error(f"Dataset or table not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error retrieving database settings: {e}")
        raise


def get_customer_usage_data(
    customer_id: str,
    start_date: str,
    end_date: str,
    tool_context: ToolContext,
    granularity: str = "daily",
    max_rows: Optional[int] = 1000
) -> dict:
    """Query customer usage data by customer_id and date range.

    Args:
        customer_id: Customer identifier (e.g., 'CUST_001')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        tool_context: Tool context containing BigQuery client
        granularity: Time granularity - 'hourly', 'daily', 'weekly' (default: 'daily')
        max_rows: Maximum rows to return (default: 1000, None for unlimited)

    Returns:
        Serialized dictionary with usage data including all disaggregated loads

    Raises:
        ValueError: If date format is invalid, customer_id is empty, or granularity invalid
        NotFound: If table doesn't exist
        Exception: For other BigQuery errors
    """
    # Validate inputs
    if not customer_id or not customer_id.strip():
        raise ValueError("customer_id cannot be empty")

    # Validate granularity
    valid_granularities = ["hourly", "daily", "weekly"]
    if granularity not in valid_granularities:
        raise ValueError(f"Invalid granularity: {granularity}. Must be one of: {', '.join(valid_granularities)}")

    # Normalize customer_id to uppercase for case-insensitive matching
    customer_id = customer_id.strip().upper()

    try:
        # Validate date formats
        datetime.datetime.strptime(start_date, "%Y-%m-%d")
        datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    try:
        client = get_bigquery_client(project=BIGQUERY_PROJECT_ID, credentials=None)

        # Build query based on granularity
        if granularity == "hourly":
            # Hourly: Return raw data with LIMIT for safety
            query = f"""
            SELECT *
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
              AND DATE(timestamp) BETWEEN @start_date AND @end_date
            ORDER BY timestamp
            {f'LIMIT {max_rows}' if max_rows else ''}
            """

        elif granularity == "daily":
            # Daily: Aggregate by date
            query = f"""
            SELECT
                DATE(timestamp) as date,
                AVG(outdoor_temperature_f) as avg_temperature_f,
                SUM(total_usage_kwh) as total_usage_kwh,
                SUM(hvac_heating_kwh) as hvac_heating_kwh,
                SUM(hvac_cooling_kwh) as hvac_cooling_kwh,
                SUM(water_heater_kwh) as water_heater_kwh,
                SUM(ev_charging_kwh) as ev_charging_kwh,
                SUM(pool_equipment_kwh) as pool_equipment_kwh,
                SUM(major_appliances_kwh) as major_appliances_kwh,
                SUM(other_loads_kwh) as other_loads_kwh,
                MAX(total_usage_kwh) as peak_hourly_kwh,
                SUM(cooling_degree_hours) as cooling_degree_hours,
                SUM(heating_degree_hours) as heating_degree_hours,
                COUNT(*) as num_hourly_readings
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
              AND DATE(timestamp) BETWEEN @start_date AND @end_date
            GROUP BY date
            ORDER BY date
            {f'LIMIT {max_rows}' if max_rows else ''}
            """

        else:  # weekly
            # Weekly: Aggregate by week
            query = f"""
            SELECT
                DATE_TRUNC(DATE(timestamp), WEEK) as week_start_date,
                AVG(outdoor_temperature_f) as avg_temperature_f,
                SUM(total_usage_kwh) as total_usage_kwh,
                SUM(hvac_heating_kwh) as hvac_heating_kwh,
                SUM(hvac_cooling_kwh) as hvac_cooling_kwh,
                SUM(water_heater_kwh) as water_heater_kwh,
                SUM(ev_charging_kwh) as ev_charging_kwh,
                SUM(pool_equipment_kwh) as pool_equipment_kwh,
                SUM(major_appliances_kwh) as major_appliances_kwh,
                SUM(other_loads_kwh) as other_loads_kwh,
                MAX(total_usage_kwh) as peak_hourly_kwh,
                SUM(cooling_degree_hours) as cooling_degree_hours,
                SUM(heating_degree_hours) as heating_degree_hours,
                COUNT(*) as num_hourly_readings
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
              AND DATE(timestamp) BETWEEN @start_date AND @end_date
            GROUP BY week_start_date
            ORDER BY week_start_date
            {f'LIMIT {max_rows}' if max_rows else ''}
            """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
                bigquery.ScalarQueryParameter("start_date", "DATE", start_date),
                bigquery.ScalarQueryParameter("end_date", "DATE", end_date),
            ]
        )

        logger.info(f"Querying {granularity} usage data for {customer_id} from {start_date} to {end_date} (max_rows: {max_rows})")
        df = client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            logger.warning(f"No data found for customer {customer_id} in date range {start_date} to {end_date}")
        else:
            logger.info(f"Retrieved {len(df)} rows at {granularity} granularity for customer {customer_id}")

        return serialize_dataframe(df)

    except NotFound as e:
        logger.error(f"Table not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error querying customer usage data: {e}")
        raise


def query_usage_patterns(
    customer_id: str,
    pattern_type: str,
    tool_context: ToolContext
) -> dict:
    """Query usage patterns with aggregation and analysis.

    Supports multiple pattern analysis types with appropriate grouping
    and statistical calculations.

    Args:
        customer_id: Customer identifier
        pattern_type: Type of pattern analysis:
            - 'hourly_average': Average usage by hour of day
            - 'daily_peaks': Peak usage by day
            - 'appliance_breakdown': Usage by appliance category
            - 'seasonal_trends': Monthly usage trends
        tool_context: Tool context containing BigQuery client

    Returns:
        Serialized dictionary with pattern analysis results

    Raises:
        ValueError: If pattern_type is invalid
        Exception: For BigQuery errors
    """
    if not customer_id or not customer_id.strip():
        raise ValueError("customer_id cannot be empty")

    # Normalize customer_id to uppercase for case-insensitive matching
    customer_id = customer_id.strip().upper()

    try:
        client = get_bigquery_client(project=BIGQUERY_PROJECT_ID, credentials=None)

        # Build query based on pattern type
        if pattern_type == "hourly_average":
            query = f"""
            SELECT
                EXTRACT(HOUR FROM timestamp) as hour_of_day,
                AVG(total_usage_kwh) as avg_total_kwh,
                AVG(hvac_cooling_kwh) as avg_hvac_cooling,
                AVG(hvac_heating_kwh) as avg_hvac_heating,
                COUNT(*) as num_readings
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """

        elif pattern_type == "daily_peaks":
            query = f"""
            SELECT
                DATE(timestamp) as date,
                MAX(total_usage_kwh) as peak_kwh,
                EXTRACT(HOUR FROM timestamp) as peak_hour,
                AVG(total_usage_kwh) as avg_kwh
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
            GROUP BY date, peak_hour
            ORDER BY date
            """

        elif pattern_type == "appliance_breakdown":
            query = f"""
            SELECT
                DATE(timestamp) as date,
                SUM(hvac_cooling_kwh) as total_hvac_cooling,
                SUM(hvac_heating_kwh) as total_hvac_heating,
                SUM(water_heater_kwh) as total_water_heater,
                SUM(ev_charging_kwh) as total_ev_charging,
                SUM(pool_equipment_kwh) as total_pool_equipment,
                SUM(major_appliances_kwh) as total_major_appliances,
                SUM(other_loads_kwh) as total_other_loads,
                SUM(total_usage_kwh) as total_usage
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
            GROUP BY date
            ORDER BY date
            """

        elif pattern_type == "seasonal_trends":
            query = f"""
            SELECT
                EXTRACT(YEAR FROM timestamp) as year,
                EXTRACT(MONTH FROM timestamp) as month,
                AVG(total_usage_kwh) as avg_kwh,
                AVG(outdoor_temperature_f) as avg_temp,
                SUM(cooling_degree_hours) as total_cdh,
                SUM(heating_degree_hours) as total_hdh
            FROM `{FULL_TABLE_NAME}`
            WHERE customer_id = @customer_id
            GROUP BY year, month
            ORDER BY year, month
            """

        else:
            raise ValueError(
                f"Invalid pattern_type: {pattern_type}. "
                f"Must be one of: hourly_average, daily_peaks, appliance_breakdown, seasonal_trends"
            )

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
            ]
        )

        logger.info(f"Querying {pattern_type} pattern for customer {customer_id}")
        df = client.query(query, job_config=job_config).to_dataframe()

        if df.empty:
            logger.warning(f"No pattern data found for customer {customer_id}")
        else:
            logger.info(f"Retrieved {len(df)} rows for {pattern_type} pattern")

        return serialize_dataframe(df)

    except Exception as e:
        logger.error(f"Error querying usage patterns: {e}")
        raise


def serialize_dataframe(df: pd.DataFrame) -> dict:
    """Convert DataFrame to JSON-serializable dictionary.

    Args:
        df: Pandas DataFrame to serialize

    Returns:
        Dictionary with serialized data
    """
    # Convert to JSON-compatible format
    # This handles datetime, date, timestamp, and other non-serializable types
    json_str = df.to_json(orient="records", date_format="iso")
    data = json.loads(json_str)

    return {
        "columns": list(df.columns),
        "data": data,
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


def deserialize_dataframe(data: Dict) -> pd.DataFrame:
    """Convert serialized dictionary back to DataFrame.

    Args:
        data: Dictionary with serialized DataFrame data

    Returns:
        Pandas DataFrame
    """
    return pd.DataFrame(data["data"], columns=data["columns"])


if __name__ == "__main__":
    """Manual test script for validating BigQuery tools against real data."""

    print("=" * 70)
    print("BigQuery Tools Manual Validation")
    print("=" * 70)
    print()

    # Create a mock ToolContext (simple object with empty attributes)
    class MockToolContext:
        def __init__(self):
            self.state = {}
            self.resources = {}

    context = MockToolContext()

    try:
        # Test 1: Get database settings
        print("Test 1: Getting database settings...")
        print("-" * 70)
        settings = get_database_settings(context)
        print(f"✓ Project ID: {settings['project_id']}")
        print(f"✓ Dataset: {settings['dataset_id']}")
        print(f"✓ Table: {settings['table_name']}")
        print(f"✓ Found {len(settings['tables'])} table(s)")

        for table_name, table_info in settings['tables'].items():
            print(f"\n  Table: {table_name}")
            print(f"  Rows: {table_info.get('row_count', 'unknown')}")
            print(f"  Schema: {len(table_info['table_schema'])} columns")
            print(f"  Sample columns: {[col[0] for col in table_info['table_schema'][:5]]}")
        print()

        # Test 2: Get customer usage data
        print("Test 2: Querying customer usage data...")
        print("-" * 70)

        # Get an actual customer_id from the database
        client = get_bigquery_client(project=BIGQUERY_PROJECT_ID, credentials=None)
        sample_query = f"SELECT DISTINCT customer_id FROM `{FULL_TABLE_NAME}` LIMIT 1"
        sample_result = client.query(sample_query).to_dataframe()

        if not sample_result.empty:
            test_customer_id = sample_result.iloc[0]['customer_id']
        else:
            test_customer_id = "CUST_001"  # Fallback

        print(f"Using test customer_id: {test_customer_id}")

        test_start_date = "2024-07-01"
        test_end_date = "2024-07-31"

        data = get_customer_usage_data(
            test_customer_id,
            test_start_date,
            test_end_date,
            tool_context=context
        )
        df = deserialize_dataframe(data)

        if not df.empty:
            print(f"✓ Retrieved {len(df)} rows for customer {test_customer_id}")
            print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"✓ Columns: {list(df.columns)}")
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
        else:
            print(f"⚠ No data found for customer {test_customer_id}")
            print("  This may be expected if the customer doesn't exist")
        print()

        # Test 3: Query usage patterns
        print("Test 3: Querying usage patterns...")
        print("-" * 70)

        pattern_data = query_usage_patterns(
            test_customer_id,
            "hourly_average",
            tool_context=context
        )
        pattern_df = deserialize_dataframe(pattern_data)

        if not pattern_df.empty:
            print(f"✓ Retrieved hourly average pattern with {len(pattern_df)} hours")
            print(f"✓ Columns: {list(pattern_df.columns)}")
            print(f"\nSample hourly averages:")
            print(pattern_df.head().to_string())
        else:
            print(f"⚠ No pattern data found for customer {test_customer_id}")
        print()

        # Test 4: Data serialization
        print("Test 4: Testing data serialization...")
        print("-" * 70)
        if not df.empty:
            serialized = serialize_dataframe(df.head(5))
            print(f"✓ Serialized {serialized['shape'][0]} rows, {serialized['shape'][1]} columns")

            deserialized = deserialize_dataframe(serialized)
            print(f"✓ Deserialized back to DataFrame: {deserialized.shape}")
            print(f"✓ Columns match: {list(deserialized.columns) == serialized['columns']}")
        print()

        # Test 5: Error handling
        print("Test 5: Testing error scenarios...")
        print("-" * 70)

        try:
            # Invalid customer ID
            get_customer_usage_data("", "2024-01-01", "2024-01-31", tool_context=context)
            print("✗ Empty customer_id should raise ValueError")
        except ValueError as e:
            print(f"✓ Empty customer_id correctly rejected: {e}")

        try:
            # Invalid date format
            get_customer_usage_data("CUST_001", "2024/01/01", "2024-01-31", tool_context=context)
            print("✗ Invalid date format should raise ValueError")
        except ValueError as e:
            print(f"✓ Invalid date format correctly rejected")

        try:
            # Invalid pattern type
            query_usage_patterns("CUST_001", "invalid_pattern", tool_context=context)
            print("✗ Invalid pattern_type should raise ValueError")
        except ValueError as e:
            print(f"✓ Invalid pattern_type correctly rejected")

        print()
        print("=" * 70)
        print("✓ All tests completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
