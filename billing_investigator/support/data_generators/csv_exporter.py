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

"""
CSV Generation and Export Module
Formats and exports synthetic energy data as BigQuery-compatible CSV files.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from .schema import CustomerProfile, ENERGY_CSV_COLUMNS, CUSTOMER_CSV_COLUMNS, ENERGY_CONSUMPTION_SCHEMA
from .customer_profiles import CustomerProfileGenerator
from .consumption_patterns import ConsumptionPatternEngine
from .load_disaggregation import LoadDisaggregator
from .anomaly_injector import AnomalyInjector, Anomaly


class CSVExporter:
    """Exports synthetic energy data to BigQuery-compatible CSV files."""
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the CSV exporter.
        
        Args:
            output_dir: Directory to write CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation settings
        self.component_sum_tolerance = 0.01  # ±1%
        self.precision = 2  # Decimal places for kWh values
    
    def validate_data(
        self,
        df: pd.DataFrame,
        profile: CustomerProfile
    ) -> List[str]:
        """
        Validate generated data for quality issues using schema-aware validation.

        Args:
            df: DataFrame with consumption data
            profile: Customer profile

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Schema-aware range validation
        for col in df.columns:
            if col in ENERGY_CONSUMPTION_SCHEMA:
                schema_def = ENERGY_CONSUMPTION_SCHEMA[col]

                # Skip non-numeric columns
                if schema_def["type"] not in ["FLOAT64", "INT64"]:
                    continue

                # Check against defined range if present
                if "range" in schema_def:
                    min_val, max_val = schema_def["range"]
                    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()

                    if out_of_range > 0:
                        errors.append(
                            f"Column '{col}' has {out_of_range} values outside "
                            f"valid range [{min_val}, {max_val}]"
                        )

        # Check total equals sum of components (±1%)
        component_cols = [
            "water_heater_kwh",
            "ev_charging_kwh",
            "pool_equipment_kwh",
            "major_appliances_kwh",
            "other_loads_kwh"
        ]

        if all(col in df.columns for col in component_cols):
            component_sum = df[component_cols].sum(axis=1)
            baseline_load = df["total_usage_kwh"] - df["hvac_heating_kwh"] - df["hvac_cooling_kwh"]

            # Calculate percentage difference
            pct_diff = abs((component_sum - baseline_load) / baseline_load)

            invalid_rows = (pct_diff > self.component_sum_tolerance).sum()
            if invalid_rows > 0:
                errors.append(
                    f"{invalid_rows} rows have component sum mismatch "
                    f"(tolerance: ±{self.component_sum_tolerance*100}%)"
                )

        # Check for missing timestamps
        if "timestamp" in df.columns:
            # Check for duplicate timestamps
            duplicates = df["timestamp"].duplicated().sum()
            if duplicates > 0:
                errors.append(f"{duplicates} duplicate timestamps found")

            # Check for gaps in hourly data
            df_sorted = df.sort_values("timestamp").copy()
            df_sorted["timestamp_dt"] = pd.to_datetime(df_sorted["timestamp"])
            time_diffs = df_sorted["timestamp_dt"].diff()
            expected_diff = pd.Timedelta(hours=1)

            gaps = (time_diffs > expected_diff * 1.1).sum()  # Allow 10% tolerance
            if gaps > 0:
                errors.append(f"{gaps} gaps in hourly timestamps detected")

        return errors

    def auto_fix_data(
        self,
        df: pd.DataFrame,
        profile: CustomerProfile
    ) -> Dict[str, int]:
        """
        Automatically fix floating point rounding errors in component sums.

        Note: Range clipping removed - generation code now respects schema limits.
        Validation will fail loudly if any values are out of range.

        Args:
            df: DataFrame with consumption data (modified in-place)
            profile: Customer profile

        Returns:
            Dictionary with counts of fixes applied
        """
        fixes = {
            "component_adjustments": 0
        }

        # Adjust component sums to match baseline (fixes floating point rounding)
        component_cols = [
            "water_heater_kwh",
            "ev_charging_kwh",
            "pool_equipment_kwh",
            "major_appliances_kwh",
            "other_loads_kwh"
        ]

        if all(col in df.columns for col in component_cols):
            for idx in df.index:
                component_sum = df.loc[idx, component_cols].sum()
                baseline_load = (
                    df.loc[idx, "total_usage_kwh"] -
                    df.loc[idx, "hvac_heating_kwh"] -
                    df.loc[idx, "hvac_cooling_kwh"]
                )

                # Check if adjustment needed
                if baseline_load > 0:
                    pct_diff = abs((component_sum - baseline_load) / baseline_load)

                    if pct_diff > self.component_sum_tolerance:
                        # Proportionally adjust all components
                        adjustment_factor = baseline_load / component_sum

                        # Define schema limits for each component
                        schema_limits = {
                            "water_heater_kwh": 5.0,
                            "ev_charging_kwh": 12.0,
                            "pool_equipment_kwh": 8.0,
                            "major_appliances_kwh": 8.0,
                            "other_loads_kwh": 5.0
                        }

                        for col in component_cols:
                            adjusted_value = df.loc[idx, col] * adjustment_factor
                            # Re-enforce schema limits after adjustment
                            if col in schema_limits:
                                adjusted_value = min(adjusted_value, schema_limits[col])
                            df.loc[idx, col] = round(adjusted_value, self.precision)

                        fixes["component_adjustments"] += 1

        return fixes

    def generate_timeseries_data(
        self,
        profile: CustomerProfile,
        start_date: datetime,
        end_date: datetime,
        pattern_engine: ConsumptionPatternEngine,
        disaggregator: LoadDisaggregator,
        anomalies: Optional[List[Anomaly]] = None
    ) -> pd.DataFrame:
        """
        Generate complete timeseries data for a customer.
        
        Args:
            profile: Customer profile
            start_date: Start date for data generation
            end_date: End date for data generation
            pattern_engine: Consumption pattern engine
            disaggregator: Load disaggregator
            anomalies: Optional list of anomalies to inject
            
        Returns:
            DataFrame with hourly consumption data
        """
        data_rows = []
        current_time = start_date
        
        while current_time <= end_date:
            # Generate baseline consumption and HVAC
            baseline, hvac_heat, hvac_cool, temp, cdh, hdh = \
                pattern_engine.generate_hourly_consumption(profile, current_time)
            
            # Disaggregate baseline into components
            components = disaggregator.disaggregate_load(profile, baseline, current_time)

            # Apply anomalies if present
            if anomalies:
                # Build consumption dict with all components
                consumption_data = {
                    "hvac_heating": hvac_heat,
                    "hvac_cooling": hvac_cool,
                    **components
                }

                # Apply anomalies
                injector = AnomalyInjector()  # Temporary injector for application
                modified_data = injector.apply_anomalies_to_data(
                    anomalies, current_time, consumption_data
                )

                # Extract modified values
                hvac_heat = modified_data.get("hvac_heating", hvac_heat)
                hvac_cool = modified_data.get("hvac_cooling", hvac_cool)
                components = {
                    k: modified_data.get(k, v)
                    for k, v in components.items()
                }
            
            # Calculate total consumption
            total_kwh = (
                hvac_heat + hvac_cool + 
                sum(components.values())
            )
            
            # Build row
            row = {
                "customer_id": profile.customer_id,
                "timestamp": current_time.isoformat(),
                "billing_period_id": f"{current_time.year}-{current_time.month:02d}",  # YYYY-MM format
                "total_usage_kwh": round(total_kwh, self.precision),
                "hvac_heating_kwh": round(hvac_heat, self.precision),
                "hvac_cooling_kwh": round(hvac_cool, self.precision),
                "ev_charging_kwh": round(components["ev_charging_kwh"], self.precision),
                "water_heater_kwh": round(components["water_heater_kwh"], self.precision),
                "pool_equipment_kwh": round(components["pool_equipment_kwh"], self.precision),
                "major_appliances_kwh": round(components["major_appliances_kwh"], self.precision),
                "other_loads_kwh": round(components["other_loads_kwh"], self.precision),
                "outdoor_temperature_f": round(temp, 1),
                "cooling_degree_hours": round(cdh, 2),
                "heating_degree_hours": round(hdh, 2)
            }
            
            data_rows.append(row)
            current_time += timedelta(hours=1)
        
        return pd.DataFrame(data_rows)
    
    def export_customer_usage(
        self,
        profile: CustomerProfile,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> Path:
        """
        Export customer usage data to CSV file.
        
        Args:
            profile: Customer profile
            df: DataFrame with usage data
            start_date: Start date
            end_date: End date
            
        Returns:
            Path to created CSV file
        """
        # Format filename
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        filename = f"customer_{profile.customer_id}_usage_{start_str}_{end_str}.csv"
        filepath = self.output_dir / filename
        
        # Ensure column order matches schema
        ordered_df = df[ENERGY_CSV_COLUMNS]
        
        # Export to CSV
        ordered_df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_customer_metadata(
        self,
        profiles: List[CustomerProfile]
    ) -> Path:
        """
        Export customer metadata to CSV file.
        
        Args:
            profiles: List of customer profiles
            
        Returns:
            Path to created CSV file
        """
        # Convert profiles to DataFrame
        profile_dicts = [p.to_dict() for p in profiles]
        df = pd.DataFrame(profile_dicts)
        
        # Ensure column order matches schema
        ordered_df = df[CUSTOMER_CSV_COLUMNS]
        
        # Export to CSV
        filepath = self.output_dir / "customer_metadata.csv"
        ordered_df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_anomaly_log(
        self,
        anomalies: List[Anomaly]
    ) -> Path:
        """
        Export anomaly log to CSV file.
        
        Args:
            anomalies: List of anomalies
            
        Returns:
            Path to created CSV file
        """
        if not anomalies:
            return None
        
        # Convert anomalies to DataFrame
        anomaly_dicts = [a.to_dict() for a in anomalies]
        df = pd.DataFrame(anomaly_dicts)
        
        # Export to CSV
        filepath = self.output_dir / "anomaly_log.csv"
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def generate_full_dataset(
        self,
        num_customers: int,
        start_date: datetime,
        end_date: datetime,
        seed: int = 42,
        inject_anomalies: bool = True
    ) -> Dict[str, Path]:
        """
        Generate complete synthetic dataset for multiple customers.
        
        Args:
            num_customers: Number of customer profiles to generate
            start_date: Start date for data generation
            end_date: End date for data generation
            seed: Random seed for reproducibility
            inject_anomalies: Whether to inject anomalies
            
        Returns:
            Dictionary with paths to generated files
        """
        print(f"Generating synthetic dataset...")
        print(f"Customers: {num_customers}")
        print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Initialize generators
        profile_gen = CustomerProfileGenerator(seed=seed)
        pattern_engine = ConsumptionPatternEngine(seed=seed)
        disaggregator = LoadDisaggregator(seed=seed)
        
        # Generate profiles
        print("Generating customer profiles...")
        profiles = profile_gen.generate_profiles(count=num_customers)
        
        # Generate anomalies if requested
        all_anomalies = []
        profile_anomalies_map = {}
        
        if inject_anomalies:
            print("Selecting anomalies...")
            anomaly_injector = AnomalyInjector(seed=seed)
            
            for profile in profiles:
                profile_anomalies = anomaly_injector.select_anomalies_for_profile(
                    profile, start_date, end_date
                )
                if profile_anomalies:
                    profile_anomalies_map[profile.customer_id] = profile_anomalies
                    all_anomalies.extend(profile_anomalies)
            
            print(f"  {len(all_anomalies)} anomalies across {len(profile_anomalies_map)} customers")
        
        # Generate usage data for each customer
        print("\nGenerating usage data...")
        generated_files = {}
        validation_errors = []
        fix_stats = {
            "total_component_adjustments": 0,
            "customers_with_fixes": 0
        }

        for i, profile in enumerate(profiles, 1):
            print(f"  [{i}/{num_customers}] {profile.customer_id}...", end="", flush=True)

            # Get anomalies for this profile
            anomalies = profile_anomalies_map.get(profile.customer_id, [])

            # Generate timeseries
            df = self.generate_timeseries_data(
                profile, start_date, end_date,
                pattern_engine, disaggregator, anomalies
            )

            # Auto-fix floating point rounding errors
            fixes = self.auto_fix_data(df, profile)
            if fixes["component_adjustments"] > 0:
                fix_stats["total_component_adjustments"] += fixes["component_adjustments"]
                fix_stats["customers_with_fixes"] += 1

            # Validate data after fixes
            errors = self.validate_data(df, profile)
            if errors:
                validation_errors.append((profile.customer_id, errors))
                print(f" ⚠️ {len(errors)} validation errors")
            else:
                print(" ✅")

            # Export to CSV
            filepath = self.export_customer_usage(profile, df, start_date, end_date)
            generated_files[profile.customer_id] = filepath
        
        # Export metadata
        print("\nExporting metadata...")
        metadata_path = self.export_customer_metadata(profiles)
        generated_files["metadata"] = metadata_path
        print(f"  Customer metadata: {metadata_path}")
        
        # Export anomaly log
        if all_anomalies:
            anomaly_path = self.export_anomaly_log(all_anomalies)
            generated_files["anomalies"] = anomaly_path
            print(f"  Anomaly log: {anomaly_path}")
        
        # Summary
        print(f"\n{'='*70}")
        print("Generation Complete!")
        print(f"{'='*70}")
        print(f"Total files: {len(generated_files)}")
        print(f"Output directory: {self.output_dir}")

        # Show floating point adjustment statistics
        if fix_stats["customers_with_fixes"] > 0:
            print(f"\n🔧 Floating Point Adjustments:")
            print(f"  Customers with component sum adjustments: {fix_stats['customers_with_fixes']}")
            print(f"  Total adjustments: {fix_stats['total_component_adjustments']}")

        # Show validation errors (any range violations will now fail loudly)
        if validation_errors:
            print(f"\n⚠️  Validation Errors: {len(validation_errors)} customers")
            for customer_id, errors in validation_errors[:5]:  # Show first 5
                print(f"  {customer_id}:")
                for error in errors:
                    print(f"    - {error}")
        else:
            print("\n✅ All data validated successfully")

        return generated_files


if __name__ == "__main__":
    # Test CSV export
    import sys
    
    # Configuration
    num_customers = 5
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)  # 1 month for testing
    output_dir = "test_output"
    
    print("Testing CSV Export Module")
    print("=" * 70)
    
    # Create exporter
    exporter = CSVExporter(output_dir=output_dir)
    
    # Generate dataset
    generated_files = exporter.generate_full_dataset(
        num_customers=num_customers,
        start_date=start_date,
        end_date=end_date,
        seed=42,
        inject_anomalies=True
    )
    
    print("\nSample of generated files:")
    for key, path in list(generated_files.items())[:5]:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  {path.name}: {size_kb:.1f} KB")
    
    # Load and display sample data
    print("\nSample data from first customer:")
    first_customer_file = next(
        (f for k, f in generated_files.items() if k.startswith("CUST_")),
        None
    )
    
    if first_customer_file:
        df = pd.read_csv(first_customer_file)
        print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        
        print(f"\nData summary:")
        print(df[["total_usage_kwh", "hvac_heating_kwh", "hvac_cooling_kwh", 
                  "water_heater_kwh", "ev_charging_kwh"]].describe())
