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
Load Disaggregation Module
Splits total consumption into individual load components realistically.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, Dict
from .schema import CustomerProfile


class LoadDisaggregator:
    """Disaggregates baseline load into individual components."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the load disaggregator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        
        # Component allocation percentages (of baseline load)
        self.allocation_ranges = {
            "water_heater": (0.15, 0.25),      # 15-25%
            "major_appliances": (0.10, 0.20),  # 10-20%
            "other_loads": (0.15, 0.25),       # 15-25% (lighting, electronics)
        }
        
        # EV charging parameters
        self.ev_charging_hours = (22, 6)  # 10pm to 6am
        self.ev_charging_duration = (2, 4)  # 2-4 hours
        self.ev_charging_power = (3.0, 7.0)  # 3-7 kW (Level 2 charger)
        
        # Pool equipment parameters
        self.pool_runtime_hours = (6, 8)  # 6-8 hours daily
        self.pool_power = (1.5, 2.5)  # 1.5-2.5 kW
        self.pool_active_months = [4, 5, 6, 7, 8, 9, 10]  # Apr-Oct
    
    def allocate_water_heater(
        self,
        baseline_load: float,
        hour: int,
        month: int
    ) -> float:
        """
        Allocate water heater load with morning/evening peaks.

        Args:
            baseline_load: Total baseline load to split
            hour: Hour of day (0-23)
            month: Month (1-12)

        Returns:
            Water heater load in kWh
        """
        # Base allocation (15-25% of baseline)
        base_percentage = self.rng.uniform(*self.allocation_ranges["water_heater"])
        base_load = baseline_load * base_percentage

        # Time-of-day multipliers (peaks during shower times)
        time_multipliers = {
            0: 0.3, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3, 5: 0.7,
            6: 1.8, 7: 2.2, 8: 1.5, 9: 0.8, 10: 0.6, 11: 0.5,
            12: 0.6, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.6, 17: 0.9,
            18: 1.3, 19: 1.7, 20: 1.4, 21: 1.0, 22: 0.7, 23: 0.4
        }

        multiplier = time_multipliers[hour]

        # Winter uses slightly more (colder inlet water)
        if month in [11, 12, 1, 2, 3]:
            multiplier *= 1.15

        load = base_load * multiplier

        # Add noise
        load *= self.rng.normal(1.0, 0.10)

        # Enforce schema max limit of 5.0 kWh
        return max(0, min(load, 5.0))
    
    def allocate_ev_charging(
        self,
        profile: CustomerProfile,
        baseline_load: float,
        hour: int,
        day_of_week: int
    ) -> float:
        """
        Allocate EV charging load with night charging pattern.
        
        Args:
            profile: Customer profile
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            
        Returns:
            EV charging load in kWh
        """
        if not profile.ev_owned:
            return 0.0
        
        # Determine if charging occurs this hour
        # Assume charging 4-5 days per week
        charging_days = self.rng.choice([4, 5])
        charges_today = day_of_week < charging_days
        
        if not charges_today:
            return 0.0
        
        # Night charging window (10pm-6am)
        start_hour, end_hour = self.ev_charging_hours
        
        # Check if we're in charging window
        if start_hour > end_hour:  # Wraps midnight
            in_window = hour >= start_hour or hour < end_hour
        else:
            in_window = start_hour <= hour < end_hour
        
        if not in_window:
            return 0.0
        
        # Determine charging session duration (persistent within a day)
        # Use day_of_week as part of seed for consistency
        session_rng = np.random.default_rng(self.rng.bit_generator.random_raw() + day_of_week)
        duration_hours = session_rng.integers(*self.ev_charging_duration)
        
        # Start charging at a random hour in the window
        # For simplicity, assume charging starts at 10pm or 11pm
        start_options = [22, 23]
        charging_start = session_rng.choice(start_options)
        charging_end = (charging_start + duration_hours) % 24
        
        # Check if current hour is in active charging session
        if charging_start < charging_end:
            is_charging = charging_start <= hour < charging_end
        else:  # Wraps midnight
            is_charging = hour >= charging_start or hour < charging_end
        
        if not is_charging:
            return 0.0
        
        # EV charging power (3-7 kW Level 2 charger)
        charging_power = self.rng.uniform(*self.ev_charging_power)
        
        # Add slight variation
        charging_power *= self.rng.normal(1.0, 0.05)
        
        return max(0, charging_power)
    
    def allocate_pool_equipment(
        self,
        profile: CustomerProfile,
        hour: int,
        month: int
    ) -> float:
        """
        Allocate pool equipment load with seasonal operation.
        
        Args:
            profile: Customer profile
            hour: Hour of day (0-23)
            month: Month (1-12)
            
        Returns:
            Pool equipment load in kWh
        """
        if not profile.pool_equipped:
            return 0.0
        
        # Only operate during warm months
        if month not in self.pool_active_months:
            return 0.0
        
        # Pool pump runs 6-8 hours per day (typically mid-morning to afternoon)
        # Run hours: 9am to 5pm (8 hours), but randomly 6-8 hours
        runtime_hours = self.rng.integers(*self.pool_runtime_hours)
        
        # Typical schedule: start around 9am-10am
        start_hour = self.rng.choice([9, 10])
        end_hour = start_hour + runtime_hours
        
        # Check if currently running
        is_running = start_hour <= hour < end_hour
        
        if not is_running:
            return 0.0
        
        # Pool pump power (1.5-2.5 kW)
        base_power = self.rng.uniform(*self.pool_power)
        
        # Peak summer months use more (pool heater, longer runtime)
        if month in [6, 7, 8]:
            base_power *= 1.3
        
        # Add variation
        power = base_power * self.rng.normal(1.0, 0.08)
        
        return max(0, power)
    
    def allocate_major_appliances(
        self,
        baseline_load: float,
        hour: int,
        is_weekend: bool
    ) -> float:
        """
        Allocate major appliances load with scheduled cycles.

        Args:
            baseline_load: Total baseline load to split
            hour: Hour of day (0-23)
            is_weekend: Whether it's a weekend

        Returns:
            Major appliances load in kWh
        """
        # Base allocation (10-20% of baseline)
        base_percentage = self.rng.uniform(*self.allocation_ranges["major_appliances"])
        base_load = baseline_load * base_percentage

        # Time-of-day patterns for appliances
        # Dishwasher: evening after dinner
        # Washer/dryer: morning/evening, more on weekends
        # Refrigerator: constant with compressor cycles
        # Oven: meal times

        time_multipliers = {
            0: 0.5, 1: 0.4, 2: 0.4, 3: 0.4, 4: 0.5, 5: 0.6,
            6: 0.8, 7: 1.3, 8: 1.5, 9: 1.2, 10: 1.0, 11: 1.1,
            12: 1.3, 13: 0.9, 14: 0.8, 15: 0.7, 16: 0.8, 17: 1.2,
            18: 1.7, 19: 2.0, 20: 1.8, 21: 1.3, 22: 0.9, 23: 0.6
        }

        multiplier = time_multipliers[hour]

        # Weekends have more laundry/cooking
        if is_weekend and hour in range(9, 20):
            multiplier *= 1.2

        load = base_load * multiplier

        # Add appliance cycling noise
        load *= self.rng.normal(1.0, 0.15)

        # Enforce schema max limit of 8.0 kWh
        return max(0, min(load, 8.0))
    
    def allocate_other_loads(
        self,
        baseline_load: float,
        hour: int
    ) -> float:
        """
        Allocate other miscellaneous loads (lighting, electronics, etc.).

        Args:
            baseline_load: Total baseline load to split
            hour: Hour of day (0-23)

        Returns:
            Other loads in kWh
        """
        # Base allocation (15-25% of baseline)
        base_percentage = self.rng.uniform(*self.allocation_ranges["other_loads"])
        base_load = baseline_load * base_percentage

        # Time-of-day patterns (lighting, TV, computers)
        time_multipliers = {
            0: 0.4, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.4,
            6: 0.7, 7: 1.0, 8: 1.2, 9: 1.0, 10: 0.9, 11: 0.9,
            12: 1.0, 13: 0.9, 14: 0.8, 15: 0.9, 16: 1.1, 17: 1.4,
            18: 1.8, 19: 2.2, 20: 2.0, 21: 1.6, 22: 1.2, 23: 0.7
        }

        multiplier = time_multipliers[hour]
        load = base_load * multiplier

        # Add noise
        load *= self.rng.normal(1.0, 0.12)

        # Enforce schema max limit of 5.0 kWh
        return max(0, min(load, 5.0))
    
    def disaggregate_load(
        self,
        profile: CustomerProfile,
        baseline_load: float,
        timestamp: datetime
    ) -> Dict[str, float]:
        """
        Disaggregate baseline load into all components.
        
        Args:
            profile: Customer profile
            baseline_load: Total baseline load from consumption patterns
            timestamp: Current timestamp
            
        Returns:
            Dictionary with component loads
        """
        hour = timestamp.hour
        month = timestamp.month
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        # Allocate each component
        water_heater = self.allocate_water_heater(baseline_load, hour, month)
        ev_charging = self.allocate_ev_charging(profile, baseline_load, hour, day_of_week)
        pool_equipment = self.allocate_pool_equipment(profile, hour, month)
        major_appliances = self.allocate_major_appliances(baseline_load, hour, is_weekend)
        other_loads = self.allocate_other_loads(baseline_load, hour)
        
        # Calculate total allocated
        total_allocated = (
            water_heater + ev_charging + pool_equipment + 
            major_appliances + other_loads
        )
        
        # Adjust to match baseline (with small variance allowed)
        # Target: total_allocated ≈ baseline_load (±1%)
        if total_allocated > 0:
            adjustment_factor = baseline_load / total_allocated
            # Allow small natural variance (±3%)
            adjustment_factor *= self.rng.normal(1.0, 0.03)

            # Adjust all components proportionally
            water_heater *= adjustment_factor
            ev_charging *= adjustment_factor
            pool_equipment *= adjustment_factor
            major_appliances *= adjustment_factor
            other_loads *= adjustment_factor

            # Re-enforce schema limits after adjustment
            water_heater = min(water_heater, 5.0)
            ev_charging = min(ev_charging, 12.0)
            pool_equipment = min(pool_equipment, 8.0)
            major_appliances = min(major_appliances, 8.0)
            other_loads = min(other_loads, 5.0)

        return {
            "water_heater_kwh": round(water_heater, 2),
            "ev_charging_kwh": round(ev_charging, 2),
            "pool_equipment_kwh": round(pool_equipment, 2),
            "major_appliances_kwh": round(major_appliances, 2),
            "other_loads_kwh": round(other_loads, 2)
        }


if __name__ == "__main__":
    # Test load disaggregation
    from .customer_profiles import generate_customer_profiles
    from .consumption_patterns import ConsumptionPatternEngine
    from datetime import datetime
    
    # Generate test profile with EV and pool
    profiles = generate_customer_profiles(count=15, seed=42)
    
    # Find a profile with EV and pool for testing
    test_profile = None
    for p in profiles:
        if p.ev_owned and p.pool_equipped:
            test_profile = p
            break
    
    if test_profile is None:
        # Use first profile
        test_profile = profiles[0]
    
    print(f"Testing Load Disaggregation")
    print(f"=" * 70)
    print(f"Profile: {test_profile.customer_id}")
    print(f"Home: {test_profile.home_size_sqft} sqft, {test_profile.climate_zone}")
    print(f"EV: {test_profile.ev_owned}, Pool: {test_profile.pool_equipped}")
    print()
    
    # Generate consumption pattern
    pattern_engine = ConsumptionPatternEngine(seed=42)
    disaggregator = LoadDisaggregator(seed=42)
    
    # Test a full day in summer
    test_date = datetime(2024, 7, 15)
    
    print(f"Sample Hours (July 15, 2024):")
    print(f"=" * 70)
    print(f"{'Time':<8} {'Base':<7} {'Water':<7} {'EV':<7} {'Pool':<7} {'Appl':<7} {'Other':<7} {'Total':<7}")
    print(f"-" * 70)
    
    daily_totals = {
        "baseline": 0,
        "water": 0,
        "ev": 0,
        "pool": 0,
        "appliances": 0,
        "other": 0
    }
    
    for hour in range(24):
        ts = test_date.replace(hour=hour)
        
        # Get baseline from pattern engine
        baseline, hvac_heat, hvac_cool, temp, cdh, hdh = \
            pattern_engine.generate_hourly_consumption(test_profile, ts)
        
        # Disaggregate
        components = disaggregator.disaggregate_load(test_profile, baseline, ts)
        
        total = sum(components.values())
        
        print(f"{hour:02d}:00    {baseline:<7.2f} "
              f"{components['water_heater_kwh']:<7.2f} "
              f"{components['ev_charging_kwh']:<7.2f} "
              f"{components['pool_equipment_kwh']:<7.2f} "
              f"{components['major_appliances_kwh']:<7.2f} "
              f"{components['other_loads_kwh']:<7.2f} "
              f"{total:<7.2f}")
        
        daily_totals["baseline"] += baseline
        daily_totals["water"] += components["water_heater_kwh"]
        daily_totals["ev"] += components["ev_charging_kwh"]
        daily_totals["pool"] += components["pool_equipment_kwh"]
        daily_totals["appliances"] += components["major_appliances_kwh"]
        daily_totals["other"] += components["other_loads_kwh"]
    
    print()
    print(f"Daily Totals:")
    print(f"=" * 70)
    print(f"Baseline:         {daily_totals['baseline']:.2f} kWh")
    print(f"Water Heater:     {daily_totals['water']:.2f} kWh "
          f"({daily_totals['water']/daily_totals['baseline']*100:.1f}%)")
    print(f"EV Charging:      {daily_totals['ev']:.2f} kWh "
          f"({daily_totals['ev']/daily_totals['baseline']*100:.1f}%)")
    print(f"Pool Equipment:   {daily_totals['pool']:.2f} kWh "
          f"({daily_totals['pool']/daily_totals['baseline']*100:.1f}%)")
    print(f"Major Appliances: {daily_totals['appliances']:.2f} kWh "
          f"({daily_totals['appliances']/daily_totals['baseline']*100:.1f}%)")
    print(f"Other Loads:      {daily_totals['other']:.2f} kWh "
          f"({daily_totals['other']/daily_totals['baseline']*100:.1f}%)")
    print(f"-" * 70)
    total_sum = sum([daily_totals[k] for k in ['water', 'ev', 'pool', 'appliances', 'other']])
    print(f"Total Components: {total_sum:.2f} kWh")
    print(f"Difference:       {abs(daily_totals['baseline'] - total_sum):.2f} kWh "
          f"({abs(daily_totals['baseline'] - total_sum)/daily_totals['baseline']*100:.2f}%)")
