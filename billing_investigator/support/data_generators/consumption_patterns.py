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
Hourly Consumption Pattern Engine
Generates realistic hourly energy consumption patterns with seasonal variations.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List
from .schema import CustomerProfile


class ConsumptionPatternEngine:
    """Generates realistic hourly energy consumption patterns."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the consumption pattern engine.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        
        # Baseline load parameters (kWh per hour per sqft)
        self.baseline_kwh_per_sqft = {
            "high": 0.0008,    # Efficient homes
            "medium": 0.0012,  # Average homes
            "low": 0.0016      # Less efficient homes
        }
        
        # Time-of-use multipliers (24-hour profile)
        self.time_of_use_profile = {
            0: 0.6, 1: 0.55, 2: 0.5, 3: 0.5, 4: 0.55, 5: 0.65,
            6: 0.8, 7: 1.5, 8: 1.8, 9: 1.2, 10: 1.0, 11: 1.0,
            12: 1.1, 13: 1.0, 14: 0.95, 15: 0.9, 16: 1.1, 17: 1.5,
            18: 2.2, 19: 2.5, 20: 2.0, 21: 1.5, 22: 1.2, 23: 0.8
        }
        
        # Weekend adjustment (slightly different pattern)
        self.weekend_adjustment = {
            0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
            6: 0.9, 7: 0.8, 8: 0.7, 9: 1.2, 10: 1.3, 11: 1.2,
            12: 1.1, 13: 1.0, 14: 1.0, 15: 1.1, 16: 1.1, 17: 1.2,
            18: 1.1, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0
        }
        
        # HVAC efficiency factors (lower is better)
        self.hvac_cop = {
            "heat_pump": 3.0,           # Coefficient of Performance
            "central_ac_gas_heat": 2.5,
            "electric_resistance": 1.0,
            "none": 0.0
        }
        
        # Climate zone temperature profiles (avg temps by month in °F)
        self.climate_temps = {
            "hot_humid": {  # Southeast
                1: 55, 2: 58, 3: 65, 4: 72, 5: 78, 6: 83,
                7: 85, 8: 85, 9: 81, 10: 73, 11: 64, 12: 57
            },
            "hot_dry": {  # Southwest
                1: 52, 2: 56, 3: 62, 4: 70, 5: 78, 6: 88,
                7: 92, 8: 90, 9: 84, 10: 72, 11: 60, 12: 52
            },
            "mixed": {  # Mid-Atlantic/Midwest
                1: 32, 2: 35, 3: 45, 4: 57, 5: 67, 6: 76,
                7: 80, 8: 78, 9: 71, 10: 59, 11: 47, 12: 36
            },
            "cold": {  # Northeast/North
                1: 22, 2: 25, 3: 36, 4: 49, 5: 60, 6: 70,
                7: 75, 8: 73, 9: 64, 10: 52, 11: 40, 12: 28
            }
        }
    
    def calculate_baseline_load(
        self,
        profile: CustomerProfile,
        hour: int,
        is_weekend: bool = False
    ) -> float:
        """
        Calculate baseline load (non-HVAC, non-major-appliance consumption).
        
        Args:
            profile: Customer profile
            hour: Hour of day (0-23)
            is_weekend: Whether it's a weekend
            
        Returns:
            Baseline load in kWh
        """
        # Base load from home size and efficiency
        base_kwh_per_hour = (
            profile.home_size_sqft * 
            self.baseline_kwh_per_sqft[profile.energy_efficiency_level]
        )
        
        # Apply time-of-use multiplier
        tou_multiplier = self.time_of_use_profile[hour]
        
        # Weekend adjustment
        if is_weekend:
            tou_multiplier *= self.weekend_adjustment[hour]
        
        # Occupancy adjustment (more people = more usage)
        occupancy_factor = 0.8 + (profile.occupancy_count * 0.1)
        
        # Calculate baseline
        baseline = base_kwh_per_hour * tou_multiplier * occupancy_factor
        
        # Add random noise (±10%)
        noise = self.rng.normal(1.0, 0.1)
        baseline *= noise
        
        return max(0.1, baseline)  # Minimum 0.1 kWh
    
    def calculate_degree_hours(
        self,
        outdoor_temp: float,
        base_temp: float = 65.0
    ) -> Tuple[float, float]:
        """
        Calculate cooling and heating degree-hours.
        
        Args:
            outdoor_temp: Outdoor temperature in °F
            base_temp: Base temperature (default 65°F)
            
        Returns:
            Tuple of (cooling_degree_hours, heating_degree_hours)
        """
        if outdoor_temp > base_temp:
            cooling_dh = outdoor_temp - base_temp
            heating_dh = 0.0
        else:
            cooling_dh = 0.0
            heating_dh = base_temp - outdoor_temp

        # Enforce schema limits of [0.0, 30.0] for degree-hours
        cooling_dh = min(cooling_dh, 30.0)
        heating_dh = min(heating_dh, 30.0)

        return cooling_dh, heating_dh
    
    def generate_hourly_temperature(
        self,
        climate_zone: str,
        month: int,
        hour: int,
        day_of_month: int
    ) -> float:
        """
        Generate realistic hourly outdoor temperature.
        
        Args:
            climate_zone: Climate zone
            month: Month (1-12)
            hour: Hour of day (0-23)
            day_of_month: Day of month (1-31)
            
        Returns:
            Temperature in °F
        """
        # Get monthly average temperature
        avg_temp = self.climate_temps[climate_zone][month]
        
        # Daily variation (warmest at 3pm, coldest at 6am)
        daily_amplitude = 12.0  # ±12°F daily swing
        hour_offset = (hour - 15) * (2 * np.pi / 24)
        daily_variation = daily_amplitude * np.cos(hour_offset)
        
        # Random day-to-day variation
        daily_noise = self.rng.normal(0, 3.0)
        
        # Occasional weather events (5% chance)
        if self.rng.random() < 0.05:
            if month in [6, 7, 8]:  # Summer heat wave
                daily_noise += self.rng.uniform(5, 15)
            elif month in [12, 1, 2]:  # Winter cold snap
                daily_noise -= self.rng.uniform(5, 15)
        
        temp = avg_temp + daily_variation + daily_noise

        # Enforce schema limit of [15.0, 95.0] °F
        temp = max(15.0, min(temp, 95.0))

        return round(temp, 1)
    
    def calculate_hvac_load(
        self,
        profile: CustomerProfile,
        outdoor_temp: float,
        hour: int
    ) -> Tuple[float, float]:
        """
        Calculate HVAC heating and cooling loads.
        
        Args:
            profile: Customer profile
            outdoor_temp: Outdoor temperature in °F
            hour: Hour of day (0-23)
            
        Returns:
            Tuple of (heating_kwh, cooling_kwh)
        """
        if profile.hvac_type == "none":
            return 0.0, 0.0
        
        # Calculate degree-hours
        cooling_dh, heating_dh = self.calculate_degree_hours(outdoor_temp)
        
        # Base HVAC power (kW) scales with home size
        base_hvac_kw = profile.home_size_sqft / 500.0  # ~5 kW per 2500 sqft
        
        # Efficiency factor (SEER to COP approximation)
        efficiency_factor = profile.hvac_efficiency_seer / 10.0
        if efficiency_factor == 0:
            efficiency_factor = 1.5  # Default
        
        # Calculate loads
        heating_load = 0.0
        cooling_load = 0.0
        
        if heating_dh > 0:
            # Heating load increases with degree-hours
            heating_multiplier = min(heating_dh / 30.0, 3.0)  # Cap at 3x
            heating_load = (base_hvac_kw * heating_multiplier) / efficiency_factor
            
            # Gas heat systems use less electricity
            if profile.hvac_type == "central_ac_gas_heat":
                heating_load *= 0.3  # Only fan/controls
        
        if cooling_dh > 0:
            # Cooling load increases with degree-hours
            cooling_multiplier = min(cooling_dh / 30.0, 4.0)  # Cap at 4x
            cooling_load = (base_hvac_kw * cooling_multiplier) / efficiency_factor
        
        # HVAC typically doesn't run constantly - duty cycle
        # Higher temps = longer runtime
        if heating_load > 0:
            runtime_factor = min(0.3 + (heating_dh / 100.0), 0.9)
            heating_load *= runtime_factor
        
        if cooling_load > 0:
            runtime_factor = min(0.3 + (cooling_dh / 100.0), 0.9)
            cooling_load *= runtime_factor
        
        # Add noise
        if heating_load > 0:
            heating_load *= self.rng.normal(1.0, 0.15)
        if cooling_load > 0:
            cooling_load *= self.rng.normal(1.0, 0.15)
        
        return max(0, heating_load), max(0, cooling_load)
    
    def apply_seasonal_variation(
        self,
        base_load: float,
        month: int,
        climate_zone: str
    ) -> float:
        """
        Apply seasonal variation to non-HVAC loads.
        
        Args:
            base_load: Base load in kWh
            month: Month (1-12)
            climate_zone: Climate zone
            
        Returns:
            Adjusted load in kWh
        """
        # Seasonal multipliers (more usage in winter/summer, less in spring/fall)
        # Winter: more lighting, holidays
        # Summer: more refrigeration, fans
        seasonal_curve = {
            1: 1.15, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.95, 6: 1.0,
            7: 1.05, 8: 1.05, 9: 0.95, 10: 0.95, 11: 1.05, 12: 1.20
        }
        
        multiplier = seasonal_curve[month]
        
        # Hot climates use more in summer (fans, refrigerators work harder)
        if climate_zone in ["hot_humid", "hot_dry"] and month in [6, 7, 8]:
            multiplier *= 1.1
        
        return base_load * multiplier
    
    def generate_hourly_consumption(
        self,
        profile: CustomerProfile,
        timestamp: datetime
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Generate complete hourly consumption breakdown.
        
        Args:
            profile: Customer profile
            timestamp: Timestamp for this hour
            
        Returns:
            Tuple of (baseline_load, hvac_heating, hvac_cooling, 
                     outdoor_temp, cooling_dh, heating_dh)
        """
        hour = timestamp.hour
        month = timestamp.month
        day = timestamp.day
        is_weekend = timestamp.weekday() >= 5
        
        # Generate outdoor temperature
        outdoor_temp = self.generate_hourly_temperature(
            profile.climate_zone, month, hour, day
        )
        
        # Calculate degree-hours
        cooling_dh, heating_dh = self.calculate_degree_hours(outdoor_temp)
        
        # Calculate baseline load
        baseline = self.calculate_baseline_load(profile, hour, is_weekend)
        
        # Apply seasonal variation
        baseline = self.apply_seasonal_variation(baseline, month, profile.climate_zone)
        
        # Calculate HVAC loads
        hvac_heating, hvac_cooling = self.calculate_hvac_load(
            profile, outdoor_temp, hour
        )
        
        return (
            round(baseline, 2),
            round(hvac_heating, 2),
            round(hvac_cooling, 2),
            round(outdoor_temp, 1),
            round(cooling_dh, 2),
            round(heating_dh, 2)
        )


def generate_consumption_timeseries(
    profile: CustomerProfile,
    start_date: datetime,
    months: int = 12,
    seed: int = 42
) -> List[Tuple[datetime, float, float, float, float, float, float]]:
    """
    Generate time series of hourly consumption data.
    
    Args:
        profile: Customer profile
        start_date: Starting date
        months: Number of months to generate
        seed: Random seed
        
    Returns:
        List of tuples (timestamp, baseline, hvac_heat, hvac_cool, temp, cool_dh, heat_dh)
    """
    engine = ConsumptionPatternEngine(seed=seed)
    timeseries = []
    
    current_time = start_date
    end_time = start_date + timedelta(days=months * 30)
    
    while current_time < end_time:
        consumption = engine.generate_hourly_consumption(profile, current_time)
        timeseries.append((current_time,) + consumption)
        current_time += timedelta(hours=1)
    
    return timeseries


if __name__ == "__main__":
    # Test the consumption pattern engine
    from .customer_profiles import generate_customer_profiles
    
    # Generate a sample profile
    profiles = generate_customer_profiles(count=1, seed=42)
    profile = profiles[0]
    
    print(f"Testing Consumption Pattern Engine")
    print(f"=" * 60)
    print(f"Profile: {profile.customer_id}")
    print(f"Home: {profile.home_size_sqft} sqft, {profile.climate_zone} climate")
    print(f"HVAC: {profile.hvac_type}, {profile.hvac_efficiency_seer} SEER")
    print(f"Efficiency: {profile.energy_efficiency_level}")
    print()
    
    # Generate one week of data
    start = datetime(2024, 7, 15)  # Summer week
    timeseries = generate_consumption_timeseries(profile, start, months=1, seed=42)
    
    # Show sample hours
    print(f"Sample Hours (July - Summer):")
    print(f"=" * 60)
    print(f"{'Time':<20} {'Base':<8} {'Heat':<8} {'Cool':<8} {'Temp':<6}")
    print(f"-" * 60)
    
    for i in [0, 6, 12, 18, 24, 168, 336]:  # Various times
        if i < len(timeseries):
            ts, base, heat, cool, temp, cdh, hdh = timeseries[i]
            print(f"{ts.strftime('%Y-%m-%d %H:00'):<20} "
                  f"{base:<8.2f} {heat:<8.2f} {cool:<8.2f} {temp:<6.1f}°F")
    
    # Calculate statistics
    baselines = [t[1] for t in timeseries]
    heating = [t[2] for t in timeseries]
    cooling = [t[3] for t in timeseries]
    temps = [t[4] for t in timeseries]
    
    print()
    print(f"Statistics (1 month):")
    print(f"=" * 60)
    print(f"Baseline Load:  {np.mean(baselines):.2f} ± {np.std(baselines):.2f} kWh/hr")
    print(f"Heating Load:   {np.mean(heating):.2f} ± {np.std(heating):.2f} kWh/hr")
    print(f"Cooling Load:   {np.mean(cooling):.2f} ± {np.std(cooling):.2f} kWh/hr")
    print(f"Temperature:    {np.mean(temps):.1f} ± {np.std(temps):.1f} °F")
    print(f"Total Avg:      {np.mean(baselines) + np.mean(heating) + np.mean(cooling):.2f} kWh/hr")
