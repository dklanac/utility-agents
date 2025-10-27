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
Anomaly Injection System
Injects specific detectable anomalies into consumption data.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from .schema import CustomerProfile


@dataclass
class Anomaly:
    """Represents an anomaly in energy consumption."""
    anomaly_type: str
    customer_id: str
    start_date: datetime
    magnitude: float
    description: str
    affected_component: str
    duration_days: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for export."""
        data = asdict(self)
        data['start_date'] = self.start_date.isoformat()
        return data


class AnomalyInjector:
    """Injects detectable anomalies into consumption data."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the anomaly injector.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.anomalies: List[Anomaly] = []
        
        # Anomaly type parameters
        self.anomaly_types = {
            "thermostat_change": {
                "magnitude_range": (0.20, 0.40),  # 20-40% change
                "duration_range": (14, 28),  # 2-4 weeks to ramp up
                "components": ["hvac_heating", "hvac_cooling"]
            },
            "new_ev_charger": {
                "magnitude_range": (10.0, 15.0),  # 10-15 kWh per session
                "duration_range": None,  # Permanent
                "components": ["ev_charging"]
            },
            "pool_equipment_failure": {
                "magnitude_range": (0.30, 0.50),  # 30-50% increase
                "duration_range": None,  # Until "fixed" (end of data)
                "components": ["pool_equipment"]
            },
            "appliance_upgrade": {
                "magnitude_range": (-0.25, -0.15),  # 15-25% decrease
                "duration_range": None,  # Permanent improvement
                "components": ["major_appliances", "water_heater"]
            }
        }
    
    def select_anomalies_for_profile(
        self,
        profile: CustomerProfile,
        data_start_date: datetime,
        data_end_date: datetime
    ) -> List[Anomaly]:
        """
        Select appropriate anomalies for a customer profile.
        
        Args:
            profile: Customer profile
            data_start_date: Start of data generation period
            data_end_date: End of data generation period
            
        Returns:
            List of anomalies to inject
        """
        selected_anomalies = []
        
        # 30-50% of profiles get anomalies
        if self.rng.random() > 0.40:
            return selected_anomalies
        
        # Determine number of anomalies (1-3)
        num_anomalies = self.rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        
        # Available anomaly types based on profile
        available_types = []
        
        # Thermostat change - everyone with HVAC
        if profile.hvac_type != "none":
            available_types.append("thermostat_change")
        
        # New EV charger - only if they DON'T have EV already
        if not profile.ev_owned:
            available_types.append("new_ev_charger")
        
        # Pool failure - only if they have a pool
        if profile.pool_equipped:
            available_types.append("pool_equipment_failure")
        
        # Appliance upgrade - everyone
        available_types.append("appliance_upgrade")
        
        if not available_types:
            return selected_anomalies
        
        # Select anomalies (without replacement if possible)
        num_to_select = min(num_anomalies, len(available_types))
        selected_types = self.rng.choice(
            available_types, 
            size=num_to_select, 
            replace=False
        ).tolist()
        
        # Generate anomaly instances
        data_duration = (data_end_date - data_start_date).days
        
        for anomaly_type in selected_types:
            # Random start date (not too early, leave room to see baseline)
            # Start between 20% and 70% through the data period
            earliest_start = data_start_date + timedelta(days=int(data_duration * 0.20))
            latest_start = data_start_date + timedelta(days=int(data_duration * 0.70))
            
            days_range = (latest_start - earliest_start).days
            start_offset = int(self.rng.integers(0, max(1, days_range)))
            start_date = earliest_start + timedelta(days=start_offset)
            
            # Generate magnitude
            params = self.anomaly_types[anomaly_type]
            magnitude = self.rng.uniform(*params["magnitude_range"])
            
            # Select affected component
            component = self.rng.choice(params["components"])
            
            # Duration (if applicable)
            duration = None
            if params["duration_range"]:
                duration = self.rng.integers(*params["duration_range"])
            
            # Create description
            descriptions = {
                "thermostat_change": f"Thermostat adjusted {'+' if magnitude > 0 else ''}{magnitude*100:.0f}% affecting {component}",
                "new_ev_charger": f"New EV charger installed, adding {magnitude:.1f} kWh nightly",
                "pool_equipment_failure": f"Pool equipment malfunction, +{magnitude*100:.0f}% consumption",
                "appliance_upgrade": f"Appliance upgraded to energy-efficient model, {magnitude*100:.0f}% reduction"
            }
            
            anomaly = Anomaly(
                anomaly_type=anomaly_type,
                customer_id=profile.customer_id,
                start_date=start_date,
                magnitude=magnitude,
                description=descriptions[anomaly_type],
                affected_component=component,
                duration_days=duration
            )
            
            selected_anomalies.append(anomaly)
            self.anomalies.append(anomaly)
        
        return selected_anomalies
    
    def apply_thermostat_change(
        self,
        anomaly: Anomaly,
        timestamp: datetime,
        current_value: float
    ) -> float:
        """
        Apply thermostat change anomaly with gradual ramp-up.

        Args:
            anomaly: Anomaly definition
            timestamp: Current timestamp
            current_value: Current HVAC consumption

        Returns:
            Modified consumption value
        """
        if timestamp < anomaly.start_date:
            return current_value

        # Gradual ramp-up over duration_days
        days_since_start = (timestamp - anomaly.start_date).days

        if anomaly.duration_days and days_since_start < anomaly.duration_days:
            # Linear ramp-up
            ramp_factor = days_since_start / anomaly.duration_days
            effective_magnitude = anomaly.magnitude * ramp_factor
        else:
            # Full magnitude reached
            effective_magnitude = anomaly.magnitude

        modified_value = current_value * (1.0 + effective_magnitude)

        # Enforce schema limits: hvac_heating max 15.0, hvac_cooling max 20.0
        if anomaly.affected_component == "hvac_heating":
            return min(modified_value, 15.0)
        elif anomaly.affected_component == "hvac_cooling":
            return min(modified_value, 20.0)
        else:
            return modified_value
    
    def apply_new_ev_charger(
        self,
        anomaly: Anomaly,
        timestamp: datetime,
        current_value: float,
        hour: int
    ) -> float:
        """
        Apply new EV charger anomaly (night charging).

        Args:
            anomaly: Anomaly definition
            timestamp: Current timestamp
            current_value: Current EV charging (should be 0)
            hour: Hour of day

        Returns:
            Modified consumption value
        """
        if timestamp < anomaly.start_date:
            return current_value

        # EV charging occurs at night (10pm-6am), not every night
        # Assume charging 4-5 days per week
        day_of_week = timestamp.weekday()
        charges_today = day_of_week < 5  # Mon-Fri charging

        if not charges_today:
            return current_value

        # Night charging window
        if hour >= 22 or hour < 6:
            # Typical session: 2-4 hours
            # Use day and hour to determine if charging this hour
            session_seed = (timestamp.date().toordinal() * 100 + hour) % 1000
            session_rng = np.random.default_rng(session_seed)

            # About 3-hour window of actual charging
            if hour in [22, 23, 0]:
                # Peak charging hours
                if session_rng.random() < 0.8:
                    modified_value = current_value + anomaly.magnitude
                    # Enforce schema max limit of 12.0 kWh for ev_charging
                    return min(modified_value, 12.0)

        return current_value
    
    def apply_pool_failure(
        self,
        anomaly: Anomaly,
        timestamp: datetime,
        current_value: float
    ) -> float:
        """
        Apply pool equipment failure anomaly.

        Args:
            anomaly: Anomaly definition
            timestamp: Current timestamp
            current_value: Current pool equipment consumption

        Returns:
            Modified consumption value
        """
        if timestamp < anomaly.start_date:
            return current_value

        # Sudden increase (malfunction causes inefficiency)
        modified_value = current_value * (1.0 + anomaly.magnitude)

        # Enforce schema max limit of 8.0 kWh for pool_equipment
        return min(modified_value, 8.0)
    
    def apply_appliance_upgrade(
        self,
        anomaly: Anomaly,
        timestamp: datetime,
        current_value: float
    ) -> float:
        """
        Apply appliance upgrade anomaly (efficiency improvement).

        Args:
            anomaly: Anomaly definition
            timestamp: Current timestamp
            current_value: Current appliance consumption

        Returns:
            Modified consumption value
        """
        if timestamp < anomaly.start_date:
            return current_value

        # Sudden decrease (new efficient appliance)
        modified_value = current_value * (1.0 + anomaly.magnitude)  # magnitude is negative

        # Enforce schema limits based on component
        if anomaly.affected_component == "major_appliances":
            return max(0, min(modified_value, 8.0))
        elif anomaly.affected_component == "water_heater":
            return max(0, min(modified_value, 5.0))
        else:
            return max(0, modified_value)
    
    def apply_anomalies_to_data(
        self,
        profile_anomalies: List[Anomaly],
        timestamp: datetime,
        consumption_data: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply all relevant anomalies to consumption data.
        
        Args:
            profile_anomalies: List of anomalies for this profile
            timestamp: Current timestamp
            consumption_data: Dict with consumption values by component
            
        Returns:
            Modified consumption data
        """
        modified_data = consumption_data.copy()
        hour = timestamp.hour
        
        for anomaly in profile_anomalies:
            component = anomaly.affected_component
            
            if component not in modified_data:
                continue
            
            current_value = modified_data[component]
            
            # Apply appropriate modification based on type
            if anomaly.anomaly_type == "thermostat_change":
                modified_data[component] = self.apply_thermostat_change(
                    anomaly, timestamp, current_value
                )
            
            elif anomaly.anomaly_type == "new_ev_charger":
                modified_data[component] = self.apply_new_ev_charger(
                    anomaly, timestamp, current_value, hour
                )
            
            elif anomaly.anomaly_type == "pool_equipment_failure":
                modified_data[component] = self.apply_pool_failure(
                    anomaly, timestamp, current_value
                )
            
            elif anomaly.anomaly_type == "appliance_upgrade":
                modified_data[component] = self.apply_appliance_upgrade(
                    anomaly, timestamp, current_value
                )
        
        return modified_data
    
    def get_anomalies(self) -> List[Anomaly]:
        """Get all generated anomalies."""
        return self.anomalies
    
    def get_anomalies_for_customer(self, customer_id: str) -> List[Anomaly]:
        """Get anomalies for a specific customer."""
        return [a for a in self.anomalies if a.customer_id == customer_id]


if __name__ == "__main__":
    # Test anomaly injection
    from .customer_profiles import generate_customer_profiles
    from datetime import datetime
    
    print("Testing Anomaly Injection System")
    print("=" * 70)
    
    # Generate profiles
    profiles = generate_customer_profiles(count=15, seed=42)
    
    # Create injector
    injector = AnomalyInjector(seed=42)
    
    # Data period
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Select anomalies for each profile
    print("Selecting anomalies for profiles...\n")
    
    for profile in profiles:
        anomalies = injector.select_anomalies_for_profile(profile, start_date, end_date)
        
        if anomalies:
            print(f"{profile.customer_id} ({profile.home_size_sqft} sqft, "
                  f"EV: {profile.ev_owned}, Pool: {profile.pool_equipped}):")
            for a in anomalies:
                print(f"  - {a.anomaly_type}: {a.description}")
                print(f"    Start: {a.start_date.strftime('%Y-%m-%d')}, "
                      f"Component: {a.affected_component}")
            print()
    
    # Summary statistics
    all_anomalies = injector.get_anomalies()
    print(f"\nAnomaly Summary:")
    print(f"=" * 70)
    print(f"Total profiles: {len(profiles)}")
    print(f"Profiles with anomalies: {len(set(a.customer_id for a in all_anomalies))}")
    print(f"Total anomalies: {len(all_anomalies)}")
    
    # Count by type
    type_counts = {}
    for a in all_anomalies:
        type_counts[a.anomaly_type] = type_counts.get(a.anomaly_type, 0) + 1
    
    print(f"\nAnomalies by type:")
    for anomaly_type, count in sorted(type_counts.items()):
        print(f"  {anomaly_type}: {count}")
    
    # Test applying an anomaly
    if all_anomalies:
        print(f"\n" + "=" * 70)
        print(f"Testing anomaly application:")
        print(f"=" * 70)
        
        test_anomaly = all_anomalies[0]
        print(f"\nTest anomaly: {test_anomaly.description}")
        print(f"Start date: {test_anomaly.start_date.strftime('%Y-%m-%d')}")
        
        # Test before and after
        before_date = test_anomaly.start_date - timedelta(days=7)
        after_date = test_anomaly.start_date + timedelta(days=30)
        
        test_value = 5.0  # kWh
        
        before_result = injector.apply_thermostat_change(test_anomaly, before_date, test_value)
        after_result = injector.apply_thermostat_change(test_anomaly, after_date, test_value)
        
        print(f"\nOriginal value: {test_value:.2f} kWh")
        print(f"Before anomaly ({before_date.strftime('%Y-%m-%d')}): {before_result:.2f} kWh")
        print(f"After anomaly ({after_date.strftime('%Y-%m-%d')}): {after_result:.2f} kWh")
        print(f"Change: {((after_result - test_value) / test_value * 100):.1f}%")
