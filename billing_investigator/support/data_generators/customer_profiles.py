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
Customer Profile Generator
Generates diverse synthetic customer profiles with realistic attributes.
"""

import numpy as np
from typing import List
from .schema import CustomerProfile


class CustomerProfileGenerator:
    """Generates synthetic customer profiles with realistic distributions."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the profile generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        
        # Distribution parameters
        self.home_size_mean = 2500  # sqft
        self.home_size_std = 800
        self.home_size_min = 1000
        self.home_size_max = 5000
        
        # Equipment ownership rates
        self.ev_ownership_rate = 0.30  # 30% have EVs
        self.pool_ownership_rate = 0.20  # 20% have pools
        self.solar_ownership_rate = 0.15  # 15% have solar
        
        # Climate zones with weights (representative US distribution)
        self.climate_zones = {
            "hot_humid": 0.25,      # Southeast, Gulf Coast
            "hot_dry": 0.15,        # Southwest
            "mixed": 0.40,          # Mid-Atlantic, Midwest
            "cold": 0.20            # Northeast, North
        }
        
        # HVAC types by climate zone
        self.hvac_by_climate = {
            "hot_humid": ["central_ac_gas_heat", "heat_pump"],
            "hot_dry": ["central_ac_gas_heat", "heat_pump"],
            "mixed": ["central_ac_gas_heat", "heat_pump", "electric_resistance"],
            "cold": ["central_ac_gas_heat", "electric_resistance"]
        }
        
        # Energy efficiency distribution
        self.efficiency_levels = ["low", "medium", "high"]
        self.efficiency_weights = [0.25, 0.50, 0.25]  # Normal distribution
    
    def generate_home_size(self) -> int:
        """Generate home size following normal distribution with constraints."""
        size = self.rng.normal(self.home_size_mean, self.home_size_std)
        size = np.clip(size, self.home_size_min, self.home_size_max)
        return int(np.round(size / 100) * 100)  # Round to nearest 100 sqft
    
    def generate_climate_zone(self) -> str:
        """Select climate zone based on distribution."""
        zones = list(self.climate_zones.keys())
        weights = list(self.climate_zones.values())
        return self.rng.choice(zones, p=weights)
    
    def generate_hvac_type(self, climate_zone: str) -> str:
        """Select HVAC type appropriate for climate zone."""
        options = self.hvac_by_climate[climate_zone]
        return self.rng.choice(options)
    
    def generate_hvac_efficiency(self, hvac_type: str) -> float:
        """Generate HVAC efficiency (SEER rating)."""
        if hvac_type == "none":
            return 0.0
        
        # Modern units: 13-25 SEER, with mean around 16
        seer = self.rng.normal(16.0, 3.0)
        return round(np.clip(seer, 10.0, 25.0), 1)
    
    def generate_ev_ownership(self) -> tuple[bool, float]:
        """
        Generate EV ownership and battery capacity.
        
        Returns:
            Tuple of (ev_owned, battery_capacity_kwh)
        """
        ev_owned = self.rng.random() < self.ev_ownership_rate
        
        if ev_owned:
            # Common EV battery sizes: 40-100 kWh
            # Distribution: smaller batteries (50-65) more common, larger (75-100) less common
            if self.rng.random() < 0.6:
                # Smaller/mid-size EVs (e.g., Leaf, Bolt, Model 3 SR)
                capacity = self.rng.uniform(40.0, 65.0)
            else:
                # Larger EVs (e.g., Model S, Model X, Rivian)
                capacity = self.rng.uniform(70.0, 100.0)
            return True, round(capacity, 1)
        
        return False, 0.0
    
    def generate_pool_ownership(self, climate_zone: str) -> bool:
        """Generate pool ownership based on climate."""
        # Higher pool ownership in hot climates
        if climate_zone in ["hot_humid", "hot_dry"]:
            rate = self.pool_ownership_rate * 1.5
        else:
            rate = self.pool_ownership_rate * 0.5
        
        return self.rng.random() < rate
    
    def generate_solar_installation(self, climate_zone: str) -> tuple[bool, float]:
        """
        Generate solar installation status and capacity.
        
        Returns:
            Tuple of (solar_installed, capacity_kw)
        """
        # Higher solar adoption in sunny climates
        if climate_zone in ["hot_dry", "hot_humid"]:
            rate = self.solar_ownership_rate * 1.5
        else:
            rate = self.solar_ownership_rate * 0.7
        
        solar_installed = self.rng.random() < rate
        
        if solar_installed:
            # Typical residential: 5-10 kW, some larger 10-15 kW
            capacity = self.rng.uniform(5.0, 12.0)
            return True, round(capacity, 1)
        
        return False, 0.0
    
    def generate_occupancy(self, home_size: int) -> int:
        """Generate occupancy count based on home size."""
        # Correlation: larger homes tend to have more occupants
        if home_size < 1500:
            mean_occupancy = 1.5
        elif home_size < 2500:
            mean_occupancy = 2.5
        elif home_size < 3500:
            mean_occupancy = 3.5
        else:
            mean_occupancy = 4.0
        
        occupancy = self.rng.normal(mean_occupancy, 0.8)
        return int(np.clip(np.round(occupancy), 1, 6))
    
    def generate_efficiency_level(self, home_size: int) -> str:
        """Generate energy efficiency level."""
        # Newer/smaller homes tend to be more efficient
        if home_size < 2000:
            weights = [0.15, 0.45, 0.40]  # Skew towards higher efficiency
        else:
            weights = [0.30, 0.50, 0.20]  # More low efficiency
        
        return self.rng.choice(self.efficiency_levels, p=weights)
    
    def generate_profile(self, customer_id: str) -> CustomerProfile:
        """
        Generate a single customer profile with realistic attributes.
        
        Args:
            customer_id: Unique customer identifier
            
        Returns:
            CustomerProfile object
        """
        # Generate interdependent attributes
        home_size = self.generate_home_size()
        climate_zone = self.generate_climate_zone()
        hvac_type = self.generate_hvac_type(climate_zone)
        hvac_efficiency = self.generate_hvac_efficiency(hvac_type)
        ev_owned, ev_capacity = self.generate_ev_ownership()
        pool_equipped = self.generate_pool_ownership(climate_zone)
        solar_installed, solar_capacity = self.generate_solar_installation(climate_zone)
        occupancy = self.generate_occupancy(home_size)
        efficiency_level = self.generate_efficiency_level(home_size)
        
        return CustomerProfile(
            customer_id=customer_id,
            home_size_sqft=home_size,
            climate_zone=climate_zone,
            hvac_type=hvac_type,
            hvac_efficiency_seer=hvac_efficiency,
            ev_owned=ev_owned,
            ev_battery_capacity_kwh=ev_capacity,
            pool_equipped=pool_equipped,
            solar_installed=solar_installed,
            solar_capacity_kw=solar_capacity,
            occupancy_count=occupancy,
            energy_efficiency_level=efficiency_level
        )
    
    def generate_profiles(self, count: int = 15) -> List[CustomerProfile]:
        """
        Generate multiple customer profiles.
        
        Args:
            count: Number of profiles to generate (default: 15)
            
        Returns:
            List of CustomerProfile objects
        """
        profiles = []
        
        for i in range(count):
            customer_id = f"CUST_{i+1:03d}"
            profile = self.generate_profile(customer_id)
            profiles.append(profile)
        
        return profiles


def generate_customer_profiles(count: int = 15, seed: int = 42) -> List[CustomerProfile]:
    """
    Convenience function to generate customer profiles.
    
    Args:
        count: Number of profiles to generate (10-20 recommended)
        seed: Random seed for reproducibility
        
    Returns:
        List of CustomerProfile objects
        
    Example:
        >>> profiles = generate_customer_profiles(count=15, seed=42)
        >>> print(f"Generated {len(profiles)} profiles")
        >>> print(f"EV owners: {sum(p.ev_owned for p in profiles)}")
    """
    generator = CustomerProfileGenerator(seed=seed)
    return generator.generate_profiles(count=count)


def print_profile_summary(profiles: List[CustomerProfile]) -> None:
    """
    Print summary statistics for generated profiles.
    
    Args:
        profiles: List of CustomerProfile objects
    """
    ev_count = sum(p.ev_owned for p in profiles)
    pool_count = sum(p.pool_equipped for p in profiles)
    solar_count = sum(p.solar_installed for p in profiles)
    
    home_sizes = [p.home_size_sqft for p in profiles]
    occupancies = [p.occupancy_count for p in profiles]
    
    climate_counts = {}
    for p in profiles:
        climate_counts[p.climate_zone] = climate_counts.get(p.climate_zone, 0) + 1
    
    efficiency_counts = {}
    for p in profiles:
        efficiency_counts[p.energy_efficiency_level] = efficiency_counts.get(p.energy_efficiency_level, 0) + 1
    
    print(f"Customer Profile Summary ({len(profiles)} profiles)")
    print("=" * 50)
    print(f"\nEquipment Ownership:")
    print(f"  EV Owners: {ev_count} ({ev_count/len(profiles)*100:.1f}%)")
    print(f"  Pool Equipped: {pool_count} ({pool_count/len(profiles)*100:.1f}%)")
    print(f"  Solar Installed: {solar_count} ({solar_count/len(profiles)*100:.1f}%)")
    
    print(f"\nHome Size:")
    print(f"  Mean: {np.mean(home_sizes):.0f} sqft")
    print(f"  Std: {np.std(home_sizes):.0f} sqft")
    print(f"  Range: {min(home_sizes)} - {max(home_sizes)} sqft")
    
    print(f"\nOccupancy:")
    print(f"  Mean: {np.mean(occupancies):.1f} people")
    print(f"  Range: {min(occupancies)} - {max(occupancies)} people")
    
    print(f"\nClimate Zones:")
    for zone, count in sorted(climate_counts.items()):
        print(f"  {zone}: {count} ({count/len(profiles)*100:.1f}%)")
    
    print(f"\nEnergy Efficiency:")
    for level, count in sorted(efficiency_counts.items()):
        print(f"  {level}: {count} ({count/len(profiles)*100:.1f}%)")


if __name__ == "__main__":
    # Generate sample profiles
    profiles = generate_customer_profiles(count=15, seed=42)
    
    # Print summary
    print_profile_summary(profiles)
    
    # Print first few profiles
    print("\n" + "=" * 50)
    print("Sample Profiles:")
    print("=" * 50)
    for profile in profiles[:3]:
        print(f"\n{profile.customer_id}:")
        for key, value in profile.to_dict().items():
            print(f"  {key}: {value}")
