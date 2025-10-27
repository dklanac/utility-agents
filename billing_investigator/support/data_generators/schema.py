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
Enhanced Energy Consumption Schema
Defines the data structure for synthetic utility usage data with disaggregated loads.
"""

from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime


# BigQuery Table Schema Definition
ENERGY_CONSUMPTION_SCHEMA = {
    # Primary Keys and Temporal
    "customer_id": {
        "type": "STRING",
        "mode": "REQUIRED",
        "description": "Unique customer identifier"
    },
    "timestamp": {
        "type": "TIMESTAMP",
        "mode": "REQUIRED",
        "description": "Hourly timestamp in UTC (ISO 8601 format)"
    },
    "billing_period_id": {
        "type": "STRING",
        "mode": "NULLABLE",
        "description": "Billing period identifier (YYYY-MM format)"
    },
    
    # Total Energy Consumption
    "total_usage_kwh": {
        "type": "FLOAT64",
        "mode": "REQUIRED",
        "description": "Total energy consumption in kilowatt-hours",
        "range": [0.0, 50.0],  # Typical hourly range for residential
        "precision": 2
    },
    
    # Disaggregated Energy Loads - HVAC
    "hvac_heating_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "HVAC heating load in kilowatt-hours",
        "range": [0.0, 15.0],
        "precision": 2
    },
    "hvac_cooling_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "HVAC cooling load in kilowatt-hours",
        "range": [0.0, 20.0],
        "precision": 2
    },
    
    # Disaggregated Energy Loads - Major Equipment
    "ev_charging_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Electric vehicle charging load in kilowatt-hours",
        "range": [0.0, 12.0],  # Typical Level 2 charger
        "precision": 2
    },
    "water_heater_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Water heater energy consumption in kilowatt-hours",
        "range": [0.0, 5.0],
        "precision": 2
    },
    "pool_equipment_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Pool pump and heater energy consumption in kilowatt-hours",
        "range": [0.0, 8.0],
        "precision": 2
    },
    "major_appliances_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Major appliances (refrigerator, washer, dryer, dishwasher, oven) in kilowatt-hours",
        "range": [0.0, 8.0],
        "precision": 2
    },
    "other_loads_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Other miscellaneous loads (lighting, electronics, etc.) in kilowatt-hours",
        "range": [0.0, 5.0],
        "precision": 2
    },
    
    # Environmental Context
    "outdoor_temperature_f": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Outdoor temperature in Fahrenheit",
        "range": [15.0, 95.0],  # Realistic range for typical utility billing scenarios
        "precision": 1
    },
    "cooling_degree_hours": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Cooling degree-hours (base 65°F) - measure of cooling demand",
        "range": [0.0, 30.0],
        "precision": 2
    },
    "heating_degree_hours": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Heating degree-hours (base 65°F) - measure of heating demand",
        "range": [0.0, 30.0],
        "precision": 2
    },
}

# Customer Profile Metadata Schema
CUSTOMER_PROFILE_SCHEMA = {
    "customer_id": {
        "type": "STRING",
        "mode": "REQUIRED",
        "description": "Unique customer identifier"
    },
    "home_size_sqft": {
        "type": "INT64",
        "mode": "REQUIRED",
        "description": "Home size in square feet",
        "range": [1000, 5000]
    },
    "climate_zone": {
        "type": "STRING",
        "mode": "REQUIRED",
        "description": "Climate zone (hot_humid, hot_dry, mixed, cold)",
        "enum": ["hot_humid", "hot_dry", "mixed", "cold"]
    },
    "hvac_type": {
        "type": "STRING",
        "mode": "REQUIRED",
        "description": "HVAC system type",
        "enum": ["central_ac_gas_heat", "heat_pump", "electric_resistance", "none"]
    },
    "hvac_efficiency_seer": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "HVAC cooling efficiency (SEER rating)",
        "range": [10.0, 25.0]
    },
    "ev_owned": {
        "type": "BOOL",
        "mode": "REQUIRED",
        "description": "Electric vehicle ownership status"
    },
    "ev_battery_capacity_kwh": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "EV battery capacity in kilowatt-hours",
        "range": [40.0, 100.0]
    },
    "pool_equipped": {
        "type": "BOOL",
        "mode": "REQUIRED",
        "description": "Pool equipment installed"
    },
    "solar_installed": {
        "type": "BOOL",
        "mode": "REQUIRED",
        "description": "Solar panels installed (net metering)"
    },
    "solar_capacity_kw": {
        "type": "FLOAT64",
        "mode": "NULLABLE",
        "description": "Solar system capacity in kilowatts",
        "range": [3.0, 15.0]
    },
    "occupancy_count": {
        "type": "INT64",
        "mode": "REQUIRED",
        "description": "Number of occupants",
        "range": [1, 6]
    },
    "energy_efficiency_level": {
        "type": "STRING",
        "mode": "REQUIRED",
        "description": "Overall energy efficiency level",
        "enum": ["low", "medium", "high"]
    },
}


@dataclass
class EnergyDataRow:
    """Represents a single hourly energy consumption record."""
    customer_id: str
    timestamp: datetime
    billing_period_id: str
    total_usage_kwh: float
    hvac_heating_kwh: float = 0.0
    hvac_cooling_kwh: float = 0.0
    ev_charging_kwh: float = 0.0
    water_heater_kwh: float = 0.0
    pool_equipment_kwh: float = 0.0
    major_appliances_kwh: float = 0.0
    other_loads_kwh: float = 0.0
    outdoor_temperature_f: float = 65.0
    cooling_degree_hours: float = 0.0
    heating_degree_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/BigQuery export."""
        return {
            "customer_id": self.customer_id,
            "timestamp": self.timestamp.isoformat(),
            "billing_period_id": self.billing_period_id,
            "total_usage_kwh": round(self.total_usage_kwh, 2),
            "hvac_heating_kwh": round(self.hvac_heating_kwh, 2),
            "hvac_cooling_kwh": round(self.hvac_cooling_kwh, 2),
            "ev_charging_kwh": round(self.ev_charging_kwh, 2),
            "water_heater_kwh": round(self.water_heater_kwh, 2),
            "pool_equipment_kwh": round(self.pool_equipment_kwh, 2),
            "major_appliances_kwh": round(self.major_appliances_kwh, 2),
            "other_loads_kwh": round(self.other_loads_kwh, 2),
            "outdoor_temperature_f": round(self.outdoor_temperature_f, 1),
            "cooling_degree_hours": round(self.cooling_degree_hours, 2),
            "heating_degree_hours": round(self.heating_degree_hours, 2),
        }
    
    def validate(self) -> bool:
        """Validate data integrity."""
        # Check no negative values
        if any(v < 0 for k, v in self.to_dict().items() 
               if k.endswith('_kwh') or k.endswith('_f') or k.endswith('_hours')):
            return False
        
        # Check total equals sum of components (within 1% tolerance)
        components_sum = (
            self.hvac_heating_kwh + self.hvac_cooling_kwh +
            self.ev_charging_kwh + self.water_heater_kwh +
            self.pool_equipment_kwh + self.major_appliances_kwh +
            self.other_loads_kwh
        )
        tolerance = self.total_usage_kwh * 0.01
        if abs(self.total_usage_kwh - components_sum) > max(tolerance, 0.01):
            return False
        
        return True


@dataclass
class CustomerProfile:
    """Represents customer profile metadata."""
    customer_id: str
    home_size_sqft: int
    climate_zone: str
    hvac_type: str
    hvac_efficiency_seer: float
    ev_owned: bool
    ev_battery_capacity_kwh: float = 0.0
    pool_equipped: bool = False
    solar_installed: bool = False
    solar_capacity_kw: float = 0.0
    occupancy_count: int = 2
    energy_efficiency_level: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/BigQuery export."""
        return {
            "customer_id": self.customer_id,
            "home_size_sqft": self.home_size_sqft,
            "climate_zone": self.climate_zone,
            "hvac_type": self.hvac_type,
            "hvac_efficiency_seer": round(self.hvac_efficiency_seer, 1) if self.hvac_efficiency_seer else None,
            "ev_owned": self.ev_owned,
            "ev_battery_capacity_kwh": round(self.ev_battery_capacity_kwh, 1) if self.ev_owned else None,
            "pool_equipped": self.pool_equipped,
            "solar_installed": self.solar_installed,
            "solar_capacity_kw": round(self.solar_capacity_kw, 1) if self.solar_installed else None,
            "occupancy_count": self.occupancy_count,
            "energy_efficiency_level": self.energy_efficiency_level,
        }


# CSV Column Order (for consistent export)
ENERGY_CSV_COLUMNS = [
    "customer_id",
    "timestamp",
    "billing_period_id",
    "total_usage_kwh",
    "hvac_heating_kwh",
    "hvac_cooling_kwh",
    "ev_charging_kwh",
    "water_heater_kwh",
    "pool_equipment_kwh",
    "major_appliances_kwh",
    "other_loads_kwh",
    "outdoor_temperature_f",
    "cooling_degree_hours",
    "heating_degree_hours",
]

CUSTOMER_CSV_COLUMNS = [
    "customer_id",
    "home_size_sqft",
    "climate_zone",
    "hvac_type",
    "hvac_efficiency_seer",
    "ev_owned",
    "ev_battery_capacity_kwh",
    "pool_equipped",
    "solar_installed",
    "solar_capacity_kw",
    "occupancy_count",
    "energy_efficiency_level",
]
