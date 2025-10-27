# Enhanced Energy Consumption Schema

This document describes the data schema for synthetic utility usage data with disaggregated energy loads.

## Energy Consumption Table

**Table Name:** `energy_consumption`

### Column Specifications

| Column Name | Type | Mode | Description | Range/Values |
|-------------|------|------|-------------|--------------|
| `customer_id` | STRING | REQUIRED | Unique customer identifier | - |
| `timestamp` | TIMESTAMP | REQUIRED | Hourly timestamp in UTC (ISO 8601) | - |
| `billing_period_id` | STRING | NULLABLE | Billing period identifier | YYYY-MM format |
| `total_usage_kwh` | FLOAT64 | REQUIRED | Total energy consumption (kWh) | 0.0 - 50.0 |
| `hvac_heating_kwh` | FLOAT64 | NULLABLE | HVAC heating load (kWh) | 0.0 - 15.0 |
| `hvac_cooling_kwh` | FLOAT64 | NULLABLE | HVAC cooling load (kWh) | 0.0 - 20.0 |
| `ev_charging_kwh` | FLOAT64 | NULLABLE | EV charging load (kWh) | 0.0 - 12.0 |
| `water_heater_kwh` | FLOAT64 | NULLABLE | Water heater consumption (kWh) | 0.0 - 5.0 |
| `pool_equipment_kwh` | FLOAT64 | NULLABLE | Pool pump/heater consumption (kWh) | 0.0 - 8.0 |
| `major_appliances_kwh` | FLOAT64 | NULLABLE | Major appliances (kWh) | 0.0 - 8.0 |
| `other_loads_kwh` | FLOAT64 | NULLABLE | Other miscellaneous loads (kWh) | 0.0 - 5.0 |
| `outdoor_temperature_f` | FLOAT64 | NULLABLE | Outdoor temperature (°F) | 0.0 - 120.0 |
| `cooling_degree_hours` | FLOAT64 | NULLABLE | Cooling degree-hours (base 65°F) | 0.0 - 30.0 |
| `heating_degree_hours` | FLOAT64 | NULLABLE | Heating degree-hours (base 65°F) | 0.0 - 30.0 |

### Data Integrity Rules

1. **No Negative Values:** All energy and temperature fields must be ≥ 0
2. **Component Sum:** Sum of disaggregated loads must equal `total_usage_kwh` within ±1% tolerance
3. **Precision:** Energy values rounded to 2 decimal places, temperature to 1 decimal place
4. **Completeness:** No gaps in hourly timestamps for each customer

## Customer Profile Table

**Table Name:** `customer_profiles`

### Column Specifications

| Column Name | Type | Mode | Description | Range/Values |
|-------------|------|------|-------------|--------------|
| `customer_id` | STRING | REQUIRED | Unique customer identifier | - |
| `home_size_sqft` | INT64 | REQUIRED | Home size (square feet) | 1000 - 5000 |
| `climate_zone` | STRING | REQUIRED | Climate zone | hot_humid, hot_dry, mixed, cold |
| `hvac_type` | STRING | REQUIRED | HVAC system type | central_ac_gas_heat, heat_pump, electric_resistance, none |
| `hvac_efficiency_seer` | FLOAT64 | NULLABLE | HVAC SEER rating | 10.0 - 25.0 |
| `ev_owned` | BOOL | REQUIRED | EV ownership status | true/false |
| `ev_battery_capacity_kwh` | FLOAT64 | NULLABLE | EV battery capacity (kWh) | 40.0 - 100.0 |
| `pool_equipped` | BOOL | REQUIRED | Pool equipment installed | true/false |
| `solar_installed` | BOOL | REQUIRED | Solar panels installed | true/false |
| `solar_capacity_kw` | FLOAT64 | NULLABLE | Solar capacity (kW) | 3.0 - 15.0 |
| `occupancy_count` | INT64 | REQUIRED | Number of occupants | 1 - 6 |
| `energy_efficiency_level` | STRING | REQUIRED | Energy efficiency level | low, medium, high |

## Typical Energy Load Distributions

### By Component (% of Total)

- **HVAC:** 40-60% (summer), 30-50% (winter)
- **Water Heater:** 15-25%
- **EV Charging:** 0-30% (if owned)
- **Pool Equipment:** 0-15% (if equipped, seasonal)
- **Major Appliances:** 10-20%
- **Other Loads:** 10-20%

### Time-of-Day Patterns

- **Morning Peak:** 7-9 AM (1.5-2x baseline)
- **Midday:** 10 AM - 3 PM (0.8-1.2x baseline)
- **Evening Peak:** 5-8 PM (2-3x baseline)
- **Night:** 11 PM - 6 AM (0.5-0.8x baseline)

### Seasonal Variations

- **Summer (Jun-Aug):** Cooling loads 2-4x winter baseline
- **Winter (Dec-Feb):** Heating loads 1.5-3x summer baseline
- **Spring/Fall:** Moderate loads, reduced HVAC usage

## Usage Examples

```python
from data_generation import EnergyDataRow, CustomerProfile
from datetime import datetime

# Create energy data row
row = EnergyDataRow(
    customer_id="CUST_001",
    timestamp=datetime(2024, 7, 15, 14, 0),
    billing_period_id="2024-07",
    total_usage_kwh=8.5,
    hvac_cooling_kwh=4.2,
    water_heater_kwh=1.5,
    major_appliances_kwh=2.0,
    other_loads_kwh=0.8,
    outdoor_temperature_f=92.5,
    cooling_degree_hours=27.5
)

# Validate
assert row.validate()

# Export
data_dict = row.to_dict()
```

## BigQuery Table Creation

```sql
CREATE TABLE `project.dataset.energy_consumption` (
  customer_id STRING NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  billing_period_id STRING,
  total_usage_kwh FLOAT64 NOT NULL,
  hvac_heating_kwh FLOAT64,
  hvac_cooling_kwh FLOAT64,
  ev_charging_kwh FLOAT64,
  water_heater_kwh FLOAT64,
  pool_equipment_kwh FLOAT64,
  major_appliances_kwh FLOAT64,
  other_loads_kwh FLOAT64,
  outdoor_temperature_f FLOAT64,
  cooling_degree_hours FLOAT64,
  heating_degree_hours FLOAT64
)
PARTITION BY DATE(timestamp)
CLUSTER BY customer_id;
```
