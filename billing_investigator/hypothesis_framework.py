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

"""Hypothesis generation framework for Bill Investigator.

This module defines the structured approach to generating hypotheses
about utility usage changes, including category definitions, analysis
criteria, and example scenarios.
"""

from typing import Dict, List
from dataclasses import dataclass
from .output_formatting import HypothesisCategory


@dataclass
class HypothesisCriteria:
    """Criteria for identifying a specific hypothesis type.

    Attributes:
        category: The hypothesis category
        triggers: Conditions that suggest this hypothesis
        required_data: Data fields needed for analysis
        confidence_factors: Factors that increase confidence
        typical_magnitude: Typical range of impact (percentage)
        example_causes: Common real-world causes
    """
    category: HypothesisCategory
    triggers: List[str]
    required_data: List[str]
    confidence_factors: List[str]
    typical_magnitude: str
    example_causes: List[str]


# Hypothesis Framework (7.1)
HYPOTHESIS_FRAMEWORK = {
    HypothesisCategory.HVAC_INCREASE: HypothesisCriteria(
        category=HypothesisCategory.HVAC_INCREASE,
        triggers=[
            "HVAC cooling/heating usage increased >20%",
            "Temperature correlation shows high sensitivity",
            "Degree-hours increased but not proportionally to usage",
            "Specific hours show HVAC spike patterns"
        ],
        required_data=[
            "hvac_cooling_kwh",
            "hvac_heating_kwh",
            "outdoor_temperature_f",
            "cooling_degree_hours",
            "heating_degree_hours"
        ],
        confidence_factors=[
            "Weather correlation r > 0.7",
            "Consistent increase across multiple days",
            "Degree-hour change doesn't explain full increase",
            "Similar pattern in previous weather extremes"
        ],
        typical_magnitude="20-60% increase in HVAC load",
        example_causes=[
            "Thermostat setting changed (lower in summer, higher in winter)",
            "HVAC system inefficiency or malfunction",
            "Duct leakage or insulation issues",
            "Filter clogged reducing efficiency",
            "Extreme weather conditions"
        ]
    ),

    HypothesisCategory.NEW_LOAD: HypothesisCriteria(
        category=HypothesisCategory.NEW_LOAD,
        triggers=[
            "Sudden sustained increase in specific appliance",
            "New load appears at specific time of day",
            "Increase starts on identifiable date",
            "Load pattern matches known appliance signature"
        ],
        required_data=[
            "ev_charging_kwh",
            "pool_equipment_kwh",
            "major_appliances_kwh",
            "timestamp"
        ],
        confidence_factors=[
            "Load starts abruptly on specific date",
            "Consistent daily pattern (e.g., EV charging at night)",
            "Magnitude matches typical appliance (3-7 kW for EV charger)",
            "No similar load in baseline period"
        ],
        typical_magnitude="10-30% increase in total usage",
        example_causes=[
            "New EV charger (typically 10-15 kWh/day)",
            "Pool equipment added or timer extended",
            "New major appliance (second refrigerator, hot tub)",
            "Space heater or window AC unit added",
            "Home office equipment for work-from-home"
        ]
    ),

    HypothesisCategory.TIME_SHIFT: HypothesisCriteria(
        category=HypothesisCategory.TIME_SHIFT,
        triggers=[
            "Peak usage hour changed",
            "Increased evening or morning usage",
            "Weekend vs weekday pattern changed",
            "Overall total similar but timing different"
        ],
        required_data=[
            "total_usage_kwh",
            "timestamp",
            "hourly usage patterns"
        ],
        confidence_factors=[
            "Clear shift in peak hours (>2 hours difference)",
            "Consistent new pattern across multiple days",
            "Total usage relatively unchanged (<10% difference)",
            "Pattern matches behavioral change hypothesis"
        ],
        typical_magnitude="5-15% change with pattern shift",
        example_causes=[
            "Work-from-home transition",
            "Schedule change (new work shift)",
            "Lifestyle change (retired, new baby)",
            "Time-of-use rate plan driving behavior",
            "Seasonal daylight changes"
        ]
    ),

    HypothesisCategory.BASELINE_INCREASE: HypothesisCriteria(
        category=HypothesisCategory.BASELINE_INCREASE,
        triggers=[
            "Gradual increase over time",
            "Increased usage across all hours",
            "No specific appliance shows spike",
            "General elevation of consumption floor"
        ],
        required_data=[
            "total_usage_kwh",
            "all appliance categories",
            "timestamp"
        ],
        confidence_factors=[
            "Increase distributed across multiple appliances",
            "Gradual trend rather than sudden change",
            "Nighttime/idle hours also increased",
            "Pattern consistent over weeks"
        ],
        typical_magnitude="10-25% gradual increase",
        example_causes=[
            "Increased occupancy (new household member)",
            "More electronic devices (vampire loads)",
            "Aging appliances becoming less efficient",
            "Multiple small behavioral changes",
            "Gradual increase in hot water usage"
        ]
    ),

    HypothesisCategory.SEASONAL_WEATHER: HypothesisCriteria(
        category=HypothesisCategory.SEASONAL_WEATHER,
        triggers=[
            "Increase matches seasonal weather change",
            "Strong correlation with degree-hours",
            "HVAC proportional to temperature change",
            "Historical pattern matches previous years"
        ],
        required_data=[
            "hvac_cooling_kwh",
            "hvac_heating_kwh",
            "cooling_degree_hours",
            "heating_degree_hours",
            "outdoor_temperature_f"
        ],
        confidence_factors=[
            "Weather correlation r > 0.8",
            "HVAC change proportional to degree-hours",
            "Pattern matches previous year same month",
            "Only HVAC affected, other loads stable"
        ],
        typical_magnitude="20-50% seasonal variation",
        example_causes=[
            "Normal seasonal transition (summer heat, winter cold)",
            "Heat wave or cold snap",
            "Unusually long hot/cold period",
            "Early/late season temperature extremes",
            "Humid conditions increasing AC runtime"
        ]
    ),

    HypothesisCategory.EQUIPMENT_ISSUE: HypothesisCriteria(
        category=HypothesisCategory.EQUIPMENT_ISSUE,
        triggers=[
            "Appliance usage increased without behavioral change",
            "Continuous or frequent cycling",
            "Unexplained sustained high load",
            "Efficiency drop in specific equipment"
        ],
        required_data=[
            "Specific appliance load",
            "Usage patterns",
            "Historical performance"
        ],
        confidence_factors=[
            "Sudden efficiency drop (same conditions, higher usage)",
            "Continuous rather than cyclic load",
            "Other factors ruled out",
            "Equipment age suggests potential failure"
        ],
        typical_magnitude="15-40% increase in specific load",
        example_causes=[
            "Water heater element failing (running constantly)",
            "HVAC compressor issue (short cycling)",
            "Pool pump impeller clogged",
            "Refrigerator seal compromised",
            "Thermostat malfunction"
        ]
    ),
}


# Analysis Workflow Template (7.2 & 7.3)
ANALYSIS_WORKFLOW = """
## Hypothesis Generation Workflow

When analyzing a usage increase, follow this systematic approach:

### Step 1: Gather Baseline Data
- Query historical period data (typically 1-3 months prior)
- Query current period data showing the increase
- Use `get_customer_usage_data()` for detailed hourly data
- Use `query_usage_patterns()` for aggregated insights

### Step 2: Calculate Key Metrics
- Use `calculate_period_comparison()` to get overall change statistics
- Identify which appliances show the largest increases
- Note: Changes >20% are typically significant, >50% are major

### Step 3: Analyze Each Significant Change
For each appliance showing >20% increase:

**For HVAC (heating/cooling):**
- Use `correlate_with_weather()` to check temperature relationship
- Compare degree-hours change to HVAC usage change
- If HVAC increased >30% but degree-hours only +10% → likely thermostat or efficiency issue

**For EV Charging:**
- Check if load pattern is new (zero in baseline)
- Verify night-time charging pattern (10pm-6am, 3-7 kW)
- Calculate kWh/day to estimate miles driven

**For Pool Equipment:**
- Verify seasonal appropriateness (summer only)
- Check if runtime extended (compare hours of operation)
- 6-8 hours daily is normal, >10 hours suggests issue

**For Other Loads:**
- Look for new sustained patterns
- Check time-of-day shifts with `analyze_time_patterns()`
- Identify gradual vs sudden changes

### Step 4: Generate Hypotheses
For each finding, create a hypothesis with:
- **Category**: Select from HVAC_INCREASE, NEW_LOAD, TIME_SHIFT, etc.
- **Confidence**: Based on data quality and correlation strength
  - High (0.7-1.0): Strong evidence, clear pattern, good correlation
  - Medium (0.4-0.7): Moderate evidence, some ambiguity
  - Low (0.0-0.4): Weak evidence, multiple possible causes
- **Evidence**: Specific data points supporting the hypothesis
- **Recommendations**: Actionable steps to verify or address

### Step 5: Rank and Present
- Rank hypotheses by confidence score and impact
- Present top 2-3 most likely causes
- Include estimated savings potential for each
"""


# Few-Shot Examples (7.4)
FEW_SHOT_EXAMPLES = """
## Example Scenarios

### Example 1: HVAC Thermostat Change

**Input Data:**
- July usage: 1,450 kWh (vs June: 1,050 kWh, +38%)
- HVAC cooling: 680 kWh (vs June: 450 kWh, +51%)
- Cooling degree-hours: 1,650 (vs June: 1,500, +10%)
- Weather correlation: r=0.65

**Analysis:**
HVAC cooling increased 51% while degree-hours only increased 10%. This disproportionate increase suggests the HVAC system is running more than weather alone would explain. Temperature correlation is moderate (r=0.65) but the magnitude mismatch is the key indicator.

**Hypothesis Generated:**
{
  "category": "hvac_increase",
  "description": "HVAC cooling usage increased 51% despite only 10% increase in cooling degree-hours, suggesting thermostat setting may have been lowered",
  "confidence_score": 0.85,
  "evidence": [
    "HVAC cooling: 680 kWh vs 450 kWh baseline (+230 kWh, +51%)",
    "Cooling degree-hours only increased 10% (1,650 vs 1,500)",
    "Expected increase based on weather: ~45 kWh, actual: 230 kWh"
  ],
  "recommendations": [
    "Check thermostat setting - recommend 78°F for cooling efficiency",
    "Inspect air filter and replace if dirty",
    "Consider programmable thermostat to maintain consistent temperature"
  ],
  "potential_savings": 45.00
}

**Conversational Output:**
"Your AC usage jumped significantly in July—up 51% from June. Here's the interesting part: it only got about 10% hotter outside, but your AC is using way more energy than the weather change would explain. This usually means the thermostat got set a few degrees lower. Bumping it back up to 78°F could save you around $45/month while keeping your home comfortable."

---

### Example 2: New EV Charger Detection

**Input Data:**
- August usage: 1,285 kWh (vs July: 1,050 kWh, +22%)
- EV charging: 0 kWh → 135 kWh (new load)
- Charging pattern: 10pm-2am, 4-6 kW, 5 days/week
- Started: August 10

**Analysis:**
A new sustained load of ~135 kWh/month appeared starting August 10, with a characteristic night-time charging pattern (10pm-2am) at Level 2 charger power levels (4-6 kW). This pattern is consistent with residential EV charging, typically 5 days per week.

**Hypothesis Generated:**
{
  "category": "new_load",
  "description": "New EV charging pattern detected starting August 10, adding ~135 kWh/month",
  "confidence_score": 0.95,
  "evidence": [
    "EV charging increased from 0 to 135 kWh",
    "Consistent night-time pattern: 10pm-2am, 4-6 kW",
    "Charging 5 days/week pattern",
    "Started abruptly on August 10"
  ],
  "recommendations": [
    "Confirm new EV purchase or charging routine",
    "Consider shifting to midnight-6am if utility offers off-peak rates",
    "Monitor charging frequency if usage seems high"
  ],
  "potential_savings": null
}

**Conversational Output:**
"I spotted something new in your August data—it looks like you started charging an electric vehicle around August 10th. The charging pattern is typical for a home EV: overnight sessions (10pm-2am) about 5 times a week, adding roughly 135 kWh per month. If your utility offers time-of-use rates, charging between midnight and 6am could save you money."

---

### Example 3: Work-From-Home Time Shift

**Input Data:**
- September usage: 1,120 kWh (vs August: 1,050 kWh, +7%)
- Peak hour shifted: 7pm → 2pm
- Weekday 9am-5pm usage: +25%
- Total usage only slightly up

**Analysis:**
Total usage increased modestly (+7%), but the timing changed dramatically. The peak usage hour shifted from evening (7pm) to afternoon (2pm), and weekday daytime usage (9am-5pm) increased 25%. This pattern is consistent with someone working from home rather than being away during the day.

**Hypothesis Generated:**
{
  "category": "time_shift",
  "description": "Usage pattern shifted to daytime hours, consistent with work-from-home transition",
  "confidence_score": 0.75,
  "evidence": [
    "Peak hour shifted from 7pm to 2pm",
    "Weekday 9am-5pm usage increased 25%",
    "Total usage only up 7% (pattern change, not new load)",
    "Weekend pattern unchanged"
  ],
  "recommendations": [
    "Consider adjusting thermostat during unoccupied hours if schedule changes",
    "Use natural lighting during daytime to reduce lighting needs",
    "Be mindful of phantom loads from always-on equipment"
  ],
  "potential_savings": 15.00
}

**Conversational Output:**
"Your September bill shows an interesting change in when you're using energy, not just how much. Your peak usage moved from evening to mid-afternoon, and weekday daytime usage is up 25%. This pattern typically happens when someone starts working from home. The good news: your total usage only went up 7%, so you're being pretty efficient! Adjusting your thermostat for your new schedule could save another $15/month."
"""


def get_hypothesis_criteria(category: HypothesisCategory) -> HypothesisCriteria:
    """Get criteria for a specific hypothesis category.

    Args:
        category: The hypothesis category

    Returns:
        Criteria for that category
    """
    return HYPOTHESIS_FRAMEWORK.get(category)


def get_all_hypothesis_categories() -> List[HypothesisCategory]:
    """Get list of all available hypothesis categories.

    Returns:
        List of hypothesis categories
    """
    return list(HYPOTHESIS_FRAMEWORK.keys())


if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to path for relative imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from bill_investigator.output_formatting import HypothesisCategory as HC

    print("=" * 70)
    print("Hypothesis Framework")
    print("=" * 70)
    print()

    for category, criteria in HYPOTHESIS_FRAMEWORK.items():
        print(f"Category: {category.value.upper()}")
        print(f"Typical Impact: {criteria.typical_magnitude}")
        print(f"Key Triggers:")
        for trigger in criteria.triggers[:3]:
            print(f"  - {trigger}")
        print(f"Example Causes:")
        for cause in criteria.example_causes[:3]:
            print(f"  - {cause}")
        print()

    print("=" * 70)
    print(f"✓ {len(HYPOTHESIS_FRAMEWORK)} hypothesis categories defined")
    print("=" * 70)
