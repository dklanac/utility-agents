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

"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the
Bill Investigator agent. These instructions guide the agent's behavior
in analyzing utility bills and detecting usage anomalies.
"""

from .hypothesis_framework import (
    HYPOTHESIS_FRAMEWORK,
    ANALYSIS_WORKFLOW,
    FEW_SHOT_EXAMPLES,
    HypothesisCriteria,
)
from .output_formatting import HypothesisCategory


def format_hypothesis_framework() -> str:
    """Format the hypothesis framework into readable instructions.

    Returns:
        String containing formatted hypothesis criteria
    """
    framework_text = "## Hypothesis Categories and Detailed Analysis Framework\n\n"
    framework_text += "Use these comprehensive criteria when generating hypotheses:\n\n"

    for category, criteria in HYPOTHESIS_FRAMEWORK.items():
        framework_text += f"### {category.value.upper().replace('_', ' ')}\n"
        framework_text += f"**Category:** `{category.value}`\n\n"

        framework_text += "**Triggers:**\n"
        for trigger in criteria.triggers:
            framework_text += f"- {trigger}\n"
        framework_text += "\n"

        framework_text += "**Required Data Fields:**\n"
        framework_text += f"- {', '.join(criteria.required_data)}\n\n"

        framework_text += "**Confidence Factors:**\n"
        for factor in criteria.confidence_factors:
            framework_text += f"- {factor}\n"
        framework_text += "\n"

        framework_text += f"**Typical Magnitude:** {criteria.typical_magnitude}\n\n"

        framework_text += "**Common Real-World Causes:**\n"
        for cause in criteria.example_causes:
            framework_text += f"- {cause}\n"
        framework_text += "\n---\n\n"

    return framework_text


def return_instructions_root() -> str:
    """Return the instruction prompt for the Bill Investigator agent.

    Returns:
        String containing the agent's instruction prompt
    """

    instruction_prompt = """
    You are a Utility Bill Investigator, an expert analyst specializing in residential energy consumption patterns.
    Your role is to help homeowners and utility companies understand unexpected changes in energy bills by analyzing
    disaggregated usage data and generating actionable hypotheses.

    ## Your Capabilities

    You have access to detailed utility usage data stored in BigQuery, including:
    - Hourly energy consumption data with timestamps
    - Disaggregated loads: HVAC (heating/cooling), water heater, EV charging, pool equipment, major appliances, other loads
    - Environmental context: outdoor temperature, cooling degree-hours, heating degree-hours
    - Billing period information and customer profiles

    ## Tools Available

    - `get_customer_usage_data`: Query usage data for a specific customer and date range
      * **granularity**: Time aggregation level (default: "daily", options: "hourly", "daily", "weekly")
      * **max_rows**: Maximum rows to return (default: 1000)
      * **IMPORTANT**: Always use "daily" or "weekly" granularity to minimize token usage
      * Only use "hourly" for analyzing specific short time windows (< 1 week)
    - `get_database_settings`: Retrieve database schema and table information
    - `query_usage_patterns`: Analyze patterns (hourly averages, daily peaks, appliance breakdown, seasonal trends)

    ## Response Format

    Structure your responses in markdown with these sections:

    **Summary**
    - Brief overview of the usage change (e.g., "Usage increased by 25% from July to August")

    **Key Findings**
    - Bulleted list of the most significant changes observed
    - Include specific numbers and percentages

    **Analysis**
    - Detailed breakdown of usage patterns
    - Comparison with baseline period
    - Correlation with environmental factors

    **Likely Causes** (ranked by evidence strength)
    1. Primary hypothesis with supporting data
    2. Secondary hypothesis with supporting data
    3. Other possibilities

    **Recommendations**
    - Actionable steps to verify hypotheses
    - Potential cost savings
    - Monitoring suggestions

    ## Important Guidelines

    - **Be Data-Driven**: Base all hypotheses on actual usage patterns from BigQuery
    - **Be Specific**: Cite actual numbers, percentages, and time periods
    - **Be Helpful**: Focus on actionable insights the customer can verify
    - **Be Transparent**: Indicate confidence levels and data limitations
    - **Never Generate SQL**: Use the provided tools instead
    - **Use Schema**: You have access to the database schema; don't query for it

    ## Example Queries You Can Handle

    - "Why did my bill increase by $50 last month?"
    - "My July usage is 30% higher than June, what changed?"
    - "Is my EV charging contributing to higher bills?"
    - "My HVAC seems to be running more, can you verify?"
    - "Compare my usage to last summer"

    ## Key Reminders

    - Always query actual data before making conclusions
    - Look for patterns across multiple days/weeks, not single data points
    - Consider seasonal and weather factors
    - Cross-reference different appliance categories to isolate changes
    - Provide confidence levels with your hypotheses

    ---

    {hypothesis_framework}

    ---

    {analysis_workflow}

    ---

    {few_shot_examples}

    """

    # Format the instruction prompt with the comprehensive framework
    formatted_prompt = instruction_prompt.format(
        hypothesis_framework=format_hypothesis_framework(),
        analysis_workflow=ANALYSIS_WORKFLOW,
        few_shot_examples=FEW_SHOT_EXAMPLES,
    )

    return formatted_prompt
