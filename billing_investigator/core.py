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

"""Bill Investigator Agent - Main agent for analyzing utility usage patterns."""

import os
import warnings
from datetime import date

# Configure GenAI to use Vertex AI BEFORE importing ADK
# These environment variables are read by google.genai.Client during initialization
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("BIGQUERY_PROJECT_ID", "")
os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv("VERTEXAI_LOCATION", "us-central1")

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from .tools.bigquery_tools import get_database_settings
from .prompts import return_instructions_root

# Suppress ADK experimental warnings
warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*", category=UserWarning)

date_today = date.today()


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the agent before each call.

    This callback:
    1. Initializes database settings in session state
    2. Retrieves BigQuery schema and sample data
    3. Injects schema into agent instruction for context

    Args:
        callback_context: The callback context containing state and agent info
    """

    # Initialize database settings in session state if not present
    if "database_settings" not in callback_context.state:
        callback_context.state["database_settings"] = get_database_settings(
            callback_context
        )

    # Inject schema into agent instruction for context
    schema = callback_context.state["database_settings"]["tables"]

    # Build schema summary for instruction
    schema_summary = []
    for table_name, table_info in schema.items():
        schema_summary.append(f"\nTable: {table_name}")
        schema_summary.append(f"  Rows: {table_info.get('row_count', 'unknown')}")
        schema_summary.append("  Schema:")
        for col_name, col_type in table_info["table_schema"]:
            schema_summary.append(f"    - {col_name}: {col_type}")

    schema_text = "\n".join(schema_summary)

    # Update agent instruction with schema context
    callback_context._invocation_context.agent.instruction = (
        return_instructions_root()
        + f"""

    --------- BigQuery Schema with Sample Data ---------
    {schema_text}

    Project: {callback_context.state["database_settings"]["project_id"]}
    Dataset: {callback_context.state["database_settings"]["dataset_id"]}
    Table: {callback_context.state["database_settings"]["table_name"]}
    Full Table Name: {callback_context.state["database_settings"]["full_table_name"]}
    """
    )


# Define tools from the bigquery_tools module
# Import as functions so they can be used directly by the agent
from .tools.bigquery_tools import (
    get_customer_usage_data,
    query_usage_patterns,
)

# Create the root agent
root_agent = Agent(
    model=os.getenv("BILL_INVESTIGATOR_MODEL", "gemini-2.5-pro"),
    name="bill_investigator",
    instruction=return_instructions_root(),
    global_instruction=f"""
    You are the Utility Bill Investigator, an expert in analyzing residential energy consumption patterns.
    Today's date: {date_today}

    Your mission is to help customers understand unexpected changes in their utility bills by:
    1. Querying disaggregated usage data from BigQuery
    2. Identifying anomalies and pattern changes
    3. Generating data-driven hypotheses about causes
    4. Providing actionable recommendations

    You have access to detailed hourly energy consumption data including HVAC, water heater,
    EV charging, pool equipment, major appliances, and environmental factors.
    """,
    tools=[
        get_customer_usage_data,
        query_usage_patterns,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1  # Low temperature for deterministic analysis
    ),
)
