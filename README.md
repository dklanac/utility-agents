# Billing Investigator

An intelligent AI agent that analyzes residential energy consumption patterns to identify and explain unexpected utility bill increases.

## What It Does

Billing Investigator uses Google's Gemini AI and advanced statistical analysis to help residential customers understand why their energy bills have increased. It analyzes disaggregated consumption data across multiple appliance categories, correlates usage with weather patterns, and generates evidence-based explanations with actionable recommendations.

## Key Features

-   **Intelligent Anomaly Detection**: Multi-method statistical analysis with configurable sensitivity
-   **AI-Powered Insights**: Natural language explanations powered by Gemini 2.5 Pro
-   **Disaggregated Analysis**: Tracks 7 distinct load categories (HVAC, EV, water heater, pool, etc.)
-   **Weather Correlation**: Temperature sensitivity modeling and degree-hour analysis
-   **Hypothesis Ranking**: Six categories of explanations ranked by confidence
-   **Actionable Recommendations**: Specific steps to address identified issues

## Quick Start

### Prerequisites

-   Python 3.12+
-   Google Cloud Project with BigQuery API enabled
-   Authenticated GCP credentials

### Installation

```bash
# Clone repository
cd ~/Code/utility-agents

# Install dependencies
uv sync

# Authenticate with Google Cloud
gcloud auth application-default login
```

### Configuration

Create `.env` file:

```bash
BIGQUERY_PROJECT_ID=your-gcp-project
BIGQUERY_DATASET=utility_usage
BIGQUERY_TABLE=energy_consumption
VERTEXAI_LOCATION=us-central1
BILL_INVESTIGATOR_MODEL=gemini-2.5-pro
```

### Generate Sample Data

```bash
# Create 20 customers with 12 months of synthetic data
uv run python billing_investigator/support/bigquery_loader/setup_all.py

# Custom generation
uv run python billing_investigator/support/bigquery_loader/setup_all.py \
  --customers 10 \
  --months 6 \
  --generate-only  # Don't load to BigQuery
  --force          # Overwrite existing

# Generate specific customer profiles
uv run python billing_investigator/support/bigquery_loader/setup_all.py \
  --profile suburban_family \
  --anomaly thermostat_change
```

### Run the Agent

```bash
# Start web interface
uv run adk web

# Open browser to http://localhost:8000
```

### Example Queries

```
# Basic analysis
"Analyze CUST_001's usage for last month"
"Why did CUST_002's bill increase?"

# Period comparison
"Compare CUST_001's February vs January usage"
"Show year-over-year changes for CUST_003"

# Specific investigations
"Check for HVAC issues in CUST_004's data"
"Identify vampire loads for CUST_005"
"Find optimal EV charging times for CUST_006"
```

## System Architecture

```
Web Interface (ADK)
        ↓
AI Agent Core (Gemini)
        ↓
Analysis Engine
    ├── Statistical Analysis
    ├── Hypothesis Framework
    └── Output Formatting
        ↓
Data Pipeline (BigQuery)
```

## Hypothesis Categories

The system generates explanations across six categories:

1. **HVAC Changes**: Thermostat adjustments or system issues
2. **New Loads**: Recently added appliances or equipment
3. **Usage Pattern Shifts**: Behavioral changes like work-from-home
4. **Baseline Increases**: Gradual consumption growth
5. **Seasonal Weather**: Normal temperature-driven variations
6. **Equipment Issues**: Malfunctioning or inefficient equipment

## Use Cases

-   **Residential Customers**: Understand bill increases and identify savings opportunities
-   **Utility Companies**: Reduce customer service calls with automated analysis
-   **Energy Consultants**: Rapid consumption analysis and recommendations

## Technical Stack

-   **AI Framework**: Google ADK with Gemini 2.5 Pro
-   **Data Infrastructure**: Google BigQuery
-   **Language**: Python 3.12+
-   **Package Management**: uv

## Field Reference

### Table Schema

| Field                | Type      | Values/Range |
| -------------------- | --------- | ------------ |
| customer_id          | STRING    | CUST_XXX     |
| timestamp            | TIMESTAMP | Hourly       |
| total_kwh            | FLOAT     | 0-50+        |
| hvac_heating_kwh     | FLOAT     | 0-20         |
| hvac_cooling_kwh     | FLOAT     | 0-20         |
| ev_kwh               | FLOAT     | 0-10         |
| water_heater_kwh     | FLOAT     | 0-5          |
| pool_kwh             | FLOAT     | 0-5          |
| appliances_kwh       | FLOAT     | 0-10         |
| other_kwh            | FLOAT     | 0-5          |
| temperature          | FLOAT     | -20 to 120°F |
| cooling_degree_hours | FLOAT     | 0+           |
| heating_degree_hours | FLOAT     | 0+           |

### Anomaly Types

| Type                | Key        | Impact           |
| ------------------- | ---------- | ---------------- |
| `thermostat_change` | HVAC       | +30-50%          |
| `new_ev`            | EV         | +3-5 kWh/night   |
| `pool_failure`      | Pool       | Constant 2-3 kWh |
| `appliance_upgrade` | Appliances | +20-30%          |

### Hypothesis Types

| Code                | Description            |
| ------------------- | ---------------------- |
| `HVAC_INCREASE`     | HVAC usage anomaly     |
| `NEW_LOAD`          | New appliance detected |
| `TIME_SHIFT`        | Pattern change         |
| `BASELINE_INCREASE` | Overall increase       |
| `SEASONAL_WEATHER`  | Weather-driven         |
| `EQUIPMENT_ISSUE`   | Malfunction            |
