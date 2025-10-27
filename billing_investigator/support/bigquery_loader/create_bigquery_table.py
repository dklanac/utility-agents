#!/usr/bin/env python3
"""
BigQuery Table Schema Creation (Task 3.2)
Creates the energy_consumption table with enhanced schema.
"""

import os
from google.cloud import bigquery
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_table_schema():
    """
    Create BigQuery table schema based on data_generators/schema.py
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_generators.schema import ENERGY_CONSUMPTION_SCHEMA

    # Convert our schema to BigQuery schema
    bq_schema = []
    
    for field_name, field_spec in ENERGY_CONSUMPTION_SCHEMA.items():
        field = bigquery.SchemaField(
            name=field_name,
            field_type=field_spec["type"],
            mode=field_spec.get("mode", "NULLABLE"),
            description=field_spec.get("description", "")
        )
        bq_schema.append(field)
    
    return bq_schema

def create_energy_consumption_table():
    """Create the energy_consumption table in BigQuery."""
    
    # Get configuration from environment
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET")
    table_id = os.getenv("BIGQUERY_TABLE")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not all([project_id, dataset_id, table_id]):
        raise ValueError(
            "Missing required environment variables: "
            "BIGQUERY_PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE"
        )
    
    print("=" * 70)
    print("BIGQUERY TABLE SCHEMA CREATION (Task 3.2)")
    print("=" * 70)
    print(f"Project ID: {project_id}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Table ID: {table_id}")
    print()
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Construct table reference
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    # Check if table already exists
    try:
        existing_table = client.get_table(table_ref)
        print(f"⚠️  Table already exists: {table_ref}")
        print(f"   Created: {existing_table.created}")
        print(f"   Rows: {existing_table.num_rows:,}")
        print(f"   Schema fields: {len(existing_table.schema)}")
        
        response = input("\nTable exists. Drop and recreate? (y/n): ")
        if response.lower() != 'y':
            print("Aborted. Table unchanged.")
            return False
        
        # Drop existing table
        client.delete_table(table_ref)
        print(f"✅ Existing table dropped")
        print()
    except exceptions.NotFound:
        print(f"✅ Table does not exist, will create new")
    
    # Create schema
    schema = create_table_schema()
    print(f"\n✅ Schema created with {len(schema)} fields:")
    print(f"\n{'Field Name':<30} {'Type':<12} {'Mode':<10} {'Description'[:40]}")
    print("-" * 70)
    for field in schema[:15]:  # Show first 15 fields
        desc = field.description[:40] if field.description else ""
        print(f"{field.name:<30} {field.field_type:<12} {field.mode:<10} {desc}")
    if len(schema) > 15:
        print(f"... and {len(schema) - 15} more fields")
    
    # Create table
    table = bigquery.Table(table_ref, schema=schema)
    table.description = (
        "Hourly energy consumption data with disaggregated loads. "
        "Contains synthetic data for testing Bill Investigator agent."
    )
    
    # Note: Partitioning and clustering will be added in Task 3.3
    
    try:
        table = client.create_table(table)
        print(f"\n✅ Table created successfully: {table_ref}")
        print(f"   Fields: {len(table.schema)}")
        print(f"   Description: {table.description}")
    except Exception as e:
        print(f"\n❌ Failed to create table: {e}")
        return False
    
    # Verify table
    print("\n" + "=" * 70)
    print("VERIFYING TABLE")
    print("=" * 70)
    
    try:
        table = client.get_table(table_ref)
        print(f"✅ Table accessible")
        print(f"   Full path: {table.full_table_id}")
        print(f"   Created: {table.created}")
        print(f"   Schema fields: {len(table.schema)}")
        print(f"   Current rows: {table.num_rows}")
        print(f"   Partitioning: {table.time_partitioning or 'None (will add in Task 3.3)'}")
        print(f"   Clustering: {table.clustering_fields or 'None (will add in Task 3.3)'}")
    except Exception as e:
        print(f"❌ Failed to verify table: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("TABLE CREATION COMPLETE")
    print("=" * 70)
    print(f"✅ Table '{table_id}' is ready in dataset '{dataset_id}'")
    print(f"\nNext steps:")
    print(f"  1. Configure partitioning by timestamp (Task 3.3)")
    print(f"  2. Configure clustering by customer_id (Task 3.3)")
    print(f"  3. Load synthetic data (Task 3.4)")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    
    try:
        success = create_energy_consumption_table()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
