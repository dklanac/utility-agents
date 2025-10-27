#!/usr/bin/env python3
"""
BigQuery Table Partitioning and Clustering Configuration (Task 3.3)
Configures time partitioning and clustering for optimal query performance.
"""

import os
from google.cloud import bigquery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def configure_partitioning_clustering():
    """Configure table partitioning by timestamp and clustering by customer_id."""
    
    # Get configuration from environment
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET")
    table_id = os.getenv("BIGQUERY_TABLE")
    
    print("=" * 70)
    print("BIGQUERY TABLE PARTITIONING & CLUSTERING (Task 3.3)")
    print("=" * 70)
    print(f"Table: {project_id}.{dataset_id}.{table_id}")
    print()
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    # Get existing table
    table = client.get_table(table_ref)
    
    print("Current configuration:")
    print(f"  Partitioning: {table.time_partitioning or 'None'}")
    print(f"  Clustering: {table.clustering_fields or 'None'}")
    print()
    
    # Check if already configured
    if table.time_partitioning and table.clustering_fields:
        print("⚠️  Table already has partitioning and clustering configured")
        response = input("Recreate table with new configuration? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
    
    # Note: BigQuery doesn't allow modifying partitioning/clustering after creation
    # We need to:
    # 1. Create a new temp table with partitioning/clustering
    # 2. Copy data (if any)
    # 3. Drop old table
    # 4. Rename temp table
    
    # For now, since table is empty, we'll just drop and recreate
    
    if table.num_rows > 0:
        print(f"⚠️  Table contains {table.num_rows:,} rows")
        print("Cannot modify partitioning on non-empty table.")
        print("Please back up data and recreate table.")
        return False
    
    print("Recreating table with partitioning and clustering...")
    
    # Get existing schema
    schema = table.schema
    description = table.description
    
    # Drop existing table
    client.delete_table(table_ref)
    print("✅ Existing table dropped")
    
    # Create new table with partitioning and clustering
    table = bigquery.Table(table_ref, schema=schema)
    table.description = description
    
    # Configure time partitioning
    # Partition by DAY on the timestamp field
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",  # Our timestamp field
        expiration_ms=None,  # Don't expire partitions automatically
    )
    
    # Configure clustering
    # Cluster by customer_id for efficient customer queries
    table.clustering_fields = ["customer_id"]
    
    # Create table
    table = client.create_table(table)
    
    print(f"\n✅ Table recreated with partitioning and clustering")
    print(f"   Partitioning:")
    print(f"     - Type: {table.time_partitioning.type_}")
    print(f"     - Field: {table.time_partitioning.field}")
    print(f"     - Expiration: {table.time_partitioning.expiration_ms or 'None'}")
    print(f"   Clustering:")
    print(f"     - Fields: {', '.join(table.clustering_fields)}")
    
    # Verify configuration
    print("\n" + "=" * 70)
    print("VERIFYING CONFIGURATION")
    print("=" * 70)
    
    table = client.get_table(table_ref)
    
    assert table.time_partitioning is not None, "Time partitioning not set"
    assert table.time_partitioning.field == "timestamp", "Wrong partitioning field"
    assert table.clustering_fields == ["customer_id"], "Wrong clustering fields"
    
    print("✅ Partitioning and clustering verified")
    print()
    
    # Query performance benefits
    print("=" * 70)
    print("PERFORMANCE BENEFITS")
    print("=" * 70)
    print("✅ Time Partitioning (by timestamp):")
    print("   - Queries with timestamp filters scan only relevant partitions")
    print("   - Example: WHERE timestamp >= '2024-01-01'")
    print("   - Reduces scan cost and improves speed")
    print()
    print("✅ Clustering (by customer_id):")
    print("   - Data for same customer stored together")
    print("   - Example: WHERE customer_id = 'CUST_001'")
    print("   - Reduces bytes scanned when filtering by customer")
    print()
    
    # Summary
    print("=" * 70)
    print("CONFIGURATION COMPLETE")
    print("=" * 70)
    print(f"✅ Table '{table_id}' optimized for:")
    print(f"   - Time-based queries (partitioned by day)")
    print(f"   - Customer-specific queries (clustered by customer_id)")
    print(f"\nNext steps:")
    print(f"  1. Load synthetic data (Task 3.4)")
    print(f"  2. Test query performance (Task 3.6)")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    
    try:
        success = configure_partitioning_clustering()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
