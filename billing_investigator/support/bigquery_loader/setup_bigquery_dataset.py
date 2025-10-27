#!/usr/bin/env python3
"""
BigQuery Dataset Setup Script (Task 3.1)
Creates the utility_usage dataset with proper configuration.
"""

import os
from google.cloud import bigquery
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_bigquery_dataset():
    """Create BigQuery dataset with proper configuration."""
    
    # Get configuration from environment
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id or not dataset_id:
        raise ValueError(
            "Missing required environment variables: "
            "BIGQUERY_PROJECT_ID, BIGQUERY_DATASET"
        )
    
    print("=" * 70)
    print("BIGQUERY DATASET SETUP (Task 3.1)")
    print("=" * 70)
    print(f"Project ID: {project_id}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Location: {location}")
    print()
    
    # Initialize BigQuery client
    try:
        client = bigquery.Client(project=project_id)
        print("✅ BigQuery client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize BigQuery client: {e}")
        print("\nPlease ensure:")
        print("  1. gcloud CLI is installed and authenticated")
        print("  2. Application default credentials are set:")
        print("     gcloud auth application-default login")
        print("  3. BigQuery API is enabled in your GCP project")
        return False
    
    # Construct dataset reference
    dataset_ref = f"{project_id}.{dataset_id}"
    
    # Check if dataset already exists
    try:
        existing_dataset = client.get_dataset(dataset_ref)
        print(f"⚠️  Dataset already exists: {dataset_ref}")
        print(f"   Created: {existing_dataset.created}")
        print(f"   Location: {existing_dataset.location}")
        print(f"   Description: {existing_dataset.description or 'None'}")
        
        # Ask if we should continue anyway
        response = input("\nDataset exists. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return False
        print()
    except exceptions.NotFound:
        print(f"✅ Dataset does not exist, will create new")
    
    # Create dataset configuration
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    dataset.description = (
        "Synthetic utility usage data for Bill Investigator agent. "
        "Contains hourly energy consumption records with disaggregated loads, "
        "environmental context, and customer profiles."
    )
    
    # Set default table expiration (optional - None means tables don't expire)
    # dataset.default_table_expiration_ms = 90 * 24 * 60 * 60 * 1000  # 90 days
    
    # Create dataset
    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"✅ Dataset created successfully: {dataset_ref}")
        print(f"   Location: {dataset.location}")
        print(f"   Description: {dataset.description}")
    except Exception as e:
        print(f"❌ Failed to create dataset: {e}")
        return False
    
    # Verify dataset access
    print("\n" + "=" * 70)
    print("VERIFYING DATASET ACCESS")
    print("=" * 70)
    
    try:
        # List tables in dataset (should be empty)
        tables = list(client.list_tables(dataset_ref))
        print(f"✅ Dataset accessible")
        print(f"   Current tables: {len(tables)}")
        
        if tables:
            print("\n   Existing tables:")
            for table in tables:
                print(f"     - {table.table_id}")
    except Exception as e:
        print(f"❌ Failed to access dataset: {e}")
        return False
    
    # Check IAM permissions
    print("\n" + "=" * 70)
    print("CHECKING IAM PERMISSIONS")
    print("=" * 70)
    
    try:
        # Get current user/service account
        credentials = client._credentials
        print(f"✅ Using credentials: {type(credentials).__name__}")
        
        # Test permissions
        required_permissions = [
            "bigquery.datasets.get",
            "bigquery.tables.create",
            "bigquery.tables.get",
            "bigquery.tables.list",
            "bigquery.tables.getData",
        ]
        
        try:
            test_permissions = client.test_iam_permissions(
                dataset_ref, 
                required_permissions
            )
            
            print(f"\n   Required permissions ({len(required_permissions)}):")
            for perm in required_permissions:
                has_perm = perm in test_permissions
                status = "✅" if has_perm else "❌"
                print(f"     {status} {perm}")
            
            missing = set(required_permissions) - set(test_permissions)
            if missing:
                print(f"\n   ⚠️  Missing permissions: {len(missing)}")
                print("   You may need to grant additional IAM roles")
        except Exception as e:
            print(f"   ⚠️  Could not test IAM permissions: {e}")
            print("   This is not critical - proceeding anyway")
    except Exception as e:
        print(f"❌ Error checking permissions: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"✅ Dataset '{dataset_id}' is ready in project '{project_id}'")
    print(f"\nNext steps:")
    print(f"  1. Create table schema (Task 3.2)")
    print(f"  2. Configure partitioning/clustering (Task 3.3)")
    print(f"  3. Load synthetic data (Task 3.4)")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    
    try:
        success = create_bigquery_dataset()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
