#!/usr/bin/env python3
"""
IAM Permissions Verification and Setup (Task 3.5)
Verifies and documents IAM permissions for ADK agent BigQuery access.
"""

import os
from google.cloud import bigquery
from google.auth import default
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_current_credentials():
    """Get current credentials being used."""
    try:
        credentials, project = default()
        return credentials, project
    except Exception as e:
        print(f"❌ Failed to get credentials: {e}")
        return None, None

def verify_bigquery_access():
    """Verify BigQuery access with current credentials."""
    
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET")
    table_id = os.getenv("BIGQUERY_TABLE")
    
    print("=" * 70)
    print("IAM PERMISSIONS VERIFICATION (Task 3.5)")
    print("=" * 70)
    
    # Get credentials
    credentials, auth_project = get_current_credentials()
    
    if credentials:
        print(f"✅ Using credentials: {type(credentials).__name__}")
        print(f"   Auth project: {auth_project}")
        print(f"   Target project: {project_id}")
        
        # Check if service account or user
        if hasattr(credentials, 'service_account_email'):
            print(f"   Service Account: {credentials.service_account_email}")
        elif hasattr(credentials, '_service_account_email'):
            print(f"   Service Account: {credentials._service_account_email}")
        else:
            print(f"   Type: User credentials (for development)")
    print()
    
    # Initialize BigQuery client
    try:
        client = bigquery.Client(project=project_id)
        print("✅ BigQuery client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize BigQuery client: {e}")
        return False
    
    print()
    print("=" * 70)
    print("TESTING PERMISSIONS")
    print("=" * 70)
    
    tests = []
    
    # Test 1: List datasets
    print("1. Testing dataset listing...")
    try:
        datasets = list(client.list_datasets())
        dataset_ids = [d.dataset_id for d in datasets]
        if dataset_id in dataset_ids:
            print(f"   ✅ Can list datasets (found {dataset_id})")
            tests.append(("List datasets", True))
        else:
            print(f"   ⚠️  Can list datasets but {dataset_id} not found")
            tests.append(("List datasets", False))
    except Exception as e:
        print(f"   ❌ Cannot list datasets: {e}")
        tests.append(("List datasets", False))
    
    # Test 2: Get dataset
    dataset_ref = f"{project_id}.{dataset_id}"
    print(f"\n2. Testing dataset access ({dataset_id})...")
    try:
        dataset = client.get_dataset(dataset_ref)
        print(f"   ✅ Can access dataset")
        tests.append(("Access dataset", True))
    except Exception as e:
        print(f"   ❌ Cannot access dataset: {e}")
        tests.append(("Access dataset", False))
        return False
    
    # Test 3: List tables
    print(f"\n3. Testing table listing...")
    try:
        tables = list(client.list_tables(dataset_ref))
        table_ids = [t.table_id for t in tables]
        if table_id in table_ids:
            print(f"   ✅ Can list tables (found {table_id})")
            tests.append(("List tables", True))
        else:
            print(f"   ⚠️  Can list tables but {table_id} not found")
            tests.append(("List tables", False))
    except Exception as e:
        print(f"   ❌ Cannot list tables: {e}")
        tests.append(("List tables", False))
    
    # Test 4: Get table
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    print(f"\n4. Testing table access ({table_id})...")
    try:
        table = client.get_table(table_ref)
        print(f"   ✅ Can access table ({table.num_rows:,} rows)")
        tests.append(("Access table", True))
    except Exception as e:
        print(f"   ❌ Cannot access table: {e}")
        tests.append(("Access table", False))
        return False
    
    # Test 5: Query execution
    print(f"\n5. Testing query execution...")
    try:
        query = f"""
        SELECT COUNT(*) as row_count
        FROM `{table_ref}`
        LIMIT 1
        """
        query_job = client.query(query)
        results = list(query_job.result())
        row_count = results[0].row_count
        print(f"   ✅ Can execute queries (verified {row_count:,} rows)")
        tests.append(("Execute queries", True))
    except Exception as e:
        print(f"   ❌ Cannot execute queries: {e}")
        tests.append(("Execute queries", False))
        return False
    
    # Test 6: Read data
    print(f"\n6. Testing data reading...")
    try:
        query = f"""
        SELECT customer_id, total_usage_kwh, timestamp
        FROM `{table_ref}`
        ORDER BY timestamp DESC
        LIMIT 5
        """
        query_job = client.query(query)
        results = list(query_job.result())
        print(f"   ✅ Can read data ({len(results)} sample rows)")
        tests.append(("Read data", True))
    except Exception as e:
        print(f"   ❌ Cannot read data: {e}")
        tests.append(("Read data", False))
        return False
    
    # Summary
    print()
    print("=" * 70)
    print("PERMISSION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All permission tests passed!")
        print("   Current credentials have sufficient access for ADK agent")
    else:
        print("\n❌ Some permission tests failed")
        print("   Additional IAM roles may be needed")
    
    return passed == total

def print_iam_documentation():
    """Print IAM setup documentation."""
    
    print("\n" + "=" * 70)
    print("IAM SETUP DOCUMENTATION")
    print("=" * 70)
    
    print("""
For Development (Current Setup):
---------------------------------
✅ Using Application Default Credentials (ADC)
   - Authenticated via: gcloud auth application-default login
   - Recommended for local development and testing
   - Uses your user account permissions

For Production Deployment:
---------------------------
When deploying to Vertex AI Agent Engine, you'll need a service account:

1. Create Service Account:
   gcloud iam service-accounts create bill-investigator-agent \\
       --display-name="Bill Investigator Agent" \\
       --project=YOUR_PROJECT_ID

2. Grant Required Roles at Project Level:
   
   # For query execution
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\
       --member="serviceAccount:bill-investigator-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \\
       --role="roles/bigquery.jobUser"

3. Grant Required Roles at Dataset Level:
   
   # For data reading
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\
       --member="serviceAccount:bill-investigator-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com" \\
       --role="roles/bigquery.dataViewer"

Minimum Required Permissions:
------------------------------
✅ bigquery.datasets.get       - List and access datasets
✅ bigquery.tables.get         - Access table metadata
✅ bigquery.tables.list        - List tables in dataset
✅ bigquery.tables.getData     - Read table data
✅ bigquery.jobs.create        - Execute queries

Recommended IAM Roles:
----------------------
✅ roles/bigquery.jobUser      - Execute queries (project level)
✅ roles/bigquery.dataViewer   - Read data (dataset level)

Security Best Practices:
------------------------
✅ Use service account (not user credentials) in production
✅ Grant minimum required permissions (principle of least privilege)
✅ Use dataset-level permissions (not project-wide)
✅ Enable audit logging for query access
✅ Rotate service account keys regularly (if using key-based auth)
✅ Prefer Workload Identity over service account keys when possible
""")

def main():
    """Main execution."""
    
    # Verify current access
    success = verify_bigquery_access()
    
    # Print documentation
    print_iam_documentation()
    
    # Final summary
    print("=" * 70)
    print("TASK 3.5 COMPLETE")
    print("=" * 70)
    
    if success:
        print("✅ IAM permissions verified and documented")
        print("\nCurrent Status:")
        print("  - Development: Using ADC (application default credentials)")
        print("  - All required permissions: ✅")
        print("  - Ready for local agent development")
        print("\nNext Steps:")
        print("  1. Validate data access and query performance (Task 3.6)")
        print("  2. For production: Create service account following docs above")
    else:
        print("⚠️  Some permission tests failed")
        print("   Review error messages above and grant required roles")
    
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    import sys
    
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
