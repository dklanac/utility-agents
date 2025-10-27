#!/usr/bin/env python3
"""
BigQuery Data Loading Script (Task 3.4)
Generates synthetic data and loads it into BigQuery.
"""

import os
import sys
import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_synthetic_data(
    num_customers: int = 15,
    start_date: datetime = None,
    end_date: datetime = None,
    output_dir: Path = None,
    months_back: int = 6
):
    """
    Generate synthetic energy consumption data.

    Args:
        num_customers: Number of customer profiles to generate
        start_date: Start date for data generation (defaults to months_back from today)
        end_date: End date for data generation (defaults to today)
        output_dir: Directory to save CSV files
        months_back: Number of months to generate if start_date not specified
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_generators.csv_exporter import CSVExporter

    # Calculate relative dates if not provided
    if end_date is None:
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if start_date is None:
        start_date = end_date - relativedelta(months=months_back)
    if output_dir is None:
        output_dir = Path("billing_investigator/support/data_files")
    
    print("=" * 70)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 70)
    print(f"Customers: {num_customers}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Duration: {(end_date - start_date).days} days")
    print(f"Output: {output_dir}")
    print()
    
    # Create exporter
    exporter = CSVExporter(output_dir=str(output_dir))
    
    # Generate dataset
    generated_files = exporter.generate_full_dataset(
        num_customers=num_customers,
        start_date=start_date,
        end_date=end_date,
        seed=42,
        inject_anomalies=True
    )
    
    return generated_files

def load_csv_to_bigquery(
    csv_path: Path,
    table_ref: str,
    client: bigquery.Client
) -> bigquery.LoadJob:
    """
    Load a single CSV file into BigQuery.
    
    Args:
        csv_path: Path to CSV file
        table_ref: BigQuery table reference
        client: BigQuery client
        
    Returns:
        Load job
    """
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,  # Skip header row
        autodetect=False,  # Use existing schema
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Append to table
        max_bad_records=10,  # Allow some bad records
    )
    
    # Load data
    with open(csv_path, "rb") as source_file:
        job = client.load_table_from_file(
            source_file,
            table_ref,
            job_config=job_config
        )
    
    return job

def load_all_data_to_bigquery(
    data_dir: Path,
    project_id: str,
    dataset_id: str,
    table_id: str
):
    """
    Load all customer CSV files into BigQuery.
    
    Args:
        data_dir: Directory containing CSV files
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
    """
    print("=" * 70)
    print("LOADING DATA TO BIGQUERY")
    print("=" * 70)
    
    # Initialize client
    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    print(f"Target table: {table_ref}")
    
    # Get initial row count
    try:
        table = client.get_table(table_ref)
        initial_rows = table.num_rows
        print(f"Current rows: {initial_rows:,}")
    except Exception as e:
        print(f"⚠️  Could not get initial row count: {e}")
        initial_rows = 0
    
    print()
    
    # Find all customer CSV files
    csv_files = sorted([
        f for f in data_dir.glob("customer_*.csv")
        if not f.name.startswith("customer_metadata")
    ])
    
    if not csv_files:
        print(f"❌ No customer CSV files found in {data_dir}")
        return False
    
    print(f"Found {len(csv_files)} customer files to load")
    print()
    
    # Load each file
    successful = 0
    failed = 0
    total_rows_loaded = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"[{i}/{len(csv_files)}] Loading {csv_file.name}...", end=" ", flush=True)
        
        try:
            # Load file
            job = load_csv_to_bigquery(csv_file, table_ref, client)
            
            # Wait for job to complete
            job.result(timeout=300)  # 5 minute timeout
            
            # Check for errors
            if job.error_result:
                print(f"❌")
                print(f"  Error: {job.error_result}")
                failed += 1
            else:
                rows_loaded = job.output_rows or 0
                total_rows_loaded += rows_loaded
                print(f"✅ ({rows_loaded:,} rows)")
                successful += 1
                
        except Exception as e:
            print(f"❌")
            print(f"  Error: {e}")
            failed += 1
    
    print()
    print("=" * 70)
    print("LOAD SUMMARY")
    print("=" * 70)
    print(f"Files processed: {len(csv_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total rows loaded: {total_rows_loaded:,}")
    
    # Verify final row count
    print("\n" + "=" * 70)
    print("VERIFYING DATA")
    print("=" * 70)
    
    try:
        table = client.get_table(table_ref)
        final_rows = table.num_rows
        new_rows = final_rows - initial_rows
        
        print(f"✅ Data loaded successfully")
        print(f"   Initial rows: {initial_rows:,}")
        print(f"   Final rows: {final_rows:,}")
        print(f"   New rows: {new_rows:,}")
        print(f"   Table size: {table.num_bytes / 1024 / 1024:.2f} MB")
        
        # Run sample query
        print("\n" + "=" * 70)
        print("SAMPLE QUERY")
        print("=" * 70)
        
        query = f"""
        SELECT 
            customer_id,
            COUNT(*) as record_count,
            MIN(timestamp) as first_record,
            MAX(timestamp) as last_record,
            ROUND(AVG(total_usage_kwh), 2) as avg_usage_kwh
        FROM `{table_ref}`
        GROUP BY customer_id
        ORDER BY customer_id
        LIMIT 5
        """
        
        query_job = client.query(query)
        results = query_job.result()
        
        print("\nFirst 5 customers:")
        print(f"{'Customer':<12} {'Records':<10} {'First Record':<20} {'Last Record':<20} {'Avg kWh'}")
        print("-" * 85)
        
        for row in results:
            print(f"{row.customer_id:<12} {row.record_count:<10,} "
                  f"{str(row.first_record):<20} {str(row.last_record):<20} {row.avg_usage_kwh}")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    return True

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic energy data and load to BigQuery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate last 6 months from today (default)
  python load_data_to_bigquery.py

  # Generate last 3 months from today
  python load_data_to_bigquery.py --months 3

  # Generate for specific date range
  python load_data_to_bigquery.py --start-date 2025-04-01 --end-date 2025-10-24

  # Generate 20 customers for last 12 months
  python load_data_to_bigquery.py --customers 20 --months 12

  # Only generate data, don't load to BigQuery
  python load_data_to_bigquery.py --generate-only
        """
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=15,
        help="Number of customer profiles to generate (default: 15)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Number of months to generate from end date (default: 6)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DD format (overrides --months)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="billing_investigator/support/data_files",
        help="Output directory for CSV files (default: billing_investigator/support/data_files)"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate CSV files, don't load to BigQuery"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate data without prompting if output directory exists"
    )

    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()

    # Parse dates if provided
    start_date = None
    end_date = None

    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Invalid end date format: {args.end_date}")
            print("   Use YYYY-MM-DD format (e.g., 2025-10-24)")
            return False

    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Invalid start date format: {args.start_date}")
            print("   Use YYYY-MM-DD format (e.g., 2025-04-01)")
            return False

    # Get BigQuery configuration
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET")
    table_id = os.getenv("BIGQUERY_TABLE")

    if not args.generate_only and not all([project_id, dataset_id, table_id]):
        print("❌ Missing required environment variables")
        print("   BIGQUERY_PROJECT_ID, BIGQUERY_DATASET, BIGQUERY_TABLE")
        print("\n💡 Use --generate-only to skip BigQuery loading")
        return False

    print("=" * 70)
    print("BIGQUERY DATA LOADING (Task 3.4)")
    print("=" * 70)
    if not args.generate_only:
        print(f"Target: {project_id}.{dataset_id}.{table_id}")
    else:
        print("Mode: Generate CSV files only (no BigQuery loading)")
    print()

    # Check if data already exists
    data_dir = Path(args.output_dir)

    if data_dir.exists() and list(data_dir.glob("customer_*.csv")):
        if args.force:
            print(f"⚠️  Data directory exists: {data_dir}")
            print("Regenerating data (--force flag set)...")
            generated_files = generate_synthetic_data(
                num_customers=args.customers,
                start_date=start_date,
                end_date=end_date,
                output_dir=data_dir,
                months_back=args.months
            )
        else:
            print(f"⚠️  Data directory already exists: {data_dir}")
            response = input("Use existing data or regenerate? (use/regen): ")

            if response.lower() == "regen":
                print("Regenerating data...")
                generated_files = generate_synthetic_data(
                    num_customers=args.customers,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=data_dir,
                    months_back=args.months
                )
            else:
                print("Using existing data")
                generated_files = None
    else:
        # Generate synthetic data
        generated_files = generate_synthetic_data(
            num_customers=args.customers,
            start_date=start_date,
            end_date=end_date,
            output_dir=data_dir,
            months_back=args.months
        )
    
    print()

    # Load data to BigQuery (unless --generate-only flag is set)
    if args.generate_only:
        print("=" * 70)
        print("✅ DATA GENERATION COMPLETE")
        print("=" * 70)
        print(f"CSV files saved to: {data_dir}")
        print("\nTo load data to BigQuery, run without --generate-only flag")
        print("=" * 70)
        return True

    success = load_all_data_to_bigquery(
        data_dir=data_dir,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )

    if success:
        print("\n" + "=" * 70)
        print("✅ DATA LOADING COMPLETE")
        print("=" * 70)
        print(f"Next steps:")
        print(f"  1. Set up IAM permissions for ADK agent (Task 3.5)")
        print(f"  2. Validate query performance (Task 3.6)")
        print("=" * 70)

    return success

if __name__ == "__main__":
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
