#!/usr/bin/env python3
"""
One-Command BigQuery Setup
Orchestrates dataset creation, table creation, and data loading in a single command.

Usage:
    # Setup with defaults (20 customers, 12 months)
    python setup_all.py

    # Custom configuration
    python setup_all.py --customers 50 --months 24

    # Generate CSV only (no BigQuery)
    python setup_all.py --generate-only

    # Force regeneration of data
    python setup_all.py --force
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from google.cloud import bigquery
from google.api_core import exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_dataset_exists(client: bigquery.Client, dataset_ref: str) -> bool:
    """Check if BigQuery dataset exists."""
    try:
        client.get_dataset(dataset_ref)
        return True
    except exceptions.NotFound:
        return False


def check_table_exists(client: bigquery.Client, table_ref: str) -> bool:
    """Check if BigQuery table exists."""
    try:
        client.get_table(table_ref)
        return True
    except exceptions.NotFound:
        return False


def create_dataset_silent(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    location: str
) -> bool:
    """
    Create BigQuery dataset (silent mode, no prompts).
    Returns True if created or already exists.
    """
    dataset_ref = f"{project_id}.{dataset_id}"

    # Check if dataset already exists
    if check_dataset_exists(client, dataset_ref):
        print(f"✅ Dataset '{dataset_id}' already exists")
        return True

    print(f"📦 Creating dataset '{dataset_id}'...", end=" ", flush=True)

    # Create dataset configuration
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = location
    dataset.description = (
        "Synthetic utility usage data for Bill Investigator agent. "
        "Contains hourly energy consumption records with disaggregated loads, "
        "environmental context, and customer profiles."
    )

    try:
        client.create_dataset(dataset, exists_ok=True)
        print("✅")
        return True
    except Exception as e:
        print(f"❌")
        print(f"Error: {e}")
        return False


def create_table_silent(
    client: bigquery.Client,
    project_id: str,
    dataset_id: str,
    table_id: str
) -> bool:
    """
    Create BigQuery table with schema (silent mode, no prompts).
    Returns True if created or already exists.
    """
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Check if table already exists
    if check_table_exists(client, table_ref):
        print(f"✅ Table '{table_id}' already exists")
        return True

    print(f"🗂️  Creating table '{table_id}'...", end=" ", flush=True)

    # Import schema
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_generators.schema import ENERGY_CONSUMPTION_SCHEMA

    # Convert schema to BigQuery format
    bq_schema = []
    for field_name, field_spec in ENERGY_CONSUMPTION_SCHEMA.items():
        field = bigquery.SchemaField(
            name=field_name,
            field_type=field_spec["type"],
            mode=field_spec.get("mode", "NULLABLE"),
            description=field_spec.get("description", "")
        )
        bq_schema.append(field)

    # Create table
    table = bigquery.Table(table_ref, schema=bq_schema)
    table.description = (
        "Hourly energy consumption data with disaggregated loads. "
        "Contains synthetic data for testing Bill Investigator agent."
    )

    try:
        client.create_table(table)
        print(f"✅ ({len(bq_schema)} fields)")
        return True
    except Exception as e:
        print(f"❌")
        print(f"Error: {e}")
        return False


def generate_and_load_data(
    num_customers: int,
    months_back: int,
    output_dir: Path,
    project_id: str = None,
    dataset_id: str = None,
    table_id: str = None,
    generate_only: bool = False,
    force: bool = False
) -> bool:
    """
    Generate synthetic data and optionally load to BigQuery.

    Args:
        num_customers: Number of customer profiles
        months_back: Number of months to generate
        output_dir: Directory for CSV files
        project_id: GCP project ID (required unless generate_only)
        dataset_id: BigQuery dataset ID (required unless generate_only)
        table_id: BigQuery table ID (required unless generate_only)
        generate_only: If True, only generate CSV files
        force: If True, regenerate data without prompting

    Returns:
        True if successful
    """
    # Import data generation and loading functions
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data_generators.csv_exporter import CSVExporter

    # Check if data already exists
    if output_dir.exists() and list(output_dir.glob("customer_*.csv")) and not force:
        print(f"✅ Data already exists in {output_dir}")
        print(f"   Using existing CSV files (use --force to regenerate)")
    else:
        # Generate synthetic data
        print(f"🔄 Generating {num_customers} customers, {months_back} months of data...", end=" ", flush=True)

        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        from dateutil.relativedelta import relativedelta
        start_date = end_date - relativedelta(months=months_back)

        exporter = CSVExporter(output_dir=str(output_dir))

        try:
            generated_files = exporter.generate_full_dataset(
                num_customers=num_customers,
                start_date=start_date,
                end_date=end_date,
                seed=42,
                inject_anomalies=True
            )
            print(f"✅ ({len(generated_files)} files)")
        except Exception as e:
            print(f"❌")
            print(f"Error: {e}")
            return False

    # If generate-only mode, we're done
    if generate_only:
        print(f"\n✅ CSV files ready in {output_dir}")
        return True

    # Load data to BigQuery
    print(f"⬆️  Loading data to BigQuery...", end=" ", flush=True)

    client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    # Get initial row count
    try:
        table = client.get_table(table_ref)
        initial_rows = table.num_rows
    except Exception:
        initial_rows = 0

    # Find all customer CSV files
    csv_files = sorted([
        f for f in output_dir.glob("customer_*.csv")
        if not f.name.startswith("customer_metadata")
    ])

    if not csv_files:
        print(f"❌")
        print(f"No CSV files found in {output_dir}")
        return False

    # Load each file
    total_rows_loaded = 0
    failed = 0

    for csv_file in csv_files:
        try:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.CSV,
                skip_leading_rows=1,
                autodetect=False,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                max_bad_records=10,
            )

            with open(csv_file, "rb") as source_file:
                job = client.load_table_from_file(
                    source_file,
                    table_ref,
                    job_config=job_config
                )

            job.result(timeout=300)

            if job.error_result:
                failed += 1
            else:
                total_rows_loaded += job.output_rows or 0

        except Exception as e:
            failed += 1

    if failed == 0:
        print(f"✅ ({total_rows_loaded:,} rows)")
    else:
        print(f"⚠️  ({total_rows_loaded:,} rows, {failed} files failed)")

    # Verify final row count
    try:
        table = client.get_table(table_ref)
        final_rows = table.num_rows
        print(f"   Total rows in table: {final_rows:,}")
    except Exception:
        pass

    return failed == 0


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="One-command BigQuery setup for Bill Investigator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup with defaults (20 customers, 12 months)
  python setup_all.py

  # Custom configuration
  python setup_all.py --customers 50 --months 24

  # Generate CSV only (no BigQuery)
  python setup_all.py --generate-only

  # Force regeneration of data
  python setup_all.py --force

  # Specify custom output directory
  python setup_all.py --output-dir my_data
        """
    )

    parser.add_argument(
        "--customers",
        type=int,
        default=20,
        help="Number of customer profiles to generate (default: 20)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of months to generate (default: 12)"
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
        help="Only generate CSV files, don't create dataset/table or load data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of data even if it exists"
    )

    return parser.parse_args()


def main():
    """Main execution flow."""
    args = parse_args()

    # Get configuration from environment
    project_id = os.getenv("BIGQUERY_PROJECT_ID")
    dataset_id = os.getenv("BIGQUERY_DATASET", "utility_usage")
    table_id = os.getenv("BIGQUERY_TABLE", "energy_consumption")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Validate environment variables (unless generate-only mode)
    if not args.generate_only and not project_id:
        print("❌ Missing required environment variable: BIGQUERY_PROJECT_ID")
        print("\nPlease set in .env file or export:")
        print("  export BIGQUERY_PROJECT_ID=your-gcp-project-id")
        print("\n💡 Use --generate-only to skip BigQuery setup")
        return False

    # Print header
    print("=" * 70)
    print("BILL INVESTIGATOR - ONE-COMMAND SETUP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Customers: {args.customers}")
    print(f"  Time range: Last {args.months} months")
    print(f"  Output directory: {args.output_dir}")

    if not args.generate_only:
        print(f"  Target: {project_id}.{dataset_id}.{table_id}")
    else:
        print(f"  Mode: CSV generation only")
    print()

    # Step 1: Create dataset (if not generate-only)
    if not args.generate_only:
        try:
            client = bigquery.Client(project=project_id)
        except Exception as e:
            print(f"❌ Failed to initialize BigQuery client: {e}")
            print("\nPlease ensure:")
            print("  1. gcloud CLI is installed and authenticated")
            print("  2. Run: gcloud auth application-default login")
            print("  3. BigQuery API is enabled")
            return False

        if not create_dataset_silent(client, project_id, dataset_id, location):
            return False

        # Step 2: Create table
        if not create_table_silent(client, project_id, dataset_id, table_id):
            return False

    print()

    # Step 3: Generate and load data
    output_dir = Path(args.output_dir)
    success = generate_and_load_data(
        num_customers=args.customers,
        months_back=args.months,
        output_dir=output_dir,
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        generate_only=args.generate_only,
        force=args.force
    )

    if not success:
        return False

    # Print summary
    print()
    print("=" * 70)
    print("✅ SETUP COMPLETE")
    print("=" * 70)

    if args.generate_only:
        print(f"CSV files ready in {output_dir}/")
        print("\nTo load to BigQuery, run without --generate-only:")
        print(f"  python setup_all.py")
    else:
        print(f"Dataset: {project_id}.{dataset_id}")
        print(f"Table: {table_id}")
        print(f"Data: {args.customers} customers, {args.months} months")
        print()
        print("Next steps:")
        print("  1. Start the agent: uv run adk web")
        print("  2. Open browser: http://localhost:8000")
        print("  3. Try a query: \"Why did CUST_001's usage increase in June?\"")

    print("=" * 70)

    return True


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
