"""
data_loader.py
--------------
PURPOSE: Load raw CSV data into a SQLite database.

WHY DO THIS? In production (like at Bayut), data lives in a SQL warehouse.
By storing our data in SQLite first and querying it with SQL, we practice
the exact same workflow used in the real job.
"""

import pandas as pd
import sqlite3
import os
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────
# Path() gives us a clean, OS-independent way to handle file paths
# This works on Windows, Mac, and Linux without changes
ROOT_DIR = Path(__file__).parent.parent  # Goes up two levels from src/
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "AmesHousing.csv"
DB_PATH = ROOT_DIR / "data" / "propiq.db"


def load_csv_to_sqlite(csv_path: Path, db_path: Path, table_name: str = "listings") -> None:
    """
    Reads a CSV file and writes it into a SQLite database table.

    Parameters
    ----------
    csv_path : Path to the source CSV file
    db_path  : Path where the SQLite database will be created
    table_name: Name of the table inside the database
    """
    print(f"📂 Reading CSV from: {csv_path}")
    
    # Read the CSV into a Pandas DataFrame
    # A DataFrame is like an Excel table in Python
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"📋 Columns: {list(df.columns[:10])}... and {len(df.columns)-10} more")

    # Connect to SQLite database (creates the file if it doesn't exist)
    print(f"\n💾 Writing to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)

    # Write the DataFrame to a SQL table
    # if_exists='replace' means: if the table already exists, overwrite it
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.close()
    print(f"✅ Table '{table_name}' created in database successfully!")


def query_database(sql_query: str, db_path: Path) -> pd.DataFrame:
    """
    Runs any SQL query against our database and returns a DataFrame.
    
    This is the core function that simulates querying a data warehouse.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # It won't run when you import the functions into another file
    
    load_csv_to_sqlite(RAW_DATA_PATH, DB_PATH)

    # Test it — run a SQL query just like you would on a real data warehouse
    print("\n🔍 Running test SQL query...")
    
    test_query = """
        SELECT 
            Neighborhood,
            COUNT(*) as total_listings,
            ROUND(AVG(SalePrice), 2) as avg_price,
            MIN(SalePrice) as min_price,
            MAX(SalePrice) as max_price
        FROM listings
        GROUP BY Neighborhood
        ORDER BY avg_price DESC
        LIMIT 10
    """
    
    result = query_database(test_query, DB_PATH)
    print("\n📊 Top 10 Neighborhoods by Average Price:")
    print(result.to_string(index=False))