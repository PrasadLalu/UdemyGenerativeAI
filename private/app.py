import pandas as pd
import psycopg2
from psycopg2 import sql

# Configuration
TABLE_NAME = "sample_demand"
CHUNK_SIZE = 1000
DATABASE_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

# Read dataset
data = pd.read_excel("../datasets/sample_data.xlsx")
# data = pd.read_excel("../datasets/Demand_data.xlsx")

# Normalize column names and select specific columns
selected_columns = ['prospectId', 'prospectName', 'groupId', 'groupName', 'productType', 'prospectMobile', 'centreName', 'Region', 'Bank']
data.columns = [col.strip().replace(" ", "") for col in data.columns]
filtered_data = data[selected_columns]

def insert_data_chunk(chunk):
    # Establish database connection
    conn = psycopg2.connect(**DATABASE_CONFIG)
    cursor = conn.cursor()

    # Prepare the query with hardcoded fields
    insert_query = """
        INSERT INTO sample_demand (prospectId, prospectName, groupId, groupName, productType, prospectMobile, centreName, Region, Bank)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    try:
        # Convert chunk to list of tuples
        records = [tuple(row) for row in chunk.to_numpy()]
        # Execute the insert query
        cursor.executemany(insert_query, records)
        conn.commit()
        print(f"Inserted {len(records)} rows successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting data: {e}")
    finally:
        cursor.close()
        conn.close()

# Split the data into chunks
chunks = [filtered_data.iloc[i:i + CHUNK_SIZE] for i in range(0, len(filtered_data), CHUNK_SIZE)]

# Insert each chunk sequentially
for chunk in chunks:
    insert_data_chunk(chunk)
