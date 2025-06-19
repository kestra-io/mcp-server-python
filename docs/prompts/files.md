
Find CSV files in the namespace company.team 
Find files in the namespace company.team that include etl

Create a directory scripts in company.team namespace
Create a directory scripts/etl in company.team namespace
Create a directory queries in company.team namespace
Create a directory queries/dbt in company.team namespace
Move the directory queries/dbt to the dbt directory in company.team namespace

Create a file products.json in company.team namespace with this content:
{
  "products": [
    {
      "id": 1,
      "title": "Essence Mascara Lash Princess",
      "category": "beauty",
      "price": 9.99,
      "discountPercentage": 10.48,
      "brand": "Essence",
      "sku": "BEA-ESS-ESS-001"
    },
    {
      "id": 2,
      "title": "Eyeshadow Palette with Mirror",
      "category": "beauty",
      "price": 19.99,
      "discountPercentage": 18.19,
      "brand": "Glamour Beauty",
      "sku": "BEA-GLA-EYE-002"
    }
  ]
}

Create a directory data in company.team namespace
Move the file products.json to data/products.json in namespace company.team
Delete the data/products.json file in namespace company.team
Delete the data directory in namespace company.team

Create a file scripts/etl.py in company.team namespace with this content:
import pandas as pd
import sqlite3

def extract_data(csv_path):
    return pd.read_csv(csv_path)

def transform_data(df):
    df['full_name'] = df['first_name'] + ' ' + df['last_name']
    df = df[df['age'] > 18]
    return df[['full_name', 'email', 'age']]

def load_data(df, db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists='replace', index=False)

if __name__ == "__main__":
    raw_data = extract_data("users.csv")
    cleaned_data = transform_data(raw_data)
    load_data(cleaned_data, "users.db", "adult_users")

List files in the scripts directory in company.team namespace
Fetch the file scripts/etl.py in company.team namespace

