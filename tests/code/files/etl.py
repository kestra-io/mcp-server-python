import pandas as pd
import sqlite3


def extract_data(csv_path):
    return pd.read_csv(csv_path)


def transform_data(df):
    df["full_name"] = df["first_name"] + " " + df["last_name"]
    df = df[df["age"] > 18]
    return df[["full_name", "email", "age"]]


def load_data(df, db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    raw_data = extract_data("users.csv")
    cleaned_data = transform_data(raw_data)
    load_data(cleaned_data, "users.db", "adult_users")
