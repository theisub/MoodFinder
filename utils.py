import psycopg2
import psycopg2.extras

import pandas as pd
import os
DATABASE_URL = "postgres://azriecbxgxjxwc:78b498f39df42d1447fc5590f7d739e0ddc208850d47c5cf4468ca8fe131d17e@ec2-52-30-75-37.eu-west-1.compute.amazonaws.com:5432/d7stplin2emuru"

def insert_dataframe(df):
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')

    if len(df) > 0:

        df_columns = list(df)
        columns = ",".join(df_columns)

        values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 

        insert_stmt = "INSERT INTO {} ({}) {}".format('album_info',columns,values)

        cur = conn.cursor()
        psycopg2.extras.exexecute_batch(cur, insert_stmt, df.values)
        conn.commit()
        cur.close()
    print("Data is added!")


def get_records():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = "select * from album_info"
    data_df = pd.read_sql_query(query,conn)
    conn.close()
    return data_df

def create_table():
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    query = """ CREATE TABLE album_info (
            album_id SERIAL PRIMARY KEY,
            album_name TEXT,
            artist_name TEXT,
            album_mood TEXT,
            album_cover TEXT
            );
            """
    curs = conn.cursor()
    try:
        curs.execute(query)
    except:
        print("Something went wrong creating table")
    conn.commit()
    conn.close()
    curs.close()
    print("Creating went successful!")