from sqlalchemy import create_engine
import io
import psycopg2
import pandas as pd
def insert_dataframe(df):
    engine = create_engine('postgresql+psycopg2://postgres:ident@db:5432/RYM')


    if len(df) > 0:
        conn = engine.raw_connection()

        df_columns = list(df)
        columns = ",".join(df_columns)

        values = "VALUES({})".format(",".join(["%s" for _ in df_columns])) 

        insert_stmt = "INSERT INTO {} ({}) {}".format('album_info',columns,values)

        cur = conn.cursor()
        psycopg2.extras.execute_batch(cur, insert_stmt, df.values)
        conn.commit()
        cur.close()
    print("Data is added!")


def get_records():
    conn = psycopg2.connect(database="RYM",user="postgres", password="ident", host="db",port=5432)
    query = "select * from album_info"
    data_df = pd.read_sql_query(query,conn)
    conn.close()
    return data_df

def create_table():
    conn = psycopg2.connect(database="RYM",user="postgres", password="ident", host="db",port=5432)
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