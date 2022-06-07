from sqlalchemy import create_engine
import io
import psycopg2

def insert_dataframe(df):
    engine = create_engine('postgresql+psycopg2://postgres:ident@localhost:5432/RYM')

    df.head(0).to_sql('album_info', engine, if_exists='replace',index=False) #drops old table and creates new empty table

    conn = engine.raw_connection()
    cur = conn.cursor()
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, 'album_info', null="") # null values become ''
    conn.commit()

def create_table():
    conn = psycopg2.connect(database="RYM",user="postgres", password="ident", host="localhost",port=5432)
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