import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text, inspect
import pymysql


load_dotenv()

def create_con():
    sql_file_path = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], "250123.sql")
    with open(sql_file_path, "r") as f:
        for i in f:
            yield i
    # pymysql.install_as_MySQLdb()
    # engine = create_engine("mysql+pymysql://root@localhost/foo")
    # inspector = inspect(engine)
    # print(inspector)
    
    # print(sql_file_path)
    # with engine.connect() as con:
    # with open(sql_file_path, "r") as file:
    #     query = text(file.read())
    #         con.execute(query)
