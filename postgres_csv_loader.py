#!/usr/bin/python3

import pandas as pd
import sys
import os
from subprocess import check_output
from sqlalchemy import create_engine
from dotenv import load_dotenv
import io
load_dotenv()

'''
    Function 
'''

arguments_error = r'''
----------------------------------------------------------------------------------
    Script takes 2 arguments: filename and tablename (which you want to create)
    Example:
        ./db_init.py filename tablename

    To get correct connection to database, you need an .env file and running database
-----------------------------------------------------------------------------------
'''
file_error = r'''
-------------------------------------
    There is no such file. Check!
-------------------------------------
'''
success = r'''
-------------------------------------------------------
    The data from file succeccfully loaded to database.
-------------------------------------------------------
'''
LIST_OF_DIR = check_output('ls -a', shell=True).decode().split('\n')

# ---- argv --------------
ARGVS = sys.argv

# ------- Check amount of arguments and .env file --------------------
if (len(ARGVS) != 3) | ('.env' not in LIST_OF_DIR): 
    print(arguments_error)
    exit()

FILE = ARGVS[1]
TABLE_NAME = ARGVS[2]

# ------- Check file exists ---------------------------------------
if not os.path.isfile(FILE):
    print(file_error)
    exit()

try:
    credit = pd.read_csv(FILE)
except:
    print("Can't load file.. Unknown error!")
    exit()

try:
    url = f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv('DB_NAME')}'
except:
    print("Can't connect to db..")
else:
    engine = create_engine(url=url, echo=False)
    credit.to_sql(TABLE_NAME, engine, index=False, if_exists='replace')
finally:
    print(success)
    exit()
