import torch
import torch.nn as nn


import pandas as pd
import sqlite3 as sql

database = sql.connect("ML\\data\\raw\\wjazzd.db")

melody = pd.read_sql("SELECT melid, onset,pitch FROM melody ORDER BY melid, onset", database)
print (melody)