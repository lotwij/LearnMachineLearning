import pandas as pd
from sklearn.datasets import load_wine
pd.set_option('max_rows', 5)

print("Setup complete.")

################## How to create OR Load a data frame with Pandas? Like this:

## Stel zelf een dataframe samen
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])
print(fruits)
fruits

## Stel een dataframe samen met benaming in de index kolom
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
print(animals)

## Lees een CSV File in
#>>reviews = pd.read_csv('../input/..../filename.csv', index_col=0)

## Lees een SQL datalijst in
import sqlite3
#>>conn = sqlite3.connect("../input/pitchfork-data/database.sqlite")
#>>music_reviews = pd.read_sql_query("SELECT * FROM artists", conn)


