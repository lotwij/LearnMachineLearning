import pandas as pd
pd.set_option('max_rows', 5)

print("Setup complete.")

#### How to create a data frame with Pandas? Like this:

## Stel zelf een dataframe samen
fruits = pd.DataFrame([[30, 21]], columns=['Apples', 'Bananas'])
print(fruits)

