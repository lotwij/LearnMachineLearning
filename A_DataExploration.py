## Gebruik pandas om de data te kunnen bekijken en aanpassen

import pandas as pd

# 1) Lees de data in en bewaar onder een naam
melbourne_data = pd.read_csv('melb_data.csv')


# 2) print de data summary
print(melbourne_data.describe())

# >> Uitleg:
# In de summary zijn er 8 getallen te zien.
# De eerste laat het aantal missende data zien. Daar wordt op teruggekomen

# De 2e kolom is het gemiddelde
# De 3e kolom is de standaard deviatie, dat de spreiding van de data aangeeft
# Daarna is de min, 25%, 50%, 75% and max waarde te zien.
                                                                                                                                   ' The 50th and 75th percentiles are defined analogously, and the max is the largest number