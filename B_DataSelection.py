#Vaak hebben dataset té veel variabele om direct iets mee te doen
# Dus wat kan je ermee doen om meer van de data te begrijpen?

# We gaan eerst data selecteren op basis van intuitie.
# Later zullen technieken naar voren komen die je hierbij
# kunnen helpen.

# Eerst hebben we een lijst van álle variabele nodig.

import pandas as pd

# 1) Laadt eerst de data weer in
melbourne_data = pd.read_csv('melb_data.csv')

# 2) Dan printen we de variabele/kolommen lijst uit
print(melbourne_data.columns)

# 3) Drop de NA waardes
# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# 4) Selecteer je Target data kolom, wat je wilt voorspellen
y = melbourne_data.Price

# 5) En dan nu de variabelen. Dit is de wijze waarop je de variabele apart
#    kunt selecteren. Nu pakken we een paar maar dit is afhankelijk van de case
#    Soms kunnen veel variabelen een beter beeld geven, soms juist minder
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

# 6) Bekijk nu de eigenschappen van de geselecteerde variabelen
print(X.describe())
# En bekijk ook hoe de data in de cellen staat:
print(X.head())

#### Je kunt je data ook nog verder visualiseren




