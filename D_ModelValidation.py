# Om na tegaan of een model goed is, wil je het valideren.
# Valideren op de trainingsset zou niet eerlijk zijn, en dus
# is er een aparte valideer set nodig

# Bij het valideren krijg je een lijst voorspellingen, zowel goede als slechte.
# Deze allemaal doorkijken leert je niet zoveel. Om de nauwkeurigheid van
# een model te bepalen kun je daarom het beste naar de
# Mean Absolute Error (MAE) kijken

# Om dit op te breken:

## Error = actual - predicted
# hiermee krijg je alle errors. Hier worden absoluut getallen van gemaakt (allemaal positief)
# En hier wordt het gemiddelde van genomen

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from B_DataSelection import X
from B_DataSelection import y
from C_BuildYourModel import melbourne_model

# Nu we het model hebben zouden we de MAE kunnen berekenen
predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# 1) Maar nu hebben we het op de trainings data gedaan. Dat is niet de goede weg
#    Op de volgende manier breken we de dataset op in een training en validation set
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# 2 ) Definieer het nieuwe model met nu enkel de train set
melbourne_model = DecisionTreeRegressor()

# 3) Fit het model met enkel de train set
melbourne_model.fit(train_X, train_y)

# 4) En nu waar het om gaat: valideer het model, bereken de MAE
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))













