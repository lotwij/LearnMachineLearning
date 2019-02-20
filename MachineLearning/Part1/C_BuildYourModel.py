# De scikit-learn library zal worden gebruikt

# Dit zijn destappen voor het bouwen van een model:

#       -- Definieer: Wat voor model zal het worden? Beslisboom? Een
#                     ander type?
#       -- Fit: Vang/bepaal de patronen van de data. Dit is het hart van modelleren.
#       -- Predict: Voorspel nieuwe data
#       -- Evaluate: Bepaal hoe nauwkeurig de voorspellingen zijn

# Hier zal een beslisboom worden gebruikt en gefit

from MachineLearning.Part1.B_DataSelection import X
from MachineLearning.Part1.B_DataSelection import y
from sklearn.tree import DecisionTreeRegressor


# 1) Definieer: Definieer het model. Specificeer ook een nummer voor for random_state om dezelfde resultaten bij
#    elke run te verzekeren
melbourne_model = DecisionTreeRegressor(random_state=1)

# 2) Fit het model (X = variabelen, y=Target)
print(melbourne_model.fit(X, y))

## Nu hebben we het model. Uiteindelijk wil je voorspellen op nieuwe huizen
## die langskomen. Maar we zullen nu eerst voorspellen op de eerste rijden
## van de data die we al hebben. Dit is het 'Trainen' van de data

# 3) Voorspel de eerste 5 rij data:
print("Maak de voorspelling voor de eerste 5 huizen:")
print(X.head())
print("De Voorspelling is:")
print(melbourne_model.predict(X.head()))

