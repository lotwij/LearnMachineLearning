from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from B_DataSelection import X
from B_DataSelection import y
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

## Er zijn meer manieren om een model te maken. Een beslisboom is vrij simpel,
## Random Forest heeft vaak een hogere preciezie. Random Forest neemt het gemiddelde
## van alle voorspellende "Trees" en maakt hier een combi van: een beter model


# 1) het maken van een model gaat ongeveer hetzelfde als bij de de beslisboom:
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
# 198635
## Dit is al een grote verbetering vergeleken met MAE: 259958 van de beslisboom.