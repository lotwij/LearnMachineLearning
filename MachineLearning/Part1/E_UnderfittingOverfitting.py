

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from MachineLearning.Part1.B_DataSelection import X
from MachineLearning.Part1.B_DataSelection import y
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# Bij een beslisboom is het goed uit te leggen wat over en onderfitten is:
# Overfitting: té veel splitsingen en eindgroepen (leaves) waardoor er maar weinig datapunten
# per eindgroep zal zijn. Hierdoor is je model wel precies maar zal weinig
# voorspellende waarde hebben
# Onderfitten: te weinig splitsingen waardoor je de per eindgroep (leaf) te algemeen blijft
# en hier is dus ook een lage voorspellende waarde. Hierbij is de

## Over en onderfitten kunnen dus in elkaar overgaan, je bent op zoek naar het (midden) gebied
## waarbij je MAE op z'n laagst is, en er precies genoeg groepen in je leaves zit:
## Geen overfitting én geen onderfitting

# Je kunt hiermee testen door de max aantal toegestane eindgroepen (leaves) te varieren. Dit
# kun je als volgt doen:

# 1) eerst een mooie functie om de beslisboom in een keer te doen + MAE te berekenen
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# 2) Dan de loop om het aantal leaves te varieren: 5, 50, 500 of 5000:
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

# Max leaf nodes: 5  		 Mean Absolute Error:  385696
# Max leaf nodes: 50  		 Mean Absolute Error:  279794
# Max leaf nodes: 500  		 Mean Absolute Error:  261718 <--- Laagste MAE dus beste keuze
# Max leaf nodes: 5000  	 Mean Absolute Error:  271996

# Overfitting: Vangen sporadische patronen die niet meer terugkomen in de toekomst, dus minder accurate voorspellingen
# Underfitting: Falen om relevante patronen te zien, dus ook hier minder accurate voorspellingen

# 3) eens kijken of we nog preciezer kunnen gaan:
for max_leaf_nodes in range(400,600):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    #     Max leaf nodes: 544  		 Mean Absolute Error:  259958 <-- 544 is beter

# 3) Dit kun je aangeven in je decision tree, en daarna je model op álle data fitten
final_model = DecisionTreeRegressor(max_leaf_nodes=544)
final_model.fit(X, y)
