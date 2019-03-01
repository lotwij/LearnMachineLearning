import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


## XGBoosting pakt een eerst gegeven model (naief model) en maakt hier een model
## van. Vervolgens berekend deze de MAE. Daarna maakt deze nog een model en berekend
## ook hier weer de MAE. Uiteindelijk pakt XGBoosting dus heel veel modellen en kan
## met deze bak aan informatie heel precieze modellen maken. XGBoosting is daarom
## een stuk preciezer. Je doet het als volgt:

# 1) eerst laad je de data, dealt met missende data
#       en  breek je de data op in train en test data
data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# 2) Net als in het sklearn pakket bouwen we het model, nu dus
#       het naieve model

from xgboost import XGBRegressor
my_model = XGBRegressor()
# met silent=True zorg je dat niet alle cycle data wordt uitgeprint
my_model.fit(train_X, train_y, verbose=False)

# 3) Net als eers laten we het model voorspellingen maken en beoordelen
#    we het model op basis van MAE
# make predictions
predictions = my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
#### MAE: 18826.185498715753

### Dit is nog geen geweldig model. Nu kun je gaan finetunen. er zijn 2 termen belangrijk:

# n_estimators en early_stopping_rounds
# n_estimators hoevaak je hem door de cyclus wil laten gaan van finetunen adhv errors

# In de underfitting vs overfitting grafiek, n_estimators laat je meer naar de overfitting
# kant (rechts) gaan
# Een te lage waarde zorgt voor underfitting, die onnauwkeurig zal voorspellen op zowel
# de train als test data. Een te hoge waarde zorgt voor overfitten: een goede voorspelling
# op de train data, maar een slechte voorspelling op de test data (en daar gaat het om).
# Je moet dus de ideale waarde vinden voor de n_estimator. Typical varieren waardes tussen 100-1000,
# Maar dit hangt veel af van het volgende:

# De early_stopping_rounds kan automatische de juiste waarde vinden voor n_estimator
# Early stopping zorgt dat de cyclus (iteraties) stoppen wanneer de validatie waarde
# omlaag gaat. Hierom is het dus wijs om een hoge n_estimator in te zetten, en dan de
#  early_stopping_rounds te gebruiken om de ideale waarde te bepalen.

#Omdat soms bij puur geluk de eerste ronde al goed is, geef je het aantal aan dat
#die aangeeft, hoeveel rondes zonder verandering je wil toevoegen (dit is dus early_stopping_rounds).
# early_stopping_rounds = 5 is vaak een goede waarde. Dus wanneer na 5 x finetunen er geen verbetering
#meer is, zal alles stoppen.


## 4) Dit kan je doen met de volgende code:
my_model = XGBRegressor(n_estimators=10000)
print(my_model.fit(train_X, train_y, early_stopping_rounds=5,
             eval_set=[(test_X, test_y)], verbose=False))
#XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
#       max_depth=3, min_child_weight=1, missing=None, n_estimators=1000,
#       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#       silent=True, subsample=1)

## 5 )Check nu nog een keer de MAE:
predictions = my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
#Mean Absolute Error : 16480.71838613014 << lagere MAE!!

## INFO: wanneer je early_stopping_rounds gebruikt, moet je een deel van je
## data apart houden om te checken hoeveel rondes je moet gebruiken. Als je daarna
## je model wil fitten op al je data, zet dan n_estimators naar de optimale
## waarde in combi met early_stopping

## Er is een extra truce voor GXBoosting: LEARNING RATE
#Inplaats van de voorspellingen van elk model bij elkaar op te tellen,
#kunnen we voorspellingen menigvuldigen met een klein getal voor frozenseterbij te tellen.
#Dit houdt eigeblijk in, dat elk model maar een kleine bijdrage heeft. Dit zorgt
#er weer voor dat elke keer het eind model maar een klein beetje bijsteld. Hierdoor
#kun je de n_estimator veel hoger zetten (meer runs!!!) zonder dat je het risico
#loopt op overfitten. Door ook gebruik te maken van early_stopping krijg je
#automatisch het beste model
# Dit zorgt voor betere modellen!! Een nadeel: de runtime is veel langer.


## 6) De volgende code kun je gebruiken voor met nu ook de learning_rate:
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
print(my_model.fit(train_X, train_y, early_stopping_rounds=5,
             eval_set=[(test_X, test_y)], verbose=False))
## 7 )Check nu nog een keer de MAE:
predictions = my_model.predict(test_X)
from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
#Mean Absolute Error : 16228.345205479452 << NOG lagere MAE!!


###### Nog een extra'tje voor grote datasets:
## n_jobs
# Met n_jobs kun je paralelle runs aanzetten om een model sneller
#tot stand te laten komen. Normaal zet je dit aantal gelijk
#aan het aantal processors van je computer. Voor kleine datasets
#heeft dit geen zin. Het resultaat is er ook niet door beinvloed,
#maar voor grote dataset kan het je wel een hoop tijd besparen.

#XGBoosting heeft zeker nog meer opties. Maar hierboven staan vast de basics
#om een heel eind te komen

print(data.columns)

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

my_plots = plot_partial_dependence(my_model,
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
