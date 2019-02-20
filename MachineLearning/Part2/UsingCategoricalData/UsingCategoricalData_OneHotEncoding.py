import pandas as pd
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 1) Eerst moet je de missende data verwijderen/vervangen
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
target = train_data.SalePrice

# We verwijderen voor nu de kolommen met missende data, maar zoals
# eerder aangegeven, dit is de meest ruwe manier
cols_with_missing = [col for col in train_data.columns
                                 if train_data[col].isnull().any()]
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# Op de volgende wijze pakken maken we een splitsing zodat we in de ene
# dataset alleen numerieke kolommen hebben, in de ander zowel numeriek
# als categorieën. Alle andere type datakolommen worden geexcludeerd
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].nunique() < 10 and
                                candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols

# En nu maken we een train en een test dataset
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

######################################  En dan nu waar het echt om gaat: One hot encoding om ############################
###################################     om te gaan met categorie data   ###############################################


## One hot encoding is een methode die wordt gebruikt voor categorische data. Het zet een kolom
## die lang is in categorieën uit, in een binaire kolom voor elk categorie niveau. Dit werkt erg goed,
## veel gebruikte methode. Alleen wanneer een kolom veel levels heeft (bij > 15) dan werkt het wat minder goed

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score



# 1) Start als je de data hebt gesplits EN de missende data hebt verwijderd. Bekijk nu eerst welke kolommen categorieën zijn:
train_predictors.dtypes.sample(10)


# 2) Voor nu willen we zien wat het effect is van het omzetten van de categorie kolommen. Dit kan op de volgende manier:
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
## Het is HEEL belangrijk om te bedenken dat de train en test set zo op elkaar zijn afgestemd dat de volgore ook veel
## uitmaakt. Die moet hetzelfde blijven voor de train en test set!!
## Daarom maak je de train en test set als volgt:
## 3) maak de train en test set
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left',
                                                                    axis=1)

# Nu gaan 2 modellen maken: 1 met de dataset met omgezette categoriën, en 1 met de dataset die alleen nummerieke data heeft
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50),
                                X, y,
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = final_train.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals))) ## 18532
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded))) ## 18203


