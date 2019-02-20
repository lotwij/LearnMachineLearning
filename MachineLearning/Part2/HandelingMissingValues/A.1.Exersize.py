import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# 1) Laad de data
melb_data = pd.read_csv('melb_data.csv')

# 2) Definieer Target (y) en Predictors (x)
melb_target = melb_data.Price
melb_predictors = melb_data.drop(['Price'], axis=1)

# 3) Voor nu worden alleen de numerieke kolommen gebruikt
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

# 4) Maak train en test sets
X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, melb_target,
                                                    train_size=0.7, test_size=0.3, random_state=0)
# Kolommen in de train_X set:
# u'Rooms', u'Distance', u'Postcode', u'Bedroom2', u'Bathroom', u'Car', u'Landsize',
# u'BuildingArea', u'YearBuilt', u'Lattitude', u'Longtitude',u'Propertycount'

# 5 )kolommen met missende data zijn:
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
print(cols_with_missing)
# Car, BuildingAre, YearBuilt


#################### Nu het dealen met missende data op de 3 manieren:
# Method 1: Verwijder kolommen met missende data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

## MAE: 190342.02689237206

# Method 2 : bereken een gemiddelde waarden voor de missende cellen
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

## MAE: 181947.11471935853


# Method 3 : Laat een model voorspellen wat de missende data zouden moeten zijn
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()
cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)
print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

## MAE: 185545.64707903782


################ Conclusie hier: Methode 2 pakt het beste uit