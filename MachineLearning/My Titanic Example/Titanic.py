######################################### A Data Exploration
import pandas as pd
from xgboost import XGBRegressor
# Lees de data in en bewaar onder een naam
titanic_data = pd.read_csv('train.csv')
#print de data summary
print(titanic_data.describe())


######################################## Missende data
new_data = titanic_data.copy()

#  Maak een kolom die aangeeft welke aangepast zullen worden
cols_with_missing = (col for col in new_data.columns
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

new_data.to_dense().to_csv("submission.csv", index = False, sep=',', encoding='utf-8')
# Conslusie Cabin heeft missende data maar deze kan best weggelaten worden


######################################### B Data Selection
print(titanic_data.columns)

# We droppen geen NA want de kolommen die NA bevatten kunnen we
# Als eerste hele kolom laten vallen
# Selecteer je Target data kolom, wat je wilt voorspellen: Survived
y = titanic_data.Survived
# Selecteer de features
titanic_features = ['Pclass','Sex','SibSp','Parch','Embarked','Fare', 'Age']
X = titanic_data[titanic_features]
#  Bekijk nu de eigenschappen van de geselecteerde variabelen
print(X.describe())
# En bekijk ook hoe de data in de cellen staat:
print(X.head())
# Split de dataset

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0)

### Check verder voor onehot encoding:
## https://medium.com/hugo-ferreiras-blog/dealing-with-categorical-features-in-machine-learning-1bb70f07262d