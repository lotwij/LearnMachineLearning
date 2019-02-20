## Missende kan ontstaan doordat de data er wel is, maar niet geregistreetd, of doordat de data inderdaad
## niet bestaat. Bijvoorbeeld, een woonruimte met 2 kamers zal een lege cel hebben bij de grootte van de
## derde kamer.


## Meeste machine learning methodes kunnen niet omgaan met missende data. Daarom kun je ze er het beste uitfilteren

# 1) detecteer en tel de lege cellen op de volgende manier:
missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0


# 2) nu heb je 2 opties: alle missende data weghalen óf een waarde berekenen die niet hélemaal klopt
#    maar die wel de leegte kan vullen. De tweede optie geeft vaak betere voorspellingen. Hieronder
#    de methodes voor beide opties:

            # A) Verwijder alle kolommen met  missende data
## Als er 1 dataset is:
data_without_missing_values = original_data.dropna(axis=1)
## Als je een train en test set hebt:
cols_with_missing = [col for col in original_data.columns
                                 if original_data[col].isnull().any()]
redued_original_data = original_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing, axis=1)

# >> Wanneer je door deze methode kolommen verwijderd met erg veel missende data en weinig waarde
#    is dit een goede methode. Echter, wanneer je hierdoor kolommen moet verwijderen die een paar
#    cellen missen en wel belangrijke info bevatten, kun je beter voor optie B gaan:

            # B) Met deze methode worden lege cellen opgevuld met een soort gemiddelde waarde:
                        # 1)
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(original_data)

                        # 2) die methode 1 werkt goed als de missende data inderdaad goed aangevuld kunnen
                                # worden met een gemiddelde, dit is meestal ook het geval. Soms werkt die methode
                                # alleen niet goed. Bijvoorbeeld als de rijen met missende data op een
                                # bepaalde manier uniek zijn. In dat geval zou je model betere voorspellingen maken,
                                #   In dat geval kun je deze methode gebruiken:
                        # 2a) Maak een kopie van je originele data om deze niet te veranderen:
new_data = original_data.copy()

                        # 2b) Maak een kolom die aangeeft welke aangepast zullen worden
cols_with_missing = (col for col in new_data.columns
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

                        # 2c) Gebruik een fit model om de missende data te voorspellen
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data.columns = original_data.columns