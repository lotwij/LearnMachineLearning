################## How to filter dataframe

reviews = pd.read_csv('winemag-data-130k-v2.csv')
print(reviews.description)

## Selecteer rijen/gegevens met specifieke indexen, en alleen bepaalde kolommen
cols = ['country', 'province', 'region_1', 'region_2']
indices = [0, 1, 10, 100]
df = reviews.loc[indices, cols]

## Filter op 1 kolom en 1 niveau
italian_wines = reviews[reviews.country == "Italy"]

## Filter op 2 kolommen en meerdere niveaus:
oceania = ['Australia','New Zealand']
top_oceania_wines = reviews[(reviews.country.isin(oceania)) & (reviews.points >= 95)]