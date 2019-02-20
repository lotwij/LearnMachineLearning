import pandas as pd
pd.set_option("display.max_rows", 5)
reviews = pd.read_csv("winemag-data-130k-v2.csv", index_col=0)

## Bereken de Mediaan
median_points = median([reviews.points])