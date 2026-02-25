# Handling inconsistent values
import pandas as pd
import fuzzywuzzy
from fuzzywuzzy import process

data_path = "../Datasets/Pakistan Intellectual Capital/pakistan_intellectual_capital.csv"
professors = pd.read_csv(data_path)

unis = professors['Graduated from'].unique()
# print(unis)

# There are unnecessary white spaces in this column names, so remove them
professors['Graduated from'] = professors['Graduated from'].str.strip()
# print(professors['Graduated from'])

countries = professors['Country'].unique()
countries = sorted(professors['Country'].unique())
# print(countries)
professors['Country'] = professors['Country'].str.strip()
countries = professors['Country'].unique()
print(countries)

# If there are columns that indicate the same country but has different column names, make them consistent
matches = fuzzywuzzy.process.extract("usa", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)
# replace_matches_in_column(df=professors, column='Country', string_to_match="usa", min_ratio=70)