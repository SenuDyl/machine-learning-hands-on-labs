# Indexing, Selecting and Assigning
first_row = reviewers.iloc[0]
first_descriptions = reviewers.loc[:9, "descriptions"]
#or
first_descriptions = reviewers.iloc[:10, "descriptions"]

df = reviewers.iloc[:99, ["country", "variety"]]

# Summary functions and maps
median_points = reviews.points.median()
countries = reviews.country.unique()
reviews_per_country = reviews.country.value_counts()
mean_price = reviews.price.mean()

# Counting missing values
n_missing_prices = reviews.price.isnull().sum()

# Fill missing values
reviews.region_1.fillna("Unknown")

# Create a Series that tells you how many reviews exist for each wine region (including unknown ones), ordered from most frequent to least.
reviews_per_country = reviews.value_counts().sort_values(ascending=False)