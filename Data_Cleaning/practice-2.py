# Scaling and Normalization

import pandas as pd
from mlxtend.preprocessing import minmax_scaling
from scipy import stats

kickstarter_2018 = "../Datasets/Kickstarter Projects/ks-projects-201801.csv"

ks_data = pd.read_csv(kickstarter_2018)
# print(ks_data.head())

original_data = pd.DataFrame(ks_data.usd_goal_real)

# Scale data
scaled_data = minmax_scaling(original_data, columns=['usd_goal_real'])
print('Original data\nPreview:\n', original_data.head())
print('Minimum value:', original_data.min(),
      '\nMaximum value:', original_data.max())
print('_'*30)

print('\nScaled data\nPreview:\n', scaled_data.head())
print('Minimum value:', scaled_data.min(),
      '\nMaximum value:', scaled_data.max())

# Normalization
positive_pledges = ks_data.pledged.loc[ks_data.pledged>0]
normalized_pledged = pd.Series(stats.boxcox(positive_pledges)[0], name='pledged')
