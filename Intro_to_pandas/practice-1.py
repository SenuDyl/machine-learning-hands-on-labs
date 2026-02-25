# Creating, reading and writing

import pandas as pd

# Creating a table
fruit_sale = pd.DataFrame({
    "Apples": [35, 40],
    "Bananas": [60, 50]
},
index = ["2017 Sales", "2018 Sales"]
)

# Creating a series
marks = pd.Series(
    ["78", "90", "86"],
    index = ["Bella", "Mike", "Dustin"],
    name = "Marks"
)

marks.to_csv("../out/marks.csv")