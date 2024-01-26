### TASK 1

# Use List Comprehension to convert the names of numeric variables in the car_crashes dataset to uppercase and add "NUM" to the beginning.
# Note: The names of non-numeric variables should also be converted to uppercase.
# Use a single list comprehension structure.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

["NUM_" + col  for col in df.columns if df[col].dtype != "O"]

["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

[f"NUM_{col.upper()}" if pd.api.types.is_numeric_dtype(df[col]) else col.upper() for col in df.columns]



### TASK 2

# Use List Comprehension to append "FLAG" to the names of variables in the car_crashes dataset that do not contain "no".

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]



### TASK 3

# Use List Comprehension to select the names of variables from the given list of variable names that are NOT in og_list,
# and create a new data frame with these names.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

og_list = ['abbrev', 'no_previous']
new_cols = [col for col in df.columns if col not in og_list]
new_df = df[new_cols]
new_df = df[new_cols]




















