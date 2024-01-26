### TASK 1

# Load the Titanic dataset from the Seaborn library.

import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.columns


### TASK 2

# Find the number of male and female passengers in the Titanic dataset.

df["sex"].value_counts()

### TASK 3

# Find the number of unique values for each column.

for col in df.columns:
    print(f"{col} : {df[col].nunique()}")

### TASK 4

# Find the number of unique values for the "pclass" column.

df["pclass"].nunique()

### TASK 5

# Find the number of unique values for the "pclass" and "parch" columns.

df[["pclass", "parch"]].nunique()

### TASK 6

# Check the data type of the "embarked" column, change it to "category," and check again.

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")

### TASK 7

# Show the information of passengers with the "embarked" value of "C."

new_df = df[df["embarked"] == "C"]

### TASK 8

# Show the information of passengers with the "embarked" value not equal to "S."

new_df = df[df["embarked"] != "S"]


### TASK 9

# Show the information of female passengers under the age of 30.

new_df = df[(df["age"] < 30) & (df["sex"] == "female")]

### TASK 10

# Show the information of passengers with fare over 500 or age over 70.

new_df = df[(df["fare"] > 500) | (df["age"] > 70)]


### TASK 11

# Find the total number of missing values for each column.

df.isnull().sum()

### TASK 12

# Drop the "who" variable from the dataframe.

df.drop(["who"], axis=1)

### TASK 13

# Fill the missing values in the "deck" variable with the mode.

df["deck"] = df["deck"].fillna(df["deck"].mode()[0])

### TASK 14

# Fill the missing values in the "age" variable with the median.

df["age"] = df["age"].fillna(df["age"].median())

### TASK 15

# Find the sum, count, and mean values of the "survived" variable for the breakdown of "pclass" and "sex" variables.

df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

### TASK 16

# Write a function that assigns 1 to those below 30 and female, and 0 to those equal to or above 30.
# Use the function to create a new variable called "age_flag" in the Titanic dataset. (Use apply and lambda structures)

df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

### TASK 17

# Load the Tips dataset from the Seaborn library.

import seaborn as sns
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("tips")
df.head()

### TASK 18

# Find the minimum, maximum, and median values of the "total_bill" variable by time categories (Dinner, Lunch).

df.groupby(["time"]).agg({"total_bill": ["min", "max", "median"]})

### TASK 19

# Find the minimum, maximum, and median values of the "total_bill" variable by day and time categories.

df.groupby(["day", "time"]).agg({"total_bill": ["min", "max", "median"]})

### TASK 20

# Find the total_bill and tip values for lunchtime and female customers and calculate the sum, min, max, and mean values by day.

df[(df['time'] == 'Lunch') & (df['sex'] == 'Female')].groupby('day').agg({'total_bill': ['sum', 'min', 'max', 'mean'],
                                          'tip': ['sum', 'min', 'max', 'mean']})

### TASK 21

# Find the average of orders with size less than 3 and total_bill greater than 10. (Use loc)

df.loc[(df["size"] < 3) & (df["total_bill"] > 10)]["total_bill"].mean()

### TASK 22

# Create a new variable called "total_bill_tip_sum" that gives the total of total_bill and tip for each customer.

df['total_bill_tip_sum'] = df['total_bill'] + df['tip']
### TASK 23

# Sort the dataframe based on the "total_bill_tip_sum" variable in descending order and assign the top 30 to a new dataframe.

df = df.sort_values("total_bill_tip_sum", ascending=False).head(30)


























