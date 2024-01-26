#############################################
# Rule-Based Classification for Calculating Potential Customer Revenue
#############################################

#############################################
# Business Problem
#############################################
# A gaming company wants to create level-based personas for its customers using some features and predict how much potential customers, based on these new personas, could contribute to the company on average.

# For example: Determining how much, on average, a 25-year-old male user from Turkey who uses IOS can contribute.

#############################################
# Data Set Story
#############################################
# The "persona.csv" dataset contains prices of products sold by an international gaming company and some demographic information of users who purchased these products. The dataset is composed of records created in each sales transaction. This means that the table is not unique. In other words, a user with certain demographic characteristics may have made more than one purchase.

# Price: Amount spent by the customer
# Source: Type of device the customer is connected to
# Sex: Gender of the customer
# Country: Country of the customer
# Age: Age of the customer

################# Before Application #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# After Application #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJECT TASKS
#############################################

# TASK 1: Answer the following questions.
#############################################

# Question 1: Read the "persona.csv" file and display general information about the dataset.

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("persona.csv")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

# Question 2: How many unique SOURCE values are there? What are their frequencies?

df["SOURCE"].nunique()
df["SOURCE"].value_counts().index[0]
print(f'There are {df["SOURCE"].nunique()} unique values in the "SOURCE" column.')

# Question 3: How many unique PRICE values are there?

df["PRICE"].nunique()

# Question 4: How many sales have occurred for each PRICE?

df["PRICE"].value_counts()

# Question 5: How many sales have occurred for each country?

df["COUNTRY"].value_counts()

# Question 6: How much revenue has been generated for each country?

df.groupby("COUNTRY").agg({"PRICE": "sum"})

# Question 7: What are the sales counts for each SOURCE type?

df.groupby("SOURCE").agg({"PRICE": "count"})

# Question 8: What are the PRICE averages for each country?

df.groupby("COUNTRY").agg({"PRICE": "mean"})

# Question 9: What are the PRICE averages for each SOURCE?

df.groupby("SOURCE").agg({"PRICE" : "mean"})

# Question 10: What are the PRICE averages for each COUNTRY-SOURCE combination?

df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE" : "mean"})

#############################################
# TASK 2: What are the average earnings by COUNTRY, SOURCE, SEX, and AGE?
#############################################

df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE" : "mean"})

#############################################
# TASK 3: Sort the output by PRICE.
#############################################
# Apply the sort_values method to the PRICE in descending order to better visualize the output.
# Save the output as "agg_df."

agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE" : "mean"}).sort_values("PRICE", ascending=False)

#############################################
# TASK 4: Convert the names in the index to variable names.
#############################################
# All variables in the output of the third question except PRICE are index names.
# Convert these names to variable names.
# Hint: reset_index()
# agg_df.reset_index(inplace=True)

agg_df.reset_index(inplace=True)
agg_df

#############################################
# TASK 5: Convert the AGE variable to a categorical variable and add it to agg_df.
#############################################
# Convert the numerical variable Age to a categorical variable.
# Create intervals in a way you find convincing.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

# 0-18: CHILD
# 19-40: YOUNG
# 41-55: MID-AGE
# 56-70: UPPER-MID

agg_df["AGE_CLASS"] = pd.cut(agg_df["AGE"], [0, 18, 40, 55, 70], labels=["CHILD", "YOUNG", "MID-AGE", "UPPER-MID"])

#############################################
# TASK 6: Define new level-based customers and add them as a variable to the dataset.
#############################################
# Define a variable named "customers_level_based" and add it to the dataset.
# Attention!
# After creating customer_level_based values with list comp, these values need to be unique.
# This can be achieved by grouping those based on COUNTRY, SOURCE, SEX, AGE_CLASS, and taking the mean of the PRICE.

agg_df["CUSTOMERS_LEVEL_BASED"] = [('_'.join(map(str, value))).upper() for value in agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CLASS"]].values]
agg_df["CUSTOMERS_LEVEL_BASED"] = [('_'.join(value)).upper() for value in agg_df[["COUNTRY", "SOURCE", "SEX", "AGE_CLASS"]].values]

agg_df.groupby("CUSTOMERS_LEVEL_BASED").agg({"PRICE" : "mean"})

#############################################
# TASK 7: Divide new customers (USA_ANDROID_MALE_0_18) into segments.
#############################################
# Divide into 4 segments based on PRICE,
# add segments to agg_df with the name "SEGMENT,"
# and describe the segments.

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D", "C", "B", "A"])

agg_df.groupby("SEGMENT").agg({"PRICE" :["mean", "max", "sum"]})

agg_df["CUSTOMERS_LEVEL_BASED"].value_counts()

#############################################
# TASK 8: Classify and estimate the income of new customers.
#############################################
def classify_and_estimate_income(new_client):
    """Classifies the entered customer and estimates their income.

    Parameters:
    - new_client: str, CUSTOMERS_LEVEL_BASED value of the new customer

    Returns:
    - segment: str, customer segment
    - estimated_income: float, estimated income of the customer"""
    segment = agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_client]["SEGMENT"].values[0]    # Filter by CUSTOMERS_LEVEL_BASED and get the segment of the first customer
    estimated_income = agg_df[agg_df["CUSTOMERS_LEVEL_BASED"] == new_client]["PRICE"].mean()  # Average income of the filtered customers

    return segment, estimated_income

# For a 33-year-old TURKISH female using ANDROID, which segment does she belong to, and what is the expected income?

new_client_1 = 'TUR_ANDROID_FEMALE_YOUNG'  # When a new customer is defined, print its segment and estimated income.
segment_1, estimated_income_1 = classify_and_estimate_income(new_client_1)

print(f"{new_client_1}")
print(f"SEGMENT: {segment_1}")
print(f"ESTIMATED INCOME: {estimated_income_1}")

# For a 35-year-old FRENCH female using IOS, which segment does she belong to, and what is the expected income?

new_client_2 = 'FRA_IOS_FEMALE_YOUNG'
segment_2, estimated_income_2 = classify_and_estimate_income(new_client_2)

print(f"{new_client_2}")
print(f"SEGMENT: {segment_2}")
print(f"ESTIMATED INCOME: {estimated_income_2}")