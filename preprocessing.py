# %%
import pandas as pd
import numpy as np
# import altair as alt
# from vega_datasets import data
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
from sklearn.neighbors import BallTree
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import RFECV
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from scipy.stats import t
# from urllib.request import urlretrieve
# import requests
import subprocess
import sys
# import os
# from urllib.request import urlretrieve
# from zipfile import ZipFile
# Function to install missing packages
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Read the requirements.txt file and store libraries with their versions
libraries = {}

with open('requirements.txt', 'r') as f:
    for line in f:
        # Each line in the file is in the form 'library==version'
        if '==' in line:
            lib, version = line.strip().split('==')
            libraries[lib] = version

# Standard libraries that do not require installation
standard_libraries = [
    'os', 'zipfile', 'urllib', 'csv', 'math'
]

# Check third-party libraries and install them if missing
for lib, version in libraries.items():
    try:
        __import__(lib)
        print(f"{lib} is already installed.")
    except ImportError:
        print(f"{lib} not found. Installing...")
        install_package(f"{lib}=={version}")

# Confirm standard libraries are available (they should be, as they are part of Python)
for lib in standard_libraries:
    try:
        __import__(lib)
        print(f"Standard library {lib} is available.")
    except ImportError:
        print(f"Standard library {lib} not found, but it should be included in the Python standard library.")

# %%
def import_cdc_survey_2021():
    """
    Reads the CDC survey input file.

    :return: CDC survey dataframe.
    
    """

    url='https://drive.google.com/file/d/17Khw3R3cTaAizs6CzbtCU8ZmxWSRn_mG/view?usp=share-link'
    url='https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])

    #limiting the number of columns we're importing since the 2021 data will only be used for Supervised learning
    col_incl = ["Interview Year", "State Abbr.", "Age Group", "Heart Disease", "Asthma", "Kidney Disease", "Diabetes", "BMI Category"]

    column_types = {'Interview Year': "int16", 'Heart Disease': bool, 'Asthma': bool, 'Kidney Disease': bool, 'Diabetes': bool}
    
    df = pd.read_csv(url, usecols=col_incl, dtype = column_types)

    return df

def import_cdc_survey_2022():
    """
    Reads the CDC survey input file.

    :return: CDC survey dataframe.
    
    """

    url='https://drive.google.com/file/d/1g69nPHfxfNtWnKBoq2SODoPzlCClYdhc/view?usp=share-link'
    url='https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])

    column_types = {'Interview Year': "int16", 'Is Adult Landline': bool, 'Num of Adults Landline': "float32",
                    'Num of Adults Cell': "float32", 'Could Afford Doctor': bool, 'Exercise in Past 30 Days': bool,
                    'Heart Attack': bool, 'Heart Disease': bool, 'Stroke': bool, 'Asthma': bool, 'Asthma in Effect': bool,
                    'Depression': bool, 'Kidney Disease': bool, 'Arthritis': bool, 'Diabetes': bool, 'Weight in Pounds': "float32",
                    'Height in Inches': "float32", 'Cigarettes per Day': "float32", 'Drinks in Last 30 Days': "float32", 
                    'Prediabetes': bool, 'Metropolitan': bool, 'Heavy Alcohol Consumption': bool, 'Hours of Sleeping': "float32", 'Shortness of Breath': bool}
    
    df = pd.read_csv(url, dtype=column_types)

    return df

def import_covid_dataset():
    """
    Reads the COVID input file.

    :return: COVID dataset as a pandas dataframe.
    """

    url='https://drive.google.com/file/d/181jSYrgqjUYc-ba94dl2Lw4Z19vV-ha-/view?usp=share_link'
    url='https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])

    col_incl = ["Start Date", "End Date", "Group", "Year", "State", "Condition Group", "Condition", "Age Group", "COVID-19 Deaths"]

    column_types = {'Year': "float32", 'COVID-19 Deaths': "float32"}
    
    df = pd.read_csv(url, usecols = col_incl, dtype=column_types)

    return df


def import_mortality():
    """
    Reads the mortality input file and parses the contents using CSV format.

    :return: Mortality dataframe.
    """
    # df = pd.read_csv('mortality.csv')
    url='https://drive.google.com/file/d/10HPyYhcNzQm4JQp6EpoWrJWEmEO6uVS7/view?usp=share-link'
    url='https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])
    df = pd.read_csv(url, index_col = [0])
    return df

def import_pop_dataset():
    """
    Reads the US Population input file.

    :return: US Resident Population dataset as a pandas dataframe.
    """

    url='https://drive.google.com/file/d/1DsnRBQ0uczuLMQySqXW91zs8sljoPO2U/view?usp=share-link'
    url='https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2])

    col_incl = ["NAME", "AGE", "POPESTIMATE2021", "POPESTIMATE2022", "SEX", "ORIGIN"]

    column_types = {'AGE': "int32", 'POPESTIMATE2021': "int64", 'POPESTIMATE2022': "int64"}
    
    df = pd.read_csv(url, usecols = col_incl, dtype=column_types)

    return df

# %% [markdown]
# Clean COVID Dataset

# %%
def clean_covid_dataset():
    """
    Cleans and aggregates the COVID-19 dataset.

    Returns:
        df_sup: A cleaned dataframe for use in the supervised portion of the project
        df_unsup: A cleaned dataframe for use in the unsupervised portion of the project
    """
    # Dictionary to map state names to abbreviations
    state_name_to_abbr = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 
        'DISTRICT OF COLUMBIA': 'DC', 'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 
        'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS', 
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 
        'NEW YORK CITY': 'NYC', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 
        'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 
        'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 
        'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 
        'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY', 'PUERTO RICO': 'PR'
    }

    # Pass in imported COVID dataset
    df = import_covid_dataset()
    
    # Filter out rows where 'state' is missing or irrelevant
    df_cleaned = df[df['State'].notna()]
    df_cleaned['State'] = df_cleaned['State'].str.upper()
    df_cleaned = df_cleaned[df_cleaned['State'] != 'UNITED STATES']

    # Map state names to abbreviations
    df_cleaned['state_abbr'] = df_cleaned['State'].map(state_name_to_abbr)

    # Handle data types and missing data
    df_cleaned['Year'] = df_cleaned['Year'].fillna(0).astype(int)
    df_cleaned['COVID-19 Deaths'] = df_cleaned['COVID-19 Deaths'].fillna(0).astype(int)

    # Drop rows where 'age_group' is 'Not stated' or 'All Ages'
    df_cleaned = df_cleaned[~df_cleaned['Age Group'].isin(['Not stated', 'All Ages'])]

    # Drop rows where the condition is 'COVID-19' (as we focus on pre-existing conditions)
    df_cleaned = df_cleaned[df_cleaned['Condition'] != 'COVID-19']

    #Here we will focus on cleaning specifically for the Supervised portion of the project
    # We only want to focus on annual data as that is the level of the BRFSS survey data that we have
    df_sup = df_cleaned[(df_cleaned['Group'] == "By Year")]

    # We are only looking at BRFSS data from 2021 and 2022, so we will filter this to match
    df_sup = df_sup[(df_sup['Year'] == 2021) | (df_sup['Year'] == 2022)] 

    # Map conditions for consistency with cdc dataset
    df_sup['Condition_new'] = df_sup['Condition'].map({
        'Hypertensive diseases': 'Heart Disease', 'Ischemic heart disease': 'Heart Disease',
        'Cardiac arrest': 'Heart Disease', 'Cardiac arrhythmia': 'Heart Disease',
        'Heart failure': 'Heart Disease', 'Chronic lower respiratory diseases': 'Asthma',
        'Renal failure': 'Kidney Disease', 'Diabetes': 'Diabetes', 'Obesity': 'Obesity'
    })
    df_sup = df_sup[df_sup['Condition_new'].notna()]

    # Map age groups for consistency with cdc dataset
    rlist = ['65-74', '75-84', '85+']
    df_sup['Age Group'] = df_sup['Age Group'].mask(df_sup['Age Group'].isin(rlist), '65+')

    # Re encode age groups so we can use it as a feature
    df_sup['Age Group'] = df_sup['Age Group'].map({
        '0-24': 1,
        '25-34': 2,
        '35-44': 3,
        '45-54': 4,
        '55-64': 5,
        '65+': 6
    })

    # New York City is a separate count in this dataset but not in our other datasets, so we will add those numbers to New York State's count
    df_sup['state_abbr'] = df_sup['state_abbr'].replace('NYC', 'NY')
    df_sup['State'] = df_sup['State'].replace('NEW YORK CITY', 'NEW YORK')

    # Drop unnecessary columns, reaggregate the new age group and rename columns where needed
    df_sup = df_sup.drop(columns=['Start Date', 'End Date', 'Condition', 'Condition Group'])
    df_sup = df_sup.groupby(['Year', 'Age Group', 'state_abbr', 'Condition_new']).agg({"COVID-19 Deaths": "sum"}).reset_index()
    df_sup.rename(columns={'state_abbr': 'State', 'Condition_new': 'Condition'}, inplace=True)

    df_sup2 = df_sup.pivot(index = ['Year', 'Age Group', 'State'], columns = 'Condition', values = 'COVID-19 Deaths').reset_index()
    df_sup2 = df_sup2.rename(columns={'Asthma': 'Asthma_covdeath', 'Diabetes': 'Diabetes_covdeath', 'Heart Disease': 'Heart Disease_covdeath',
                      'Kidney Disease': 'Kidney Disease_covdeath', 'Obesity': 'Obesity_covdeath'})

    #Here we will focus on cleaning specifically for the Unsupervised portion of the project
    df_unsup = df_cleaned.copy()

    # Ensure 'start_date' and 'end_date' are in datetime format
    df_unsup['Start Date'] = pd.to_datetime(df_unsup['Start Date'])
    df_unsup['End Date'] = pd.to_datetime(df_unsup['End Date'])

    # Calculate the time difference in days, months, and years
    days_difference = (df_unsup['End Date'] - df_unsup['Start Date']).dt.days
    # months_difference = days_difference / 30.44
    years_difference = days_difference / 365.25  # Accounts for leap years

    # Create new columns for daily, monthly, and yearly averages of covid_19_deaths
    # df_unsup['daily_avg'] = df_unsup['COVID-19 Deaths'] / days_difference
    # df_unsup['monthly_avg'] = df_unsup['COVID-19 Deaths'] / months_difference
    df_unsup['yearly_avg'] = df_unsup['COVID-19 Deaths'] / years_difference

    # # Drop unnecessary columns and reset the index
    # df_unsup = df_unsup.drop(columns=['Start Date', 'End Date', 'Year', 'COVID-19 Deaths', 'Group'])
    
    df_unsup = df_unsup.groupby('state_abbr').agg({'yearly_avg': 'mean'}).reset_index()

    return df_sup2, df_unsup

# %% [markdown]
# Clean/Parse Mortality Dataset

# %%
def clean_mortality_dataset():
    """
    Cleans and aggregates the cardiovascular disease mortality dataset.

    Returns:
        df_cleaned: A cleaned dataframe for use in the unsupervised portion of the project
    """

    # Pass in imported mortality dataset
    df = import_mortality()

    # Impute at county level
    df_mortality_county = df[df['GeographicLevel'] == 'County'].copy()

    # Convert lat and lon coordinates into radians
    for col in df_mortality_county[["Y_lat", "X_lon"]]:
        rad = np.deg2rad(df_mortality_county[col].values)
        df_mortality_county[f'{col}_rad'] = rad

    # Use Ball Tree to reduce search time for missing data. Ball tree only works on radian data
    df_mortality_county_without_nan = df_mortality_county[df_mortality_county['Data_Value'].notna()].copy()
    ball = BallTree(df_mortality_county_without_nan[['Y_lat_rad', 'X_lon_rad']].values, metric='haversine')

    # Efficiently find four nearest neighbor counties and avg the data from them
    def impute_from_neigbours(row):
        if np.isnan(row['Data_Value']):
            _, indices = ball.query([row[['Y_lat_rad', 'X_lon_rad']].values], k=4)
            row['Data_Value'] = df_mortality_county_without_nan.iloc[indices[0]]['Data_Value'].mean()
        return row
    
    # Impute at county level
    df_mortality_county_imputed = df_mortality_county.apply(impute_from_neigbours, axis=1)

    # Aggregate data for unsupervised portion of project
    df_cleaned = df_mortality_county_imputed.groupby('LocationAbbr').agg({'Data_Value': 'mean'}).reset_index()

    return df_cleaned

# %% [markdown]
# Merge BRFSS surveys and clean/calculate

# %%
def clean_cdc_unsup():
    """
    Cleans and aggregates the CDC BRFSS survey dataset for the unsupervised portion of the project.

    Returns:
        df_cleaned: A cleaned dataframe for use in the unsupervised portion of the project
    """
    # Import the dataset
    df = import_cdc_survey_2022()

    # Map General Health
    df['General Health'] = df['General Health'].map({
        'Excellent': 5, 'Very good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1
    })

    # Map Smoking 
    df['Smoking'] = df['Smoking'].map({
        'never': 0,      # Non-smoker
        'some_days': 1,  # Occasional smoker
        'every_day': 2,  # Daily smoker
    })

    # Map True to 1 and False to 0 for 'Exercise in Past 30 Days'
    df['Exercise in Past 30 Days'] = df['Exercise in Past 30 Days'].map({
        True: 1,
        False: 0
    })

    # Map categorical 'Years Since Last Checkup' values to numeric
    df['Years Since Last Checkup'] = df['Years Since Last Checkup'].map({
        'within_past_year': 1,
        'within_past_two_years': 2,
        'within_past_five_years': 3,
        'five_or_more_years': 5
    })

    # Map categorical BMI values to numeric
    df['BMI Category'] = df['BMI Category'].map({
        'underweight': 0,
        'normal_weight': 1,
        'over_weight': 2,
        'obese': 3
    })

    # Replace NaN values in 'Cigarettes per Day' and 'Drinks in Last 30 Days' with 0
    # This is so we can get a average
    df['Cigarettes per Day'] = df['Cigarettes per Day'].fillna(0)
    df['Drinks in Last 30 Days'] = df['Drinks in Last 30 Days'].fillna(0)

    # Aggregating df_cdc_survey
    df_cleaned = df.groupby('State Abbr.').agg({
        'General Health': 'mean', 
        'Exercise in Past 30 Days': 'mean', 
        'Smoking': 'mean', 
        'Shortness of Breath': 'sum', 
        'Hours of Sleeping': 'mean',
        'BMI Category': 'mean',
        'Years Since Last Checkup': 'mean',
        'Cigarettes per Day': 'mean',
        'Drinks in Last 30 Days': 'mean'
    }).reset_index()


    return df_cleaned

# %%
def clean_cdc_sup():
    """
    Cleans and aggregates the CDC BRFSS survey dataset for the supervised portion of the project. Also merges the 2021 and 2022 data.

    Returns:
        df_agg: A cleaned dataframe for use in the supervised portion of the project
    """

    # Import data
    df21 = import_cdc_survey_2021()
    df22 = import_cdc_survey_2022()

    # Confirm that each dataset only contains the year it's supposed to
    df21 = df21[df21['Interview Year'] == 2021]
    df22 = df22[df22['Interview Year'] == 2022]

    # Filter out unnecessary columns; this was done upon import for the 2021 dataset
    df22 = df22[['State Abbr.', 'Interview Year', 'Heart Disease', 'Asthma', 'Kidney Disease', 'Diabetes', 'Age Group', 'BMI Category']]

    # Merge the two datasets
    df = pd.concat([df21, df22])

    # Get total count of responses per year, age group, and state so that we can create a percentage
    temp = df.groupby(['Interview Year', 'Age Group', 'State Abbr.']).count()['Diabetes']
    temp = temp.rename('Response Count')
    dfmerge = pd.merge(temp, df, how='inner', on=['Interview Year', 'Age Group', 'State Abbr.'])

    # Simplify BMI to classify obesity or none
    dfmerge.loc[dfmerge['BMI Category'] == 'obese', 'Obesity'] = 1
    dfmerge.loc[dfmerge['BMI Category'] != 'obese', 'Obesity'] = 0
    dfmerge['Obesity'] = dfmerge['Obesity'].astype(int)

    # Rename columns where needed
    dfmerge = dfmerge.rename(columns={'State Abbr.': 'State', 'Interview Year': 'Year'})

    # Replace all boolean values with 0/1 values to make summarizing easier
    dfmerge = dfmerge.replace({True: 1, False: 0})

    # Remove unnecessary columns
    dfmerge = dfmerge.drop(columns=['BMI Category'])

    # Map age groups for consistency with COVID dataset
    dfmerge['Age Group'] = dfmerge['Age Group'].map({
        'between_18_and_24': 1, 'between_25_and_34': 2, 'between_35_and_44': 3,
        'between_45_and_54': 4, 'between_55_and_64': 5, 'older_than_65': 6
    })

    # Aggregate data by state, interview year, and age group
    df_agg = dfmerge.groupby(['State', 'Year', 'Age Group']).agg({"Heart Disease": "sum", "Asthma": "sum", "Kidney Disease": "sum",
                                                                  "Diabetes": "sum", "Obesity": "sum", "Response Count": "mean"}).reset_index()
    
    # Convert counts to percentages so that we can generalize to the population
    df_agg['Heart Disease'] = df_agg['Heart Disease'] / df_agg['Response Count']
    df_agg['Asthma'] = df_agg['Asthma'] / df_agg['Response Count']
    df_agg['Kidney Disease'] = df_agg['Kidney Disease'] / df_agg['Response Count']
    df_agg['Diabetes'] = df_agg['Diabetes'] / df_agg['Response Count']
    df_agg['Obesity'] = df_agg['Obesity'] / df_agg['Response Count']

    # Drop unnecessary column
    df_agg = df_agg.drop(columns=['Response Count'])

    return df_agg

# %% [markdown]
# Clean Resident Population dataset

# %%
def clean_census():
    """
    Cleans and aggregates the US Census dataset for the supervised portion of the project.

    Returns:
        df_agg: A cleaned dataframe for use in the supervised portion of the project
    """

    # Import the dataset
    df = import_pop_dataset()

    # Filter dataset to alleviate duplicate counting
    df = df[(df['SEX'] == 0) & (df['ORIGIN'] == 0)]

    # Dictionary to map state names to abbreviations
    state_name_to_abbr = {
        'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 
        'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 
        'DISTRICT OF COLUMBIA': 'DC', 'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 
        'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS', 
        'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD', 
        'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 
        'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 
        'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 
        'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 
        'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 
        'SOUTH CAROLINA': 'SC', 'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 
        'UTAH': 'UT', 'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 
        'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY', 'PUERTO RICO': 'PR'
    }

    # Convert state names to abbreviations for consistency
    df['NAME'] = df['NAME'].str.upper()
    df['State'] = df['NAME'].map(state_name_to_abbr)

    # Check that there are no blank values in the age column
    df['AGE'] = df['AGE'].astype(int)
    df = df[df['AGE'].notna()]

    # Map the ages to the same age range we have in the other datasets
    criteria = [df['AGE'].between(0,24), df['AGE'].between(25,34), df['AGE'].between(35,44), df['AGE'].between(45,54), df['AGE'].between(55,64), df['AGE'].between(65,1000)]
    values = [1, 2, 3, 4, 5, 6]
    df['Age Group'] = np.select(criteria, values, 0)

    # Get updated counts for age ranges
    df_agg = df.groupby(['Age Group', 'State']).agg({'POPESTIMATE2021': 'sum', 'POPESTIMATE2022': 'sum'}).reset_index()

    # Rename columns for the pivot table
    df_agg = df_agg.rename(columns={'POPESTIMATE2021': 2021, 'POPESTIMATE2022': 2022})

    # Unpivot so that we have a year column
    df_melt = pd.melt(df_agg, id_vars = ['Age Group', 'State'], value_vars = [2021, 2022], var_name = "Year", value_name = "Population")

    # Ensure correct datatype
    df_melt['Population'] = df_melt['Population'].astype(int)
    
    return df_melt

# %% [markdown]
# Merge datasets for Supervised Learning project

# %%
def get_sup_df():
    """
    Merges the population, both CDC, and COVID datasets and performs feature engineering to extract useful features for the supervised learning portion of the project.

    Returns:
        df: A cleaned, merged dataframe for use in the supervised learning portion of the project
    """

    # Import all necessary datasets
    pop = clean_census()
    covid, _ = clean_covid_dataset()
    cdc = clean_cdc_sup()

    # Merge datasets
    df_cov_cdc = pd.merge(cdc, covid, how='inner', on=['Year', 'Age Group', 'State']).reset_index(drop=True)
    df_merged = pd.merge(df_cov_cdc, pop, how='inner', on=['Year', 'Age Group', 'State']).reset_index(drop=True)

    # Final feature engineering calculations
    df_merged['All Conditions_covdeath'] = df_merged[['Obesity_covdeath', 'Kidney Disease_covdeath', 'Asthma_covdeath', 'Diabetes_covdeath', 'Heart Disease_covdeath']].sum(axis=1)
    df_merged['Rate of COVID Deaths Due to Conditions'] = df_merged['All Conditions_covdeath'] / df_merged['Population']

    # Remove unnecessary columns
    df = df_merged[['State', 'Year', 'Age Group', 'Heart Disease', 'Asthma', 'Kidney Disease', 'Diabetes', 'Obesity', 'Population', 'Rate of COVID Deaths Due to Conditions']]

    return df

# %% [markdown]
# Merge datasets for Unsupervised Learning project

# %%
def get_unsup_df():
    """
    Merges the mortality, CDC, and COVID datasets to create a dataset for unsupervised analysis.

    Returns:
        df: A cleaned, merged dataframe for use in the unsupervised
    """

    # Get datasets to merge
    _, covid = clean_covid_dataset()
    mortality = clean_mortality_dataset()
    cdc = clean_cdc_unsup()

    # Merge cdc with mortality
    df = cdc.merge(mortality, left_on = 'State Abbr.', right_on = 'LocationAbbr', how = 'inner')

    # Merge the result with covid
    df = df.merge(covid, left_on = 'State Abbr.', right_on = 'state_abbr', how = 'inner')

    # Drop duplicate columns for state
    df.drop(columns = ['LocationAbbr', 'state_abbr'], inplace = True)

    return df

# %%

