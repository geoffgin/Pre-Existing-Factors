## Pre-Exisitng Conditions & Mortality Analysis

This project analyzes the correlation between pre-existing conditions (such as heart disease, asthma, kidney disease, etc.) and COVID-19 mortality rates across different states and age groups in the United States. The analysis uses CDC and COVID datasets along with mortality data to create both supervised and unsupervised learning models.

### Project Structure
#### 1. Preprocessing:
The preprocessing stage involves data cleaning, aggregation, and feature engineering to prepare the data for machine learning models. Datasets are merged, cleaned, and organized into supervised and unsupervised datasets. This involves handling missing values, normalizing data, and creating new features that represent percentages of pre-existing conditions by age group, state, and year.
* Imports:
  * Behavior Risk Factor Survey (BFRSS)
  * Cardiovascular Mortality Dataset
  * COVID-19 Mortality Dataset
  * Population Dataset
*Cleaning Functions:
  * `import_cdc_survey_2021()`, `import_cdc_survey_2022()`, `import_covid_dataset()`, `import_mortality()`, and `import_pop_dataset()` are used to import and clean the data.
  * `clean_covid_dataset()`, `clean_mortality_dataset()`, `clean_census()` are used to clean individual datasets.
*Merging:
  * Supervised learning dataset: Combines `Population`, `COVID_19`, and `BFRSS` datasets for regression models.
  * Unsupervised learning dataset: Aggregates `Cardiovascular Mortality`, pre-existing conditions from the `BFRSS`, and `COVID-19` rates for clustering and PCA analysis.

#### 2. Supervised Learning (`Supervised.ipynb`):
This notebook focuses on predicting COVID-19 mortality rates using pre-existing conditions. Several regression models are explored, including:
* Models:
  * Linear Regression
  * Ridge Regression
  * K-Nearest Neighbors (KNN)
  * Random Forest
* Key Steps:
  * Hyperparameter tuning using RandomizedSearchCV.
  * Feature scaling and polynomial feature transformations.
  * Model evaluation using R² and Mean Squared Error (MSE).
  * Sensitivity analysis and feature importance for KNN and Random Forest models.
* Outcome:
  * Best-performing model: KNN with `n_neighbors=10`, achieving an R² score of 0.8490 and MSE of 0.00000047.

#### 3. Unsupervised Learning (Unsupervised.ipynb)
This notebook applies clustering and dimensionality reduction techniques to analyze the relationships between pre-existing conditions and COVID-19 mortality across different states.
* Dimensionality Reduction:
  * Principal Component Analysis (PCA) reduces the dataset to key components that explain most variance.
* Clustering Techniques:
  * K-Means clustering (k=4) identifies patterns and groups states with similar mortality trends.
  * DBSCAN and Agglomerative clustering provide insights into state-level variations and noise points.
* Model Evaluation:
  * Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index used for cluster quality evaluation.
* Visualizations:
  * PCA and clustering visualizations show state relationships and cluster assignments.

 #### 4. Key Insights:
* States with high pre-existing conditions (like heart disease) tend to have higher mortality rates.
* Dimensionality reduction shows that factors such as general health and exercise have strong correlations with pre-existing conditions and mortality.
---------------------------
### Requirements
To run this project, you'll need to install the required packages. Run the following command to install the dependencies:

`pip install -r requirements.txt`

### Usage
1. Preprocessing: Run the preprocessing script to load and clean the datasets.
2. Supervised Learning: Execute Supervised.ipynb for predictive modeling and regression analysis.
3. Unsupervised Learning: Run Unsupervised.ipynb to explore patterns in the data using PCA and clustering.

### Visualizations
The notebooks include visualizations of heatmaps, pairplots, PCA scatterplots, and clustering results to provide insights into the relationships between pre-existing conditions and mortality outcomes.

--------------------------------------
### Author's

Natalie LaRowe (nlarowe@umich.edu), Geoffrey Gin (ggin@umich.edu), Denesh Chandrahasan (denesh@umich.edu), 2024
