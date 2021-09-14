# Estimating Home Values from Zillow

## Project Overview

In this project, I will be working with a Zillow dataset to create a model that will predict a property's value. Specifically for this scenario... This machine learning model will predict single-unit property values that are sold during the peak season of real-estate (May-Aug) of 2017.


## Project Description

We are to build a regression model to predict a continuous variable (tax_value) using features that will help us better predict tax value. Since there are heaps of missing data, we'll have to find ways to combat it. We also want to know which counties these properties are located in and calculate tax rate.

## Goals

Deliver a Jupyter notebook going through the steps of the data science pipeline
Create a regression model that performs better than the baseline
Present to audience about my findings

## Deliverables
Finalized Jupyter notebook complete with comments
A README.md with executive summary, contents, data dictionary, conclusion and next steps, and how to recreate this project.
Presentation slide deck

## Project Summary
I built a regression model to predict single unit property values in 3 California counties using a Zillow data frame.

## Data Dictionary 

| Column Name                  | Renamed   | Info                                            |
|------------------------------|-----------|-------------------------------------------------|
| parcelid                     | N/A       | ID of the property (unique)                     |
| bathroomcnt                  | baths     | number of bathrooms                             |
| bedroomcnt                   | beds      | number of bedrooms                              |
| calculatedfinishedsquarefeet | sqft      | number of square feet                           |
| fips                         | N/A       | FIPS code (for county)                          |
| propertylandusetypeid        | N/A       | Type of property                                |
| yearbuilt                    | N/A       | The year the property was built                 |
| taxvaluedollarcnt            | tax_value | Property's tax value in dollars                 |
| transactiondate              | N/A       | Day the property was purchased                  |
| age                          | N/A       | 2017-yearbuilt (to see the age of the property) |
| taxamount                    | tax_amount| amount of tax on property                       |
| tax_rate                     | N/A       | tax_rate on property                            |
| county  # 6037               | float64   | Los Angeles                                     |
| county # 6059                | float64   | Orange                                          |
| county # 6111                | float64   | Ventura                                         |

<br>
<br>

## Hypothesis 

1.) The larger the square footage, the higher the property value

2.) The more bedrooms a house has, the higher its property value will be

3.) The more bathrooms a house has, the higher its property value will be

## Findings and Next Steps 
   - Square footage was the best feature for predicting home value, followed up by bedrooms and bathrooms.
   - Age had little factor, however it was used to better our model, but there may be other features we could look into next time.
   - Location still may have a factor in value, but we would need data that is more normally distributed. Most of the properties were in Los Angeles County. 


Next steps would be:
 - gather more information on location
 - try to clean up/fill in missing values for other location-based columns such as ZIP code, longitude/latitude
 - clean up other columns on home features and see if our model would perform with them (lower RMSE, higher r^2) 


# The Pipeline

## Planning 
Goal: Plan out the project
How does square footage, bathroom count, and bedroom count relate to property value. I believe there will be a 
positive correlation among these variables. 

I also want to look into other features, like age and see if that will also correlate to property value. 
A lot of these features could play hand in hand and help my model make better predictions.

Hypotheses: Square footage, number of bedrooms, number of bathrooms have a positive relationship with value. Age has a negative relationship with value. 


## Acquire 
Goal: Have Zillow dataframe ready to prepare in acquire.py
In this stage, I used a connection URL to access the CodeUp database. Using a SQL query, I brought in the Zillow dataset with only properties set for single use (260, 261, 262, 263, 264, 265, 266, 273, 275, 276, 279, and were sold in between May-August 2017. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline. 

## Prep 
Goal: Have Zillow dataset that is split into train, validate, test, and ready to be analyzed. Assure data types are appropriate and that missing values/duplicates/outliers are addressed. Put this in a prep.py. 
In this stage, I handled outliers by dropping any rows with values that were 3 standard deviations above or below the mean.
All columns had a numeric data type, and renamed them for ease of use.
Duplicates were dropped (in parcelid)
Nulls were also dropped, due to the strong correlation between square feet in respect to property value. 
I split the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test.
I scaled it on a min-max scaler and also returned X_train, X_validate, and X_test scaled. 

## Explore 
Goal: Visualize the data and explore relationships. Use the visuals and statistics tests to help answer your questions. 
I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset. 
Plotted a pairplot to see combinations of variables.
I ran t-tests with the features in respect to tax_value. Also a few to see if the independent variables were related to each other. 
I found that square footage, bedroom count, and bathroom count were all statistically significant. They are not independent to property value. Bedroom count and bathroom count were also dependent on each other. 

## Modeling and Evaluation 
Goal: develop a regression model that performs better than the baseline.

The models worked best with sqft, baths, beds, and age. Polynomial Regression performed the best, so I did a test on it.

| Model                            | RMSE Training | RMSE Validate | R^2   |
|----------------------------------|---------------|---------------|-------|
| Baseline                         | 357,185.61    | 359,454.06    | -3.07 |
| LinearRegression                 | 280,731.20    | 279,672.68    | 0.395 |
| LassoLars                        | 280,731.60    | 279,675.52    | 0.395 |
| TweedieRegressor                 | 280,731.20    | 279,672.68    | 0.395 |
| PolynomialRegression (3 degrees) | 276,310.65    | 274,076.31    | 0.403 |

Test:
 - RMSE of 272,168.27
 - R^2 of 0.403

## Delivery 
I will be giving a presentation over my findings
 - All acquire and prepare .py files are uploaded for easy replication.
 - Presentation slides
 - This README 
 - Final notebook that documents a commented walkthrough of my process

# Conclusion 

To conclude... We took a very large Zillow dataset and condensed it down to 38,622 rows to work with. We dropped rows with outliers of 3 standard deviations above or below the mean for that column.

1.) Square footage was the best feature. As square footage increased, the value increased.

2.) The more bedrooms and bathrooms a house has, the more it was worth. These number of rooms also related to square footage in a positive relationship.

3.) Using all of square footage, number of bedrooms, number of bathrooms, into a model performed better than the baseline.

4.) All three counties have similar tax rates, but LA has the highest.

How to Recreate Project

 - You'll need your own username/pass/host credentials in order to use the get_connection function in my acquire.py to access the Zillow database
 - Have a copy of my acquire, prep, explore .py files. 
 - My final notebook has all of the steps outlined, and it is really easy to adjust parameters.


