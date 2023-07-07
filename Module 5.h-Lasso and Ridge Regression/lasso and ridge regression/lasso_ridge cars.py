## Problem Statement
'''
The client/business deals with used cars sales.

The customers in this sector give strong preference to less-aged cars 
and popular brands with good resale value. 
This puts a very strong challenge as they only have a very limited range of 
vehicle options to showcase.

#### No Pre-Set Standards

`How does one determine the value of a used car?`

The Market scenario is filled with a lot of malpractices. There is no defined standards exist to determine the appropriate price for the cars, the values are determined by arbitrary methods.

The unorganized and unstructured methods are disadvantageous to the both the parties trying to strike a deal. The look and feel can be altered in used cars, but the performance cannot be altered beyond a point.


#### Revolutionizing the Used Car Industry Through Machine Learning

**Linear regression**
Linear regression is a ML model that estimates the relationship between independent variables and a dependent variable using a linear equation (straight line equation) in a multidimensional space.

**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance


**Objective(s):** Maximize the profits

**Constraints:** Maximize the customer satisfaction

**Success Criteria**

- **Business Success Criteria**: Improve the profits from anywhere between 10% to 20%

- **ML Success Criteria**: RMSE should be less than 0.15

- **Economic Success Criteria**: Second/Used cars sales delars would see an 
increase in revenues by atleast 20%
'''

# Load the Data and perform EDA and Data Preprocessing

# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sidetable

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from feature_engine.outliers import Winsorizer

# from sklearn.linear_model import LinearRegression
from statsmodels.tools.tools import add_constant

# from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# from sklearn.model_selection import train_test_split
# import statsmodels.formula.api as smf
import statsmodels.api as sm

import joblib
import pickle

from sklearn.model_selection import GridSearchCV

from sqlalchemy import create_engine


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root",# user
                               pw = "Seemscrazy1994#", # passwrd
                               db = "cars_db")) #database


# Load the offline data into Database to simulate client conditions
cars = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5.h-Lasso and Ridge Regression\lasso and ridge regression\CarswithEnginetype.csv")
cars.to_sql('cars', con = engine, if_exists = 'replace', chunksize = 1000, index= False)


#### Read the Table (data) from MySQL database

sql = 'SELECT * FROM cars'
# sql2="show tables"
# tables = pd.read_sql_query(sql2, engine)

dataset = pd.read_sql_query(sql, engine)


# dataset = pd.read_csv(r"C:/Data/CarswithEnginetype.csv")
# Priliminary Analysis
#### Descriptive Statistics and Data Distribution
dataset.describe()

# Missing values check
dataset.isnull().any()
dataset.info()


# Seperating input and output variables 
X = pd.DataFrame(dataset.iloc[:, 1:6])
y = pd.DataFrame(dataset.iloc[:, 0])

# Checking for unique values
X["Enginetype"].unique()

X["Enginetype"].value_counts()

# Build a frequency table using sidetable library
X.stb.freq(["Enginetype"])

# Segregating Non-Numeric features
categorical_features = X.select_dtypes(include = ['object']).columns
print(categorical_features)


# Segregating Numeric features
numeric_features = X.select_dtypes(exclude = ['object']).columns
print(numeric_features)


## Missing values Analysis
# Define pipeline for missing data if any

num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'mean'))])

preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, numeric_features)])

# Fit the imputation pipeline to input features
imputation = preprocessor.fit(X)

# Save the pipeline
joblib.dump(imputation, 'meanimpute')

# Transformed data
cleandata = pd.DataFrame(imputation.transform(X), columns = numeric_features)
cleandata


## Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

X.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Winsorization for outlier treatment
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.columns))

winsor

clean = winsor.fit(cleandata)

# Save winsorizer model
joblib.dump(clean, 'winsor')

cleandata1 = pd.DataFrame(clean.transform(cleandata), columns = numeric_features)

# Boxplot
cleandata1.plot(kind = 'box', subplots = True, sharey = False, figsize = (25, 18)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


# Scaling
## Scaling with MinMaxScaler
scale_pipeline = Pipeline([('scale', MinMaxScaler())])

scale_columntransfer = ColumnTransformer([('scale', scale_pipeline, numeric_features)]) # Skips the transformations for remaining columns

scale = scale_columntransfer.fit(cleandata1)

# Save Minmax scaler pipeline model
joblib.dump(scale, 'minmax')

scaled_data = pd.DataFrame(scale.transform(cleandata1), columns = numeric_features)
scaled_data.describe()



## Encoding
# Categorical features
encoding_pipeline = Pipeline([('onehot', OneHotEncoder())])

preprocess_pipeline = ColumnTransformer([('categorical', encoding_pipeline, categorical_features)])

clean =  preprocess_pipeline.fit(X)   # Works with categorical features only

# Save the encoding model
joblib.dump(clean, 'encoding')

encode_data = pd.DataFrame(clean.transform(X).todense())


# To get feature names for Categorical columns after Onehotencoding 
encode_data.columns = clean.get_feature_names_out(input_features = X.columns)
encode_data.info()


clean_data = pd.concat([scaled_data, encode_data], axis = 1)  # concatenated data will have new sequential index
clean_data.info()
clean_data.describe().T  # Transposed view


# Library to call OLS model
# import statsmodels.api as sm

# Build a vanilla model on full dataset

# By default, statsmodels fits a line passing through the origin, i.e. it 
# doesn't fit an intercept. Hence, you need to use the command 'add_constant' 
# so that it also fits an intercept

P = add_constant(clean_data)
basemodel = sm.OLS(y, P).fit()
basemodel.summary()

# p-values of coefficients found to be insignificant due to colinearity

# Identify the variale with highest colinearity using Variance Inflation factor (VIF)
# Variance Inflation Factor (VIF)
# Assumption: VIF > 10 = colinearity
# VIF on clean Data
vif = pd.Series([variance_inflation_factor(P.values, i) for i in range(P.shape[1])], index = P.columns)
vif
# inf = infinity

# WT shows highest VIF 
# Drop colinearity variable - WT

# Rebuild model with remaining columns
clean_data1 = P.drop('WT', axis = 1)
clean_data1.columns

# Build a model on  dataset
basemode2 = sm.OLS(y, clean_data1).fit()
basemode2.summary()


# Tune the model by verifying for influential observations
sm.graphics.influence_plot(basemode2)

clean_data1_new = clean_data1.drop(clean_data1.index[[76, 78]])
y_new = y.drop(y.index[[76, 78]])

# Build model on dataset
basemode3 = sm.OLS(y_new, clean_data1_new).fit()
basemode3.summary()



# Regularization Techniques: LASSO, RIDGE and ElasticNet Regression
################
### LASSO MODEL ###
#pip install statsmodel
#pip install patsy
from sklearn.linear_model import Lasso
#help(Lasso)

lasso = Lasso(alpha = 0.13, fit_intercept = True)
#lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(clean_data, y)

# Coefficient values for all independent variables#
lasso.intercept_
lasso.coef_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(clean_data.columns))

# Create a function called lasso,
pred_lasso = lasso.predict(clean_data)

# Adjusted r-square
s1 = lasso.score(clean_data, y.MPG)
s1

# RMSE
np.sqrt(np.mean((pred_lasso - np.array(y['MPG']))**2))



### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)

rm = Ridge(alpha = 0.13, fit_intercept = True)

rm.fit(clean_data, y)

# Coefficients values for all the independent vairbales
rm.intercept_
rm.coef_
result = rm.coef_.flatten()
result

plt.bar(height = pd.Series(result), x = pd.Series(clean_data.columns))

rm.alpha

pred_rm = rm.predict(clean_data)

# Adjusted r-square
s2 = rm.score(clean_data, y.MPG)
s2

# RMSE
np.sqrt(np.mean((pred_rm - np.array(y['MPG']))**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)

enet = ElasticNet(alpha = 0.13, fit_intercept = True)

enet.fit(clean_data, y.MPG) 

# Coefficients values for all the independent vairbales
enet.intercept_
enet.coef_


plt.bar(height = pd.Series(enet.coef_), x = pd.Series(clean_data.columns))

pred_enet = enet.predict(clean_data)

# Adjusted r-square
s3 = enet.score(clean_data, y.MPG)
s3

# RMSE
np.sqrt(np.mean((pred_enet - np.array(y.MPG))**2))


####################
# Lasso Regression
# from sklearn.model_selection import GridSearchCV

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.13, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(clean_data, y.MPG)

lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(clean_data)

# Adjusted r-square#
s4 = lasso_reg.score(clean_data, y.MPG)
s4

# RMSE
np.sqrt(np.mean((lasso_pred - np.array(y.MPG))**2))


# Ridge Regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import Ridge

ridge = Ridge()

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.13, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(clean_data, y.MPG)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(clean_data)

# Adjusted r-square#
s5 = ridge_reg.score(clean_data, y.MPG)
s5
    # RMSE
np.sqrt(np.mean((ridge_pred - np.array(y.MPG))**2))



# ElasticNet Regression
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import ElasticNet

enet = ElasticNet()

# parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 0.13, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'r2', cv = 5)

enet_reg.fit(clean_data, y.MPG)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(clean_data)

# Adjusted r-square
s6 = enet_reg.score(clean_data, y.MPG)
s6

# RMSE
np.sqrt(np.mean((enet_pred - np.array(y.MPG))**2))

scores_all = pd.DataFrame({'models':['Lasso', 'Ridge', 'Elasticnet', 'Grid_lasso', 'Grid_ridge', 'Grid_elasticnet'], 'Scores':[s1, s2, s3, s4, s5, s6]})
scores_all

'Save the Best score model'

finalgrid = enet_reg.best_estimator_
finalgrid

# Save the best model
pickle.dump(finalgrid, open('grid_elasticnet.pkl', 'wb'))

##########




# Prediction
model1 = pickle.load(open('grid_elasticnet.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

data = pd.read_excel(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 5.h-Lasso and Ridge Regression\lasso and ridge regression\carswithenginetype_test.xlsx")

clean = pd.DataFrame(impute.transform(data), columns = data.select_dtypes(exclude = ['object']).columns)
clean1 = winsor.transform(clean)
clean2 = pd.DataFrame(minmax.transform(clean1))
clean3 = pd.DataFrame(encoding.transform(data).todense())
clean_data = pd.concat([clean2, clean3], axis = 1, ignore_index = True)
clean_data1 = clean_data.drop(3, axis = 1)
prediction = pd.DataFrame(model1.predict(clean_data), columns = ['Predict_MPG'])

final = pd.concat([prediction,data], axis = 1)

final
