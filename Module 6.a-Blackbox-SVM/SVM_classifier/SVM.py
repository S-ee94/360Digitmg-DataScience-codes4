import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import pickle, joblib

from sqlalchemy import create_engine


user = 'root'  # user name
pw = 'Seemscrazy1994#'  # password
db = 'letter_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


letters = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6.a-Blackbox-SVM\SVM_classifier\letterdata.csv")

letters.to_sql('letters_svm', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from letters_svm;'
letters = pd.read_sql_query(sql, engine)


letters.describe()

# Predictors and Target
X = letters.iloc[:, 1:]
Y = letters.iloc[:, 0]

# Numeric features
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

X[numeric_features].info()

num_pipeline1 = Pipeline(steps = [('impute1', SimpleImputer(strategy = 'mean'))])                                                

# Imputation Transformer
preprocessor = ColumnTransformer([('mean', num_pipeline1, numeric_features)])
print(preprocessor)

impute_data = preprocessor.fit(X)

# Save the data preprocessing pipeline
joblib.dump(impute_data, 'impute')

X1 = pd.DataFrame(impute_data.transform(X), columns = X.columns)
X1

X1.isna().sum()


# Outlier Treatment

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.
X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(X1.columns))

outlier_pipeline = Pipeline(steps = [('winsor', winsor)])
outlier_pipeline

preprocessor1 = ColumnTransformer(transformers = [('wins', outlier_pipeline, 
                                                   numeric_features)], 
                                  remainder = 'passthrough')
print(preprocessor1)


winz_data = preprocessor1.fit(X1)

# Save the data preprocessing pipeline
joblib.dump(winz_data, 'winzor')

X1[list(X1.columns)] = winz_data.transform(X1)

# Boxplot
X1.plot(kind = 'box', subplots = True, sharey = False, figsize = (15, 8)) 
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()

desc = X1.describe()

# Scaling with MinMaxScaler
scale_pipeline = Pipeline(steps = [('scale', MinMaxScaler())])

preprocessor2 = ColumnTransformer(transformers = [('scale', scale_pipeline,
                                                   numeric_features)], 
                                  remainder='passthrough')

print(preprocessor2)

scale = preprocessor2.fit(X1)

# Save the data preprocessing pipeline
joblib.dump(scale, 'scale')

X2 = pd.DataFrame(scale.transform(X1), columns = X1.columns)
X2.describe()


# Data Partition into Train and Test
train_X, test_X, train_y, test_y = train_test_split(X2, Y, test_size = 0.2, stratify = Y)


# Support Vector Classifier
# SVC with linear kernel trick
model_linear = SVC(kernel = "linear")
model1 = model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

# Accuracy
np.mean(pred_test_linear == test_y)


### Hyperparameter Optimization
# RandomizedSearchCV

# Base model
model = SVC()

# Parameters set
parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

# Randomized Search Technique for exhaustive search for best model
rand_search =  RandomizedSearchCV(model, parameters, n_iter = 10, 
                                  n_jobs = 3, cv = 3, scoring = 'accuracy', random_state = 0)
  
# Fitting the model for grid search
randomised = rand_search.fit(train_X, train_y)

# Best parameters
randomised.best_params_

# Best Model
best = randomised.best_estimator_

# Evaluate on Test data
pred_test = best.predict(test_X)

np.mean(pred_test == test_y)


# Saving the best model - rbf kernel model 
pickle.dump(best, open('svc_rcv.pkl', 'wb'))


##########  New Data Prediction ####

data = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6.a-Blackbox-SVM\SVM_classifier\test_svm.csv")

numeric_features = data.select_dtypes(exclude = ['object']).columns
  
model1 = pickle.load(open('svc_rcv.pkl', 'rb'))
impute = joblib.load('impute')
winzor = joblib.load('winzor')
minmax = joblib.load('scale')


clean = pd.DataFrame(impute.transform(data), columns = data.columns)
clean1 = pd.DataFrame(winzor.transform(clean), columns = data.columns)
clean2 = pd.DataFrame(minmax.transform(clean1), columns = data.columns)


prediction = pd.DataFrame(model1.predict(clean2), columns = ['choice_pred'])

final = pd.concat([prediction, data], axis = 1)
