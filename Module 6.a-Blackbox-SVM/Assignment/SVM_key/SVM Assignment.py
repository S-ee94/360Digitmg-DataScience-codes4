import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler #used to standardize the input features and encode categorical features, respectively
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer
import pickle
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import joblib #used to save and load trained models as binary files.
from sklearn.model_selection import RandomizedSearchCV



# Load the dataset


from sqlalchemy import create_engine


user = 'root'  # user name
pw = 'Seemscrazy1994#'  # password
db = 'salary_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")


salary = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6.a-Blackbox-SVM\Assignment\SVM_key\SalaryData_Train.csv")

salary.to_sql('salary_svm', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql = 'select * from salary_svm;'
train_df = pd.read_sql_query(sql, engine)



test_df = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6.a-Blackbox-SVM\Assignment\SVM_key\SalaryData_Test.csv")
train_df.head()
test_df.head()
# Perform univariate analysis
print(train_df.describe())
# Perform univariate analysis
print(test_df.describe())
# Perform bivariate analysis on training data
corr_train = train_df.corr()
print(corr_train)
# Perform bivariate analysis on test data
corr_test = train_df.corr()
print(corr_test)
# check for missing values in training data set
train_df.isna().sum()
# check for missing values in test data set
test_df.isna().sum()
train_df.dtypes


# AutoEDA
# Automated Libraries
# pip install dtale
import dtale
d = dtale.show(train_df)
d.open_browser()


X = train_df.iloc[:, :-1]
Y = train_df[['Salary']]
label_encoder = LabelEncoder()
Y['Salary']= label_encoder.fit_transform(Y['Salary'])
numeric_features = X.select_dtypes(exclude = ['object']).columns
numeric_features

categorical_features = X.select_dtypes(include = ['object']).columns
categorical_features

num_pipeline = Pipeline([('impute',SimpleImputer(strategy = 'mean')),('scale',MinMaxScaler())])
num_pipeline
cat_pipeline = Pipeline([('cat',OneHotEncoder(sparse_output=False))])
#Encoding on State column
preprocess_pipeline = ColumnTransformer([('numerical', num_pipeline, numeric_features),
                                         ('categorical', cat_pipeline, categorical_features)])
imp_enc_scale = preprocess_pipeline.fit(X)  # Pass the raw data through pipeline

imp_enc_scale
joblib.dump(imp_enc_scale, 'imp_enc_scale')
clean_data = pd.DataFrame(imp_enc_scale.transform(X), columns = imp_enc_scale.get_feature_names_out())
clean_data.head()
clean_data.iloc[:,0:5].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''

# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


list(clean_data.iloc[:,0:5])
#### Outlier analysis: Columns 'months_loan_duration', 'amount', and 'age' are continuous, hence outliers are treated
winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = ['numerical__age','numerical__educationno','numerical__hoursperweek'])
 
winsor2 =  Winsorizer(capping_method = 'gaussian', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 0.05,
                          variables = ['numerical__capitalgain','numerical__capitalloss'])


outlier = winsor.fit(clean_data[['numerical__age','numerical__educationno','numerical__hoursperweek']])
outlier2 = winsor2.fit(clean_data[['numerical__capitalgain','numerical__capitalloss']])
joblib.dump(outlier, 'winsor')

clean_data[['numerical__age','numerical__educationno','numerical__hoursperweek']] = outlier.transform(clean_data[['numerical__age','numerical__educationno','numerical__hoursperweek']])

joblib.dump(outlier2, 'winsor2')

clean_data[['numerical__capitalgain','numerical__capitalloss']] = outlier2.transform(clean_data[['numerical__capitalgain','numerical__capitalloss']])

clean_data.iloc[:,0:5].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 

'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''


# increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


clean_data.head()
X_test = test_df.iloc[:, :-1]
Y_test = test_df[['Salary']]

Y_test['Salary']= label_encoder.fit_transform(Y_test['Salary'])
test_data = pd.DataFrame(imp_enc_scale.transform(X_test), columns = imp_enc_scale.get_feature_names_out())
test_data[['numerical__age','numerical__educationno','numerical__hoursperweek']] = outlier.transform(test_data[['numerical__age','numerical__educationno','numerical__hoursperweek']])
test_data[['numerical__capitalgain','numerical__capitalloss']] = outlier2.transform(test_data[['numerical__capitalgain','numerical__capitalloss']])
# Train the SVM classifier

svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(clean_data, Y)
# Make predictions on the test set
y_pred = svm.predict(test_data)
# Evaluate the performance of the SVM classifier
print(confusion_matrix(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
np.mean(y_pred == Y_test.Salary)
# Hyperparameter Tuning
############  RandomizedSearchCV
model = SVC()

parameters = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['linear', 'poly', 'rbf']}
rand_search =  RandomizedSearchCV(model, parameters, n_iter = 10, n_jobs = 3, cv = 3, random_state = 0)
# fitting the model for grid search
randomised = rand_search.fit(clean_data, Y)
randomised.best_params_
best = randomised.best_estimator_


pred_test_random = best.predict(test_data)

np.mean(pred_test_random == Y_test.Salary)

#saving rbf kernel model because accuracy of rbf kernel is more 
pickle.dump(best,open('svc_rcv.pkl','wb'))
#from the above model we can preidct the salaries of localities and help construction firm with their plan.
import os
os.getcwd()
