# ECG of different age groups of people has been recorded. The survival time in hours after the operation is 
# given and the event type is denoted by 1 (if dead) and 0 (if alive). Perform survival analysis on the dataset
#  given below and provide your insights in the documentation.
from lifelines import KaplanMeierFitter
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote
import pickle

user = 'root'
db = 'ecg_db'
pw = 'Seemscrazy1994#'

engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

ECG_Surv = pd.read_excel(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 7a.Survival Analytics\Assignment\ECG_Surv.xlsx")
ECG_Surv.to_sql('ecg_surv', con = engine, if_exists='replace', chunksize = 1000, index =False)

sql = 'select * from ecg_surv;'
df = pd.read_sql_query(sql, con = engine)
df.head()

df.isna().sum()

#dropping nan values

df1 = df.dropna()

df1.duplicated().sum()

df.describe()
# pip install lifelines
# Importing the KaplanMeierFitter model to fit the survival analysis

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()
df1.alive.value_counts()

#time
T = df1.iloc[:,:1]
# Fitting KaplanMeierFitter model on Time and death event
kmf = kmf.fit(T, df1.alive)
pickle.dump(kmf,open('kmf.pkl', 'wb'))

tb1 = kmf.event_table
tb2 = kmf.survival_function_
tb3 = kmf.confidence_interval_cumulative_density_
tb4 = kmf.confidence_interval_survival_function_

result_kmf = pd.concat([tb1, tb2, tb3, tb4 ], axis = 1)

# Time-line estimations plot 
kmf.plot()
#from the plot we can see that most patients die within few(5) hours of operation, over time death rate decreases

# with age
kmf1 = kmf.fit(df1.age, df1.alive)
pickle.dump(kmf1,open('kmf_age.pkl', 'wb'))

re1 = kmf1.event_table
re2 = kmf1.survival_function_
re3 = kmf1.confidence_interval_cumulative_density_
re4 = kmf1.confidence_interval_survival_function_

result_kmf1 = pd.concat([re1, re2, re3, re4 ], axis = 1)


kmf.plot()
#upto 50 years death rate is low, beyond 50 years death rate increases suddenly

#with pericardialeffusion	and fractionalshortening
kmf.fit(df1.pericardialeffusion, df1.alive)
ax = kmf.plot()
kmf.fit(df1.fractionalshortening, df1.alive)
kmf.plot(ax = ax)
# with epss
kmf.fit(df1.epss, df1.alive)
kmf.plot()

#with lvdd
kmf.fit(df1.lvdd, df1.alive)
kmf.plot()


#with group
df1.group.value_counts()
# Applying KaplanMeierFitter model on Time and alive for the group "0"
kmf.fit(T[df1.group == 1], df1.alive[df1.group == 1], label = '1')
ax = kmf.plot()
kmf.fit(T[df1.group == 2], df1.alive[df1.group == 2], label = '2')
kmf.plot(ax=ax)
kmf.fit(T[df1.group == 3], df1.alive[df1.group == 3], label = '3')
kmf.plot(ax=ax)

#from plot we can see that group1 : people has more deaths within few hours of operation but stables after few hours
#group 3 death rate is slow initial but increases sinificantly over time
#group 2 fall in between group1 and group 3

#group 2 has better success rate over other groups


