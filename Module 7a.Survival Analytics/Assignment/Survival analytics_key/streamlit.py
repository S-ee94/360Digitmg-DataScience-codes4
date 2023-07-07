import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st 

# from lifelines import KaplanMeierFitter
from urllib.parse import quote
from sqlalchemy import create_engine
global kmf1
kmf1 = pickle.load(open('kmf.pkl','rb'))
global kmf2
kmf2 = pickle.load(open('kmf_age.pkl','rb'))



def Survival(data,user,pw,db):
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    re1 = kmf1.event_table
    re2 = kmf1.survival_function_
    re3 = kmf1.confidence_interval_cumulative_density_
    re4 = kmf1.confidence_interval_survival_function_

    result_kmf1 = pd.concat([re1, re2, re3, re4 ], axis = 1)

    result_kmf1.to_sql('result_kmf1', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
    return result_kmf1

def Survival2(data,user,pw,db):
    engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))
    re1 = kmf2.event_table
    re2 = kmf2.survival_function_
    re3 = kmf2.confidence_interval_cumulative_density_
    re4 = kmf2.confidence_interval_survival_function_

    result_kmf1 = pd.concat([re1, re2, re3, re4 ], axis = 1)

    result_kmf1.to_sql('result_kmf1', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
    return result_kmf1

def main():
    st.title("Survival_Analytics")
    st.sidebar.title("Survival_Analytics")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Survival_Analytics</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data=pd.DataFrame(uploadedFile)
        
        
    else:
        st.sidebar.warning("you need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    

    
    if st.button("Predict"):
        
        st.subheader(":red[KaplanMeierFitter_1]", anchor=None)
        results = Survival(data, user, pw, db)
        
      
        # cm = sns.light_palette("blue", as_cmap=True)
       
        st.dataframe(results.iloc[:,:-2])

        st.subheader(":red[plot]", anchor=None)

        #########
        fig1, ax1 = plt.subplots()
        kmf1.plot(ax=ax1)
        ax1.set_title('Survival with confidence intervals')
        st.pyplot(fig1)
        
        st.subheader(":red[KaplanMeierFitter_2]", anchor=None)
        results = Survival2(data, user, pw, db)
        
      
        # cm = sns.light_palette("blue", as_cmap=True)
       
        st.dataframe(results.iloc[:,:-2])
        
        st.subheader(":red[plot]", anchor=None)
        fig2 = plt.figure()
       
        ax2 = fig2.add_subplot(111)
        
        kmf2.plot(ax=ax2)
        st.pyplot(plt)
        
        st.text("")
        
        

                           
if __name__=='__main__':
    main()


