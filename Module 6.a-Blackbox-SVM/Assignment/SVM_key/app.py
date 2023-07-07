from flask import Flask, render_template, request
from sqlalchemy import create_engine
from urllib.parse import quote
import pandas as pd
app = Flask(__name__)

import joblib
import pickle
model = pickle.load(open('svc_rcv.pkl', 'rb'))
imp_enc_scale = joblib.load('imp_enc_scale')  # Imputation and Scaling pipeline
winsor = joblib.load('winsor')
winsor2 = joblib.load('winsor2')

def SVM(data):
    clean1 = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())
    clean1[['numerical__age','numerical__educationno','numerical__hoursperweek']] = winsor.transform(clean1[['numerical__age','numerical__educationno','numerical__hoursperweek']])
    clean1[['numerical__capitalgain','numerical__capitalloss']] = winsor2.transform(clean1[['numerical__capitalgain','numerical__capitalloss']])
    
    prediction = pd.DataFrame(model.transform(clean1), columns=['Salary'])
    
    final = pd.concat([prediction, data], axis = 1)
    
    return(final)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']
        db = request.form['databasename']
        engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote (f'{pw}'))
        try:

            data = pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)
                    
                  
        # Drop the unwanted features

        prediction=SVM(data)
        
        prediction.to_sql('svm_prediction', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = prediction.to_html(classes = 'table table-striped')
        
        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")
       

if __name__=='__main__':
    app.run(debug = True)
