from flask import Flask, render_template, request
from keras.models import model_from_json
import numpy as np
import pandas as pd
# imports

#from keras import model_from_json 

# opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model

loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success/',methods=['GET','POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_csv(f)
        data_pred = data.values.astype('float32')
       
        data_pred = data_pred/255

        prediction = pd.DataFrame(np.argmax(loaded_model.predict(data_pred), axis = 1), columns = ['label'])

        final = pd.concat([prediction, data], axis = 1)
        html_table = final.to_html(classes='table table-striped')
        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #8f6b39;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #32b8b8;\
                    }}\
                            .table tbody th {{\
                            background-color: #3f398f;\
                        }}\
                </style>\
                {html_table}") 
if __name__ == '__main__':
    app.run(debug=True)