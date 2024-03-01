from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def new():
    return render_template('new.html')


@app.route('/predict', methods=['POST','GET'])
def predict():

    data=float(request.form['model_input'])
    
    features = np.array([[data**3, data**2, data**1, data**0]])
    
    model=pickle.load(open('model.pickle','rb'))
    pred = model.predict(features)[0][0]


    prediction_statement =  f"The output of the model is {pred}"
    
    return render_template('new.html',statement=prediction_statement)


if __name__=='__main__':
    app.run()