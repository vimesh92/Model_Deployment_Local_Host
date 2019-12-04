import numpy as np
import pandas as pd
from flask import Flask, request,jsonify,render_template
import pickle

app= Flask(__name__)
model= pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_variables= [int(x) for x in request.form.values()]
    final_features = [np.array(input_variables)] 
    prediction = model.predict(final_features)
    
    output={0:'No',1:'Yes'}
    
    return render_template('main.html',result=output[prediction[0]])

if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
