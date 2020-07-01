#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model_final.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('j1.html')

@app.route('/predict',methods=['POST'])
def predict():
    features=[]
    features.append(request.form.get('vintage'))
    features.append(request.form.get('age'))
    if request.form.get('gender') is "Male":
        features.append(1)
    elif request.form.get('gender') is "Female":
        features.append(0)
    else:
        features.append(-1)
    features.append(request.form.get('dependents'))
    features.append(request.form.get('city'))
    features.append(request.form.get('branch_code'))
    features.append(request.form.get('days_since_last_transaction'))
    if request.form.get('occupation') is "company":
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
    elif request.form.get('occupation') is "retired":
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
        features.append(0)
    elif request.form.get('occupation') is "salaried":
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
        features.append(0)
    elif request.form.get('occupation') is "self_employed":
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
        features.append(0)
    else:
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(0)
        features.append(1)
    
    features.append(request.form.get('customer_nw_category'))
    features.append(request.form.get('current_balance'))
    features.append(request.form.get('previous_month_end_balance'))
    features.append(request.form.get('avarage_monthly_end_balance_prevQ2'))
    features.append(request.form.get('average_monthly_end_balance_prevQ'))
    features.append(request.form.get('current_month_credit'))
    features.append(request.form.get('previous_month_credit'))
    features.append(request.form.get('previous_month_debit'))
    features.append(request.form.get('current_month_debit'))
    features.append(request.form.get('current_month_balance'))
    features.append(request.form.get('previous_month_balance'))
    
    
    final_features=[np.array(features)]
    prediction=model.predict(final_features)
   
    return render_template('j1.html',prediction_text="Customer may leave" if prediction == 1  else "Customer may not leave")      
    
if __name__ == "__main__":
    app.run()


# In[ ]:




