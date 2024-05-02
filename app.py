import pandas as pd
import pickle
from flask import Flask, render_template,request
import numpy as np
import joblib

app = Flask(__name__)

Log_reg=open('mental_health_model.pkl','rb')
Log_reg_model=joblib.load(Log_reg)


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=='POST':
        try:
            Gender=float(request.form['Gender'])
            Age=float(request.form['Age'])
            Course=float(request.form['Course'])
            Year_of_Study=float(request.form['Year_of_Study'])
            CGPA=float(request.form['CGPA'])
            Marital_Status=float(request.form['Marital_Status'])
            Depression=float(request.form['Depression'])
            Anxiety=float(request.form['Anxiety'])
            Panic_Attack=float(request.form['Panic_Attack'])
            pred_args=[Gender,Age,Course,Year_of_Study,CGPA,Marital_Status,Depression,Anxiety,Panic_Attack]
            pred_args_arr=np.array(pred_args)
            pred_args_num=pred_args_arr.reshape(-1,1)
            model_prediction=Log_reg_model.predict(pred_args_num)
        except ValueError:
            return "Please check the entered value are correct"
    return render_template('predict.html',prediction=model_prediction)

if __name__=="__main__":
    app.run()


