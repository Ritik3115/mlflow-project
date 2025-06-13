from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from Mlflow_project.pipeline.prediction import PredictionPipeline
import traceback

app = Flask(__name__)
@app.route('/', methods = ['GET'])
def homepage():
    return render_template("index.html")

@app.route('/train', methods = ['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!"

@app.route('/prediction', methods = ['POST','GET'])
def index():
    if request.method == 'POST':
        try:
            
            children = int(request.form.get('children'))
            bmi = float(request.form.get('bmi'))
            age = int(request.form.get('age'))

            sex_raw = request.form.get('sex').lower()
            sex = 1 if sex_raw == 'male' else 0

            smoker_raw = request.form.get('smoker').lower()
            smoker = 1 if smoker_raw == 'yes' else 0

            region_raw = request.form.get('region').lower()
            region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}
            region = region_map.get(region_raw, -1)


            data = [age, sex, bmi, children, smoker, region]
            column = ["age", "sex", "bmi", "children", "smoker", "region"]
            data = pd.DataFrame([data], columns= column)

            data['bmi_category'] = pd.cut(
            data['bmi'], 
            bins=[0, 18.5, 25, 30, 40, 50, 60],
            labels=['underweight', 'normal', 'overweight', 'obese', 'obese +', 'obese ++']
            )

            data['age_group'] = pd.cut(
                data['age'], 
                bins=[0, 18, 60, 100],
                labels=['child', 'adult', 'senior citizen']
            )

            data['bmi_category'] = data['bmi_category'].cat.codes
            data['age_group'] = data['age_group'].cat.codes

            data['age_bmi'] = data['age'] * data['bmi']
            data['bmi_smoker'] = data['bmi'] * data['smoker']
            data['age_smoker'] = data['age'] * data['smoker']
            data['children_age_ratio'] = data['children'] / (data['age'] + 1)

            selected_features = ['smoker', 'bmi_smoker', 'age_bmi', 'age', 'age_smoker', 'bmi', 'bmi_category']
            data = data[selected_features]


            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = round(float(predict),3))
        
        except Exception as e:
            print("The Exception message is:", e)
            traceback.print_exc()
            return "Something is wrong"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 8080)