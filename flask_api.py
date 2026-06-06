"""
A simple flask web app for the developer salary prediction model
Run
 python flask_api.py

 then open in browser:
    https://127.0.0.1.5000

"""
import os  # Helps in navigation
import joblib # Help load our model
import numpy as np
import pandas as pd
from flask import Flask,request,render_template_string,jsonify

# App setup
app=Flask(__name__)


MODEL_PATH= 'Models/salary_pipeline.pkl'


# Picking the pipeline
pipeline =joblib.load(MODEL_PATH)
print(f"Model loaded from {MODEL_PATH}")

# Html file
HTML ="""
<html>
    <head><title> Salary Predictor </title></head>

    <body>
        <h1> Developer salary predictor </h1>

        <form method="POST" action="/predict">
            <label> Country:</label> <br>
            <select name ="country">
                <option>United States of America </option>
                <option>Germany</option>
                <option>United Kingdom of Great Britain and Nothern Ireland</option>
                <option>India</option>
                <option>Canada</option>
                <option>Brazil</option>
                <option>Other</option>
            </select>
            <br><br>

            <label> Education level:</label> 
            <select name ="education">
                <option>Bachelor's</option>
                <option>Master's</option>
                <option>PhD</option>
                <option>Associate's</option>
                <option>Some college</option>
                <option>High School</option>
            </select>
            <br><br>

            <label> Employment type:</label> 
            <select name ="employment">
                <option>Full-Time</option>
                <option>Part-time</option>
                <option>Student</option>
                <option>Freelance/Self-employed</option>
                <option>Other</option>
            </select>
            <br><br>

             <label> Yearsof experience:</label>
             <input type= "number" name="years" value="5" min="0" maxx="40">
             <br><br>

             <label> Number of programming languages used:</label>
             <input type= "number" name="languages" value="5" min="0" maxx="40">
             <br><br>

             <button type="submit"> Predict salary </button>
        </form>

        {% if salary %}
            <hr>
            <h2> Predicted Salary: ${{salary}}</h2>
        {%endif %}
    </body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route("/predict",methods=['POST'])
def predict():
    input_df=pd.DataFrame([{
        'Country': request.form.get('country'),
        'YearsCode':float(request.form.get('years')),
        'EdLevel':request.form.get('education'),
        'Employment':request.form.get('employment'),
        'LanguageHaveWorkedWith':float(request.form.get('languages'))
    }])

    prediction= pipeline.predict(input_df)[0]
    salary= f"{int(np.clip(prediction,10_000,500_000)):,}" # sets limits of the max and min prediction
    return render_template_string(HTML, salary=salary)

if __name__ =='__main__':
    app.run(debug=True)



