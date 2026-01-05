from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
# Cargar el modelo persistido (Requisito 2.2 del PDF)
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {
        'Age': [float(request.form['age'])],
        'Gender': [request.form['gender']],
        'Education Level': [request.form['education']],
        'Job Title': [request.form['job_title']],
        'Years of Experience': [float(request.form['experience'])]
    }
    
    df_input = pd.DataFrame(input_data)
    prediction = model.predict(df_input)[0]
    
    return render_template('index.html', prediction=f"{prediction:,.2f}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
