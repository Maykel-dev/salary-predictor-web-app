from flask import Flask, render_template, request
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Cargamos el modelo y el dataset para las comparativas
model = joblib.load('model.pkl')
df = pd.read_csv('salarydataset.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capturar datos del formulario
    age = request.form.get('age')
    gender = request.form.get('gender')
    education = request.form.get('education')
    job_title = request.form.get('job_title')
    experience = request.form.get('experience')

    # 2. Realizar la predicción
    input_df = pd.DataFrame([[float(age), gender, education, job_title, float(experience)]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    prediction = float(model.predict(input_df)[0])

    # 3. Obtener el promedio del dataset para comparar
    avg_val = df[df['Education Level'] == education]['Salary'].mean()
    if pd.isna(avg_val):
        avg_val = df['Salary'].mean()

    # 4. CREACIÓN MANUAL DEL GRÁFICO (Solución definitiva para las barras en blanco)
    graph_data = {
        "data": [
            {
                "x": ["Tu Predicción", f"Promedio {education}"],
                "y": [prediction, float(avg_val)],
                "type": "bar",
                "marker": {"color": ["#0d6efd", "#ffc107"]}, # Azul y Amarillo
                "name": "Salario"
            }
        ],
        "layout": {
            "title": {"text": "Comparativa de Salarios"},
            "yaxis": {"title": "Salario ($)", "range": [0, max(prediction, avg_val) * 1.2]},
            "xaxis": {"title": "Categoría"},
            "template": "plotly_white",
            "margin": {"t": 50, "b": 50, "l": 50, "r": 50}
        }
    }
    
    # Convertimos el diccionario directamente a texto JSON
    graphJSON = json.dumps(graph_data)

    return render_template('index.html', 
                           prediction_text=f'Salario Estimado: ${prediction:,.2f}',
                           graphJSON=graphJSON,
                           age=age, gender=gender, education=education, 
                           job_title=job_title, experience=experience)

if __name__ == '__main__':
    app.run(debug=True)