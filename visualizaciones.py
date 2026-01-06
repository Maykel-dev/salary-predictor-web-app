import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# 1. Cargar datos y modelo
df = pd.read_csv('salarydataset.csv')
model = joblib.load('model.pkl')

# Configuración estética de las gráficas
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# --- GRÁFICA 1: Distribución de Salarios ---
plt.subplot(1, 1, 1)
sns.histplot(df['Salary'], kde=True, color='skyblue')
plt.title('Distribución Global de Salarios 💰')
plt.savefig('distribucion_salarios.png')
plt.close()

# --- GRÁFICA 2: Experiencia vs Salario ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Years of Experience', y='Salary', hue='Education Level', alpha=0.7)
plt.title('Relación: Experiencia vs Salario 📈')
plt.savefig('experiencia_vs_salario.png')
plt.close()

# --- GRÁFICA 3: Importancia de Variables ---
# Extraemos el regresor del pipeline
regressor = model.named_steps['regressor']
# Obtenemos los nombres de las columnas después de la transformación
features = model.named_steps['preprocessor'].get_feature_names_out()
importances = regressor.feature_importances_

# Creamos un DataFrame para graficar las 10 más importantes
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Variables que más influyen en el Salario 🧠')
plt.savefig('importancia_variables.png')
plt.close()

print("¡Gráficas generadas con éxito! Revisa los archivos .png en tu carpeta.")
