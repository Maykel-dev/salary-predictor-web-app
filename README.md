# Salary Predictor Web Application 🚀

Esta aplicación utiliza Machine Learning para predecir el salario de profesionales basado en su perfil demográfico y laboral.

## 📊 Métricas del Modelo
El modelo fue entrenado con un **Random Forest Regressor** utilizando un Pipeline de Scikit-Learn:
* **MAE (Error Absoluto Medio):** 2614.76
* **R2 Score:** 0.98 (98% de precisión en los datos de prueba).

## 🛠️ Pipeline de Procesamiento (Requisito A)
Se implementó un `Pipeline` que realiza:
1. **Imputación:** Manejo de valores nulos (mediana para números, valor más frecuente para categorías).
2. **Codificación:** Transformación de variables de texto a numéricas con `OneHotEncoder`.
3. **Regresión:** Entrenamiento del modelo RandomForest.

## 🧠 Reflexión (Mandatoria)
* **¿Qué característica influye más en el salario?**
  Los años de experiencia son el factor determinante en las predicciones.
* **¿Qué limitación tiene tu modelo?**
  El R2 de 0.98 sugiere que el modelo es muy preciso para este dataset, pero podría no generalizar igual de bien con datos de diferentes países o economías.
* **¿Qué mejorarías con más tiempo?**
  Compararía el modelo con una Regresión Lineal simple y añadiría gráficos de importancia de características.

---
**Autor:** Maykel Santos
