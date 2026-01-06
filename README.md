# 📊 Salary Predictor Web App - Dashboard Interactivo 🚀

Esta aplicación utiliza Machine Learning para predecir el salario de profesionales basado en su perfil demográfico y laboral, ofreciendo una **visualización dinámica** para comparar resultados en tiempo real.



## 🚀 Características de la Versión Interactiva
- **Gráficos con Plotly:** Comparativa visual entre tu predicción y el promedio del mercado según el nivel educativo.
- **Interfaz Moderna:** Diseño responsivo construido con Bootstrap 5.
- **Persistencia de Datos:** El formulario mantiene los valores ingresados para facilitar ajustes rápidos sin perder la información.

## 📈 Métricas del Modelo
El modelo fue entrenado con un **Random Forest Regressor** utilizando un Pipeline de Scikit-Learn:
* **MAE (Error Absoluto Medio):** 2614.76
* **R2 Score:** 0.98 (98% de precisión en los datos de prueba).

## 🛠️ Pipeline de Procesamiento (Requisito A)
Se implementó un `Pipeline` que realiza:
1. **Imputación:** Manejo de valores nulos (mediana para números, valor más frecuente para categorías).
2. **Codificación:** Transformación de variables de texto a numéricas con `OneHotEncoder`.
3. **Regresión:** Entrenamiento del modelo RandomForest.
## 📸 Demo Visual
![Dashboard Interactivo](https://raw.githubusercontent.com/Maykel-dev/salary-predictor-web-app/main/captura.png)

## 🧠 Reflexión
* **¿Qué característica influye más en el salario?**
  Los años de experiencia son el factor determinante en las predicciones.
* **¿Qué limitación tiene tu modelo?**
  El R2 de 0.98 sugiere que el modelo es muy preciso para este dataset, pero podría no generalizar igual de bien con datos de diferentes países o economías.
* **¿Qué mejorarías con más tiempo?**
  Implementaría una comparación en tiempo real con una Regresión Lineal simple y añadiría gráficos de importancia de características directamente en la interfaz.

---


