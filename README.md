# Predicción de abandono de clientes con PyCaret

Proyecto de la materia de Ciencia de Datos.  
Integrantes: David Célleri y José Solórzano.

## Sobre el proyecto

La idea es ver si PyCaret, que es una librería de AutoML, logra predecir qué clientes se van a ir de una empresa de telecomunicaciones igual o mejor que una regresión logística hecha a mano con scikit-learn. Además queremos medir cuánto tiempo y cuántas líneas de código nos ahorra usarla, comparando los resultados de PyCaret con la línea base de scikit-learn.

Para esto usamos el dataset Telco Customer Churn de IBM, que está en [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Son 7043 clientes con 21 columnas: datos personales, servicios que tienen contratados, lo que pagan y si dejaron o no la empresa (columna `Churn`).

---

## Estructura del proyecto

El proyecto se divide en tres notebooks y una presentación:

- **[pipeline_churn.ipynb](pipeline_churn.ipynb)**: Contiene la descarga de datos, el análisis exploratorio, la limpieza, la configuración de PyCaret y la prueba de los 14 modelos de AutoML.
- **[baseline_sklearn.ipynb](baseline_sklearn.ipynb)**: Contiene la línea base hecha a mano con scikit-learn (Regresión Logística) y las visualizaciones diagnósticas.
- **[comparacion_modelos.ipynb](comparacion_modelos.ipynb)**: Compara el mejor modelo de PyCaret con la línea base, con márgenes de error, y tiene las conclusiones.
- **[presentacion_churn.pdf](presentacion_churn.pdf)**: Presentación con el resumen del trabajo.

---

## Lo que hemos hecho hasta ahora

### Descarga de los datos

El notebook descarga el dataset desde Kaggle y lo guarda en la carpeta `data/`. Por si Kaggle no funciona, dejamos como respaldo una copia del mismo archivo que IBM tiene en GitHub.

### Análisis inicial

Revisamos cómo venían los datos y qué variables tienen relación con el abandono. Lo que más nos llamó la atención fue:

- Más o menos 1 de cada 4 clientes se va (26,5 %), así que las clases no están balanceadas.
- Los que se van suelen ser clientes nuevos. La mitad de ellos lleva 10 meses o menos, mientras que en los que se quedan esa cifra sube a 38 meses.
- Se van mucho más los que tienen contrato mes a mes, fibra óptica, pagan con cheque electrónico o no tienen soporte técnico.
- El género y tener o no servicio de teléfono casi no hacen diferencia.

### Limpieza

Los datos venían bastante limpios, solo tuvimos que arreglar algunas cosas:

- `TotalCharges` venía como texto y tenía 11 valores vacíos. Revisando, eran clientes que acababan de entrar y todavía no pagaban nada, así que les pusimos 0.
- Quitamos `customerID` porque es solo un código y no sirve para predecir.
- Cambiamos `SeniorCitizen` a "Yes"/"No" para que quede igual que las demás columnas, y `Churn` a 1/0.

El dataset limpio queda guardado en `data/telco_churn_clean.csv`.

### Configuración de PyCaret y partición compartida

Con la función `setup()` dejamos listo el experimento en PyCaret. Separamos los datos en 80 % para entrenamiento y 20 % para prueba, cuidando que en los dos haya la misma proporción de clientes que se van. También usamos validación cruzada de 5 folds. PyCaret se encarga de pasar las variables categóricas a números y de escalar todo.

Guardamos además los índices exactos de entrenamiento y prueba en `data/particion_train_test.csv`. Así la regresión logística en scikit-learn usa exactamente la misma división y la comparación es 100% justa.


### Línea base con scikit-learn

En el notebook `baseline_sklearn.ipynb` armamos la línea base manual usando `ColumnTransformer` (con `StandardScaler` y `OneHotEncoder`) y `LogisticRegression`. Usamos la misma validación cruzada de 5 folds y el mismo conjunto de prueba del 20%.

Obtuvimos un AUC de 0,8422 en la prueba, prácticamente igual al 0,8418 de la regresión logística en PyCaret. Además analizamos los coeficientes: contratar fibra óptica (`InternetService_Fiber optic`) es la variable que más aumenta el riesgo de abandono, mientras que los contratos a dos años (`Contract_Two year`) y la antigüedad (`tenure`) son los factores que más ayudan a retener al cliente.

### Comparación entre PyCaret y scikit-learn

Comparamos el mejor modelo de PyCaret (Gradient Boosting) contra la regresión logística de scikit-learn. Como las métricas siempre tienen un margen de error, no basta con ver cuál número es más alto: solo decimos que un modelo es mejor si la diferencia se mantiene dentro de un intervalo de confianza del 95 % que no incluya el 0.

Para eso evaluamos los dos modelos de dos formas:

- Validación cruzada repetida (5 folds × 10 veces), usando los mismos folds para ambos.
- Bootstrap sobre el 20 % de prueba, remuestreando 2.000 veces los mismos clientes para los dos modelos.

| Métrica | PyCaret | scikit-learn | ¿Hay diferencia? |
|---|---|---|---|
| AUC | 0,847 | 0,846 | No |
| F1 | 0,587 | 0,596 | No |
| Recall | 0,526 | 0,547 | No concluyente |

En esfuerzo, PyCaret necesitó 5 líneas de código y probó 14 modelos, mientras que scikit-learn necesitó 17 líneas para un solo modelo. A cambio, PyCaret tardó unas 10 veces más (unos 16 s frente a 1,5 s).

## Conclusiones

- Los dos modelos rinden igual. En AUC y F1 la diferencia está dentro del margen de error.
- En recall la regresión logística sale un poco mejor en la prueba, pero no se confirma en la validación cruzada y depende del umbral de 0,5, así que no es concluyente.
- PyCaret ahorra código y explora muchos modelos rápido, pero tarda más y da menos control.
- Respuesta a nuestra pregunta: PyCaret iguala a la regresión logística hecha a mano, pero no la supera.
- Para este problema nos quedamos con la regresión logística, porque rinde igual, es más rápida y se puede interpretar.

Limitaciones: usamos los hiperparámetros por defecto, el umbral fijo de 0,5 y no corregimos por comparaciones múltiples.

---


## Cómo ejecutarlo

PyCaret solo funciona con Python 3.11, por eso hay que crear un entorno aparte. Nosotros usamos [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name telco-churn --display-name "Python 3.11 (telco-churn)"
```

### Orden de ejecución
1. Abrir `pipeline_churn.ipynb`, elegir el kernel "Python 3.11 (telco-churn)" y correr todas las celdas.
2. Abrir `baseline_sklearn.ipynb`, elegir el mismo kernel y correr todas las celdas para ver la comparación con scikit-learn.
3. Abrir `comparacion_modelos.ipynb`, elegir el mismo kernel y correr todas las celdas para ver la comparación final y las conclusiones.

---

## Uso de IA

Usamos inteligencia artificial (Claude Code) para generar todo el código del proyecto. Le fuimos dando instrucciones por medio de prompts y luego pidiendo ajustes hasta llegar al resultado final: la descarga de datos, el análisis, la limpieza, los modelos, la comparación, los gráficos y la presentación.

Nosotros definimos la pregunta y los criterios de comparación, revisamos y ejecutamos los notebooks, y verificamos que los resultados y las conclusiones tuvieran sentido.

## Bibliografía

- Ali, M. (2020). *PyCaret: An open source, low-code machine learning library in Python* (versión 3.3.2). https://pycaret.org
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://doi.org/10.1214/aos/1013203451
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- IBM. (s. f.). *Telco Customer Churn* [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- LeDell, E., & Poirier, S. (2020). H2O AutoML: Scalable automatic machine learning. *7th ICML Workshop on Automated Machine Learning (AutoML)*.
- McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a
- Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. *Machine Learning, 52*(3), 239–281. https://doi.org/10.1023/A:1024068626366
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
