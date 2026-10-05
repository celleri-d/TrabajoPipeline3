# Predicción de abandono de clientes con PyCaret

Proyecto de la materia de Ciencia de Datos.  
Integrantes: David Célleri y José Solórzano.

## Sobre el proyecto

La idea es ver si PyCaret, que es una librería de AutoML, logra predecir qué clientes se van a ir de una empresa de telecomunicaciones igual o mejor que una regresión logística hecha a mano con scikit-learn. Además queremos medir cuánto tiempo y cuántas líneas de código nos ahorra usarla, comparando los resultados de PyCaret con la línea base de scikit-learn.

Para esto usamos el dataset Telco Customer Churn de IBM, que está en [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Son 7043 clientes con 21 columnas: datos personales, servicios que tienen contratados, lo que pagan y si dejaron o no la empresa (columna `Churn`).

---

## Estructura del proyecto

El proyecto se divide en dos notebooks:

- **[pipeline_churn.ipynb](pipeline_churn.ipynb)**: Contiene la descarga de datos, el análisis exploratorio, la limpieza, la configuración de PyCaret y la prueba de los 14 modelos de AutoML.
- **[baseline_sklearn.ipynb](baseline_sklearn.ipynb)**: Contiene la línea base hecha a mano con scikit-learn (Regresión Logística) y las visualizaciones diagnósticas.

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

Guardamos además los índices exactos de entrenamiento y prueba en `data/partition_indices.json`. Así la regresión logística en scikit-learn usa exactamente la misma división y la comparación es 100% justa.


### Línea base con scikit-learn

En el notebook `baseline_sklearn.ipynb` armamos la línea base manual usando `ColumnTransformer` (con `StandardScaler` y `OneHotEncoder`) y `LogisticRegression`. Usamos la misma validación cruzada de 5 folds y el mismo conjunto de prueba del 20%.

Obtuvimos un AUC de 0,8422 en la prueba, prácticamente igual al 0,8418 de la regresión logística en PyCaret. Además analizamos los coeficientes: contratar fibra óptica (`InternetService_Fiber optic`) es la variable que más aumenta el riesgo de abandono, mientras que los contratos a dos años (`Contract_Two year`) y la antigüedad (`tenure`) son los factores que más ayudan a retener al cliente.

---|


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
