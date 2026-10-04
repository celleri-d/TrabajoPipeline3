# Predicción de abandono de clientes con PyCaret

Proyecto de la materia de Ciencia de Datos.
Integrantes: David Célleri y José Solórzano.

## Sobre el proyecto

La idea es ver si PyCaret, que es una librería de AutoML, logra predecir qué clientes se van a ir de una empresa de telecomunicaciones igual o mejor que una regresión logística hecha a mano con scikit-learn. Además queremos medir cuánto tiempo y cuántas líneas de código nos ahorra usarla.

Para esto usamos el dataset Telco Customer Churn de IBM, que está en
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Son 7043 clientes con 21 columnas: datos personales, servicios que tienen contratados, lo que pagan y si dejaron o no la empresa (columna `Churn`).

## Lo que hemos hecho hasta ahora

Todo el trabajo está en el notebook [pipeline_churn.ipynb](pipeline_churn.ipynb).

### Descarga de los datos

El notebook descarga el dataset desde Kaggle y lo guarda en la carpeta `data/`. Por si Kaggle no funciona, dejamos como respaldo una copia del mismo archivo que IBM tiene en GitHub.

### Análisis inicial

Revisamos cómo venían los datos y qué variables tienen relación con el abandono. Lo que más nos
llamó la atención fue:

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

### Configuración de PyCaret

Con la función `setup()` dejamos listo el experimento. Separamos los datos en 80 % para entrenamiento y 20 % para prueba, cuidando que en los dos haya la misma proporción de clientes que se van. También usamos validación cruzada de 5 folds. PyCaret se encarga de pasar las variables categóricas a números y de escalar todo.

Guardamos además qué filas quedaron en entrenamiento y cuáles en prueba en
`data/particion_train_test.csv`. Así la regresión logística en scikit-learn puede usar exactamente la misma división y la comparación es justa.

### Comparación de modelos

Con `compare_models()` PyCaret probó 14 modelos distintos en unos 14 segundos, todos con la misma validación cruzada. Los ordenamos por AUC porque es la métrica principal que elegimos y no se ve tan afectada por el desbalance de clases.

El mejor fue Gradient Boosting, con un AUC de 0,847, pero la diferencia con los siguientes es mínima. La regresión logística quedó segunda con 0,846 y AdaBoost tercero con 0,844, así que en la práctica están empatados. Al probarlos con el 20 % de datos que habíamos separado, los resultados se mantuvieron casi iguales, y la regresión logística incluso tuvo mejor recall y F1.

Algo que notamos es que todos los modelos detectan más o menos la mitad de los clientes que se van (recall cercano a 0,5), así que hay margen para mejorar en ese punto.

## Cómo ejecutarlo

PyCaret solo funciona con Python 3.11, por eso hay que crear un entorno aparte. Nosotros usamos
[uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name telco-churn --display-name "Python 3.11 (telco-churn)"
```

Después abren `pipeline_churn.ipynb`, eligen el kernel "Python 3.11 (telco-churn)" y corren
todas las celdas. Los datos se descargan solos.
