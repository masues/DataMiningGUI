# Interfaz gráfica para el minado de datos ⛏️
## Introducción
Este proyecto consiste en una aplicación web que permite aplicar
algoritmos de minería de datos a conjuntos de datos variables.

## Módulos de la aplicación
### Carga de datos 📓
Módulo que se encarga de cargar datos a la aplicación.
### EDA - Análisis exploratorio de datos 🔎
Permite entender la estructura del conjunto de datos, identificar la variable objetivo y posibles técnicas de modelado.
### PCA - Análisis de componentes principales 📊
Se utiliza análisis de componentes principales (ACP o PCA, Principal Component
Analysis) para reducir la cantidad de variables en el conjunto de datos, mientras
se conserva la mayor cantidad de información posible.
### k-medias - Agrupamiento particional por k-medias 🧮
Se aplica agrupamiento particional mediante el algoritmo de
k-medias utilizando el conjunto de datos obtenido en la carga de datos, EDA o PCA.
### Entrenamiento del claisificador - Modelo de Regresión Logística 📈
Se entrena un modelo clasificación por regresión logística utilizando el
conjunto de datos obtenido en la *carga de datos*, *EDA* o *PCA*.
### Predicción de clases - Predicción utilizando el modelo de clasificación 🎯
Se utiliza el modelo de clasificación entrenado en el módulo *Entrenamiento del clasificador* para generar predicciones.

# Requerimientos
- Python 3.8 o superior

# Tecnologías utilizadas
- Python 
- Streamlit 
- Pandas
- Scikit-Learn
- Plotly

# Pasos para instalar el proyecto

1. Clonar el repositorio
2. Crear una un ambiente virtual de python
```
$ python3 -m venv env
```
3. Activar el ambiente virtual (en Linux, MacOS o UNIX)
```
$ source env/bin/activate
```
3. Activar el ambiente virtual (en Windows)
```
$ env\Scripts\activate.bat
```
4. Instalar las dependencias
```
$ pip install -r requirements.txt
```
5. Correr utilizando streamlit
```
$ streamlit run app.py
```
