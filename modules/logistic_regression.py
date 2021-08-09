"""
Moldule Logistic Regression Classifier Model
"""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px

@st.cache
def getMeanValues(data,classVar):
  return data.groupby(classVar).agg(['mean'])

@st.cache
def splitData(X,y,testSize):
  return train_test_split(X, y, test_size=testSize/100, shuffle = True)

def app():
  st.header('Modelo de Regresión Logística 📈')
  st.write(
    """
    En esta sección se entrena un modelo clasificación por regresión logística.
    """
  )
  
  # Check if the data has been uploaded previously
  if 'dataReduced' in st.session_state:
    data = st.session_state.dataReduced
  elif 'data' in st.session_state:
    data = st.session_state.data
  else:
    st.warning('Primero carga un conjunto de datos.')
    st.stop()
  
  # Print the data uploaded in upload_dataset module
  st.subheader('Datos')
  st.write(data)
  st.markdown('El conjunto de datos contiene **'+str(data.shape[0])+
    '** registros y **'+str(data.shape[1])+'** columnas.')
  
  # Class variable definition
  st.subheader('Definción de variables predictoras y variables clase')
  classVar = st.selectbox('Selecciona a la variable clase', data.columns)
  st.write(
    """
    El promedio de el resto de variables con respecto a la variable clase es
    """
  )
  st.write(getMeanValues(data,classVar))
  y = data[classVar] # Class variable

  col1, col2 = st.beta_columns(2)
  col1.write('Los valores de la variable clase son')
  col1.write(list(y.unique()))

  # Define the independent variables based on the remaining numeric variables
  X = data.drop([classVar], axis='columns').select_dtypes(exclude='object').\
    dropna(axis='columns')
  if X.empty:
    st.error(
      """
      Error. Todas las columnas numéricas del conjunto de datos leido poseen
      datos nulos. Primero elimina registros nulos dentro del módulo de EDA
      para poder continuar.
      """
    )
    st.stop()
  col2.write('Las variables independientes son')
  col2.write(list(X.columns))
  
  # Train data and test data split
  testSize = st.slider(
    label='Selecciona el procentaje de los datos de prueba', min_value=0,
    max_value=50, format='%d%%', value=20
  )
  
  X_train, X_test, y_train, y_test = splitData(X,y,testSize)

  # Training the model
  if st.button('Entrenar mdoelo'):
    st.spinner()
    with st.spinner(text='Entrenando'):
      # Train the model with training data
      classModel = LogisticRegression().fit(X_train,y_train)
      # Save the model, dependent and independent variables for subsequent
      # predictions
      st.session_state.model = classModel
      st.session_state.x_columns = X.columns
      st.success('Modelo entrenado')
  else:
    st.stop()
  
  st.subheader('Validación del modelo')
  # Mean accuracy
  st.markdown('El modelo tiene una exactitud media de **'+
    str(classModel.score(X_test,y_test))+'**'
  )
  y_pred = classModel.predict(X_train)

  # Classification report
  st.write('Reporte de clasificación')
  st.text(classification_report(y_train,y_pred))

  # Confusion matrix
  confMat = pd.crosstab(y_train.values.ravel(), y_pred,
    rownames=['Real'], colnames=['Predicho']
  )
  fig = px.imshow(confMat,color_continuous_scale='teal',
    title='Matriz de confusión', labels={'color':'conteo'}
  )
  st.plotly_chart(fig, use_container_width=True)
