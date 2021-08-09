"""
Module Logistic Regression Class Predictor
"""

import streamlit as st
import pandas as pd

def app():

  st.header('Predicción utilizando el modelo de clasificación 🎯')

  st.markdown(
    """
    En esta sección se utiliza el modelo de clasificación entrenado en el módulo
    **Entrenamiento del clasificador** para generar predicciones.
    """
  )
  
  # Check if the model has been trained
  if 'model' and 'x_columns' in st.session_state:
    model = st.session_state.model
    x_columns = st.session_state.x_columns
    data = st.session_state.dataReduced if 'dataReduced' in st.session_state\
      else st.session_state.data
  else:
    st.warning('Primero entrena un modelo de clasificación.')
    st.stop()
  
  # Print the data
  st.subheader('Datos')
  st.write(data)
  st.markdown('El conjunto de datos contiene **'+str(data.shape[0])+
    '** registros y **'+str(data.shape[1])+'** columnas.')

  # Print the form to predict a class
  form = st.form(key='classification')
  # Create a number_input for each column
  inputs = list(map(lambda col: form.number_input(label=col,key=col),x_columns))
  submitButton = form.form_submit_button(label='Predecir clase')

  if submitButton:
    # Create a dataframe with the selected inputs
    X_data = pd.DataFrame(columns=x_columns)
    X_data.loc[0] = inputs
    
    pred = model.predict(X_data)
    st.markdown('La clase predicha es: **'+str(pred[0])+'**')
