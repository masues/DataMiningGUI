"""
Module for upload datasets
"""

import streamlit as st
import pandas as pd

def app():
  st.header('Carga de datos 📓')

  col1, col2 = st.beta_columns([2,1])
  # Code to read a single file 
  uploaded_file = col1.file_uploader('Ingresa el conjunto de datos a analizar', 
    type = ['csv', 'txt'])
  typ = col2.radio('Selecciona la extensión', ['csv', 'txt'])

  # Default dataset for debbug
  st.session_state.data = pd.read_csv('https://raw.githubusercontent.com/masues/MD_datasets/main/melb_data.csv')

  if st.button('Cargar'):
    try:
      if typ == 'csv':
        st.session_state.data = pd.read_csv(uploaded_file)
      else:
        st.session_state.data = pd.read_table(uploaded_file)
      
      # Write the uploaded dataframe
      st.subheader('Datos')
      st.write(st.session_state.data)
    except Exception as e:
      st.error('Error al tratar de cargar el conjunto de datos.'
        ' Revisa el archivo que subiste.')
