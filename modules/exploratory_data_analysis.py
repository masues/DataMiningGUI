"""
Module Exploratory Data Analysis
"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

@st.cache
def getHistogram(data,dataColumn):
  # Histogram
  fig1 = px.histogram(data, x=dataColumn)
  # Violin diagram
  fig2 = px.violin(data, x=dataColumn)
  return fig1,fig2

@st.cache
def getCorrelationMatrix(data):
  return px.imshow(data.corr(),
    labels={'x':'variable 1', 'y':'variable 2', 'color':'correlación'},
    title='Matriz de correlaciones',color_continuous_scale='reds'
  )

@st.cache
def hasCategoricalData(data):
  return np.dtype('O') in list(data.dtypes)

@st.cache
def getPossibleClasses(data,numClasses):
  possibleCols = []
  for col in data.select_dtypes(include='object'):
    if data[col].nunique() <= numClasses:
      possibleCols.append(col)
  return possibleCols

@st.cache
def getMeanValues(data,classVar):
  return data.groupby(classVar).agg(['mean'])

def getColumnsWithNulls(data):
  return data.loc[:, data.isnull().any()].columns

def app():
  st.header('Análisis exploratorio de datos 🔎')

  # Check if the data has been uploaded previously
  if 'data' not in st.session_state:
    st.warning('Primero carga un conjunto de datos.')
    st.stop()
  else:
    data = dataReduced = st.session_state.data
  
  # Print the data uploaded in upload_dataset module
  st.subheader('Datos')
  st.write(data)

  col1, col2 = st.beta_columns(2)
  
  # Step 1. Data structure description
  col1.subheader('Descripción de la estructura de los datos')
  col1.markdown('El conjunto de datos contiene **'+str(data.shape[0])+
    '** registros y **'+str(data.shape[1])+'** columnas.')
  col1.write('Los tipos de datos son:')
  dataTypes = pd.DataFrame(data.dtypes,columns=['Tipo de dato'])
  dataTypes.reset_index(level=0,inplace=True)
  dataTypes.rename(columns={'index':'Variable',0:'Tipo de dato'},inplace=True)
  col1.dataframe(dataTypes)

  # Step 1. Empty data detection
  col2.subheader('Identificación de datos faltantes')
  col2.write('La cantidad de datos faltantes por variable es:')
  nullData = pd.DataFrame(data.isnull().sum())
  nullData.reset_index(level=0,inplace=True)
  nullData.rename(columns={'index':'Variable',0:'No. Nulos'},inplace=True)
  col2.dataframe(nullData,height=362)

  # Step 3. Atypical data detection
  st.subheader('Detección de datos atípicos')
  
  st.write('Generación gráficos por variable')
  dataColumn = st.selectbox('Selecciona a la variable', data.columns)
  fig1, fig2 = getHistogram(data,dataColumn)
  
  st.markdown('*Histograma*')
  st.plotly_chart(fig1, use_container_width=True)
  
  st.markdown('*Diagrama de Violín*')
  st.plotly_chart(fig2, use_container_width=True)

  st.write('Resumen estadístico de variables numéricas')
  st.write(data.describe())

  # Check if the dataframe has categorical data
  if(hasCategoricalData(data)):
    st.write('Resumen estadístico de variables categóricas')
    st.write(data.describe(include='object'))

    # Possible variable class detection
    st.write('Posibles variables clase')
    col3, col4 = st.beta_columns(2)
    numClasses = col3.number_input(
      label='Selecciona el número de clases máximo',
      value=10,step=1
    )
    numClasses = int(numClasses)

    col4.write('Las posibles variables clase son')
    possibleCols = getPossibleClasses(data,numClasses)
    col4.write(possibleCols)
    if possibleCols:
      st.write('Valores promedio con respecto a la variable clase')
      classVar = st.selectbox(label='Selecciona a la variable clase',
        options=possibleCols,key='varClass')
      st.write(getMeanValues(data,classVar))
  
  # Step 4. Relatioship between pair of variables detection
  st.subheader('Identificación de relaciones entre pares de variables')
  st.write('Mapa de calor de la matriz de correlaciones')
  fig3 = getCorrelationMatrix(data)
  st.plotly_chart(fig3, use_container_width=True)

  st.write(
    """
    A partir del análisis correlacional realizado. ¿Cuáles variables serán 
    seleccionadas?
    """
  )

  # Multiselect for select the variables
  variables = st.multiselect(
    label='Selecciona las columnas del conjunto de datos que se utilizarán',
    options=data.columns, default=list(data.columns), key='variables'
  )
  varWithNulls = st.multiselect(
    label='Selecciona las columnas que contienen registros nulos para eliminar',
    options=getColumnsWithNulls(data[variables]), key='varWithNulls'
  )
  if not variables:
    st.error("Por favor, selecciona al menos una variable.")
    st.stop()
  else:
    dataReduced = data.dropna(subset=varWithNulls) if varWithNulls else data
    dataReduced = dataReduced[variables]
    st.write('El conjunto de datos reducido es')
    st.write(dataReduced)
    st.markdown('El conjunto de datos reducido contiene **'+
    str(dataReduced.shape[0])+
    '** registros y **'+str(dataReduced.shape[1])+
    '** columnas.')
  if st.button('Guardar conjunto de datos reducido'):
    # Save the dataReduced as a session variable
    st.session_state.dataReduced = dataReduced
    st.success('Conjunto de datos guardado')
