"""
Module Principal Component Analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def app():
  st.header('Análisis de componentes principales 📊')
  st.write("""
  En esta sección se aplica el análisis de componentes principales (PCA) para
  reducir la dimensionalidad de un conjunto de datos.
  """)

  # Check if the data has been uploaded previously
  if 'data' not in st.session_state:
    st.warning('Primero carga un conjunto de datos.')
    st.stop()
  else:
    data = dataReduced = st.session_state.data
  
  # Print the data uploaded in upload_dataset module
  st.subheader('Datos')
  st.write("""
  Análisis de componentes principales trabaja sólo con datos numéricos.
  Se muestran las variables numéricas que no contienen datos nulos dentro del
  conjunto de datos cargado.
  """)
  # Select numeric variables with non empty values
  numData = data.select_dtypes(exclude='object').dropna(axis='columns')
  st.write(numData)
  st.markdown('El conjunto de datos contiene **'+str(numData.shape[0])+
    '** registros y **'+str(numData.shape[1])+'** columnas.')
  
  # Step 1. Data standardization
  st.subheader('Estandarización de datos')
  scaler = StandardScaler()
  scaler.fit(numData)
  dataStandarized = pd.DataFrame(scaler.transform(numData),
    columns=numData.columns
  )  
  st.write('Los datos estandarizados son')
  st.write(dataStandarized)

  # Step 2 and 3 covariance matrix, eigenvectors and eigenvalues
  st.subheader('Obtención de los eigenvectores y eigenvalores')
  pca = PCA()
  pca.fit(dataStandarized)
  # Create a dataframe with the pca transform of dataStandarized
  dataReduced = pd.DataFrame(
    pca.transform(dataStandarized),
    columns=['Componente %s' % _ for _ in range(numData.shape[1])]
  )
  st.write(
    """
    Conjunto de datos utilizando los eigenvectores (componentes principales)
    como vectores base
    """
  )
  st.write(dataReduced)

  # Step 4. Principal component selection
  st.subheader('Selección del número de componentes principales')
  st.write(
    """
    Se grafica la varianza acumulada en las nuevas dimensiones para seleccionar
    el número de componentes que acumulen entre el 75 y 90% de varianza total.
    """
  )
  variances = np.cumsum(pca.explained_variance_ratio_)*100
  fig1 = go.Figure(
    data=go.Scatter(x=np.arange(start=1,stop=len(variances)+1),y=variances)
  )
  fig1.update_layout(
    xaxis_title="Número de componentes principales",
    yaxis_title="Porcentaje de varianza acumulada"
  )
  st.plotly_chart(fig1, use_container_width=True)
  #st.write(np.where(np.logical_and(variances>=75, variances<=90))[0][0])
  numComponents = st.number_input(
      label='Selecciona el número de componentes principales',
      min_value=np.where(np.logical_and(variances>=75, variances<=90))[0][0]+1,
      max_value=np.where(np.logical_and(variances>=75, variances<=90))[0][-1]+1,
      value=np.where(np.logical_and(variances>=75, variances<=90))[0][0]+1,
      step=1
  )
  numComponents = int(numComponents)

  # Step 5. Proportion of relevancies
  st.subheader('Proporción de relevancias')
  st.write(
    """
    Se revisan los valores absolutos de las componentes principales
    seleccionadas. Cuanto mayor sea el valor absoluto, más importante es esa
    variable en la componente principal.
    Para facilitar la selección de variables con alta importancia se muestra un
    mapa de calor.
    """
  )
  components = pd.DataFrame(pca.components_, columns=numData.columns)
  # Create a heatmap with the absolute value of each component
  fig2 = px.imshow(abs(components.head(numComponents)),
    labels={'x':'variables', 'y':'componentes principales', 'color':'varianza'},
    color_continuous_scale='teal'
  )
  st.plotly_chart(fig2, use_container_width=True)

  st.write(
    """
    A partir del análisis de reducción de dimensionalidad realizado.
    ¿Cuáles variables serán seleccionadas?
    """
  )
  
  # Multiselect for select the variables
  variables = st.multiselect(label='Multiselect',options=data.columns,
    default=list(data.columns))
  if not variables:
    st.error("Por favor, selecciona al menos una variable.")
  else:
    dataReduced = data[variables]
    st.write('El conjunto de datos reducido es')
    st.write(dataReduced)
    st.markdown('El conjunto de datos reducido contiene **'+
    str(dataReduced.shape[0])+
    '** registros y **'+str(dataReduced.shape[1])+
    '** columnas.')
  if st.button('Guardar conjunto de datos reducido'):
    # Save dataReduced as a session variable
    st.session_state.dataReduced = dataReduced
    st.success('Conjunto de datos guardado')
