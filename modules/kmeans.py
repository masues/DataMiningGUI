"""
Moldule k-means clustering
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

@st.cache
def getSSE(data,kMax):
  SSE = [] # Sum of squared errors
  for i in range(2,kMax):
    kMeans = KMeans(n_clusters=i,random_state=0)
    kMeans.fit(data)
    SSE.append(kMeans.inertia_)
  return SSE

def app():
  st.header('Agrupamiento particional')
  st.write(
    """
    En esta sección se aplica agrupamiento particional mediante el algoritmo
    k-medias.
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
  st.write("""
  Agrupamiento trabaja sólo con datos numéricos.
  Se muestran las variables numéricas que no contienen datos nulos dentro del
  conjunto de datos cargado.
  """)
  # Select numeric variables with non empty values
  data = data.select_dtypes(exclude='object').dropna(axis='columns')
  st.write(data)
  st.markdown('El conjunto de datos contiene **'+str(data.shape[0])+
    '** registros y **'+str(data.shape[1])+'** columnas.')

  # Elbow method
  st.subheader('Método del codo')
  st.write(
    """
    El método del codo permite seleccionar una cantidad de grupos óptima.
    Selecciona un número máximo de grupos para aplicar el método del codo.
    """
    )
  kMax = st.number_input(label='Selecciona el máximo de grupos', min_value=3, 
    max_value=100, value=10, step=1
  )
  kMax = int(kMax)
  
  SSE = getSSE(data,kMax)

  k1 = KneeLocator(range(2,kMax),SSE,curve="convex",direction="decreasing")
  st.markdown(
    'De forma programática, el número óptimo de grupos es '+'**'+str(k1.elbow)
    +'**'
  )
  
  fig1 = go.Figure(
    data=go.Scatter(x=np.arange(start=2,stop=kMax),y=SSE)
  )
  fig1.add_vline(x=k1.elbow, line_dash="dash", line_color="green")
  fig1.update_layout(
    xaxis_title="k",
    yaxis_title="SSE",
    title_text="Método del codo"
  )
  st.plotly_chart(fig1, use_container_width=True)

  # k selection
  st.subheader('Selección del número de grupos')
  kSelected = st.number_input(label='Selecciona el número de grupos (k)', min_value=2, 
    max_value=kMax, value=k1.elbow, step=1
  )

  st.subheader('Obtención de los grupos')
  kMeansModel = KMeans(n_clusters=kSelected,random_state=0).fit(data)
  kMeansModel.predict(data)
  dataCluster = data.copy()
  dataCluster['Cluster'] = kMeansModel.labels_
  st.write(dataCluster)

  st.subheader('Centroides')
  centroids = pd.DataFrame(kMeansModel.cluster_centers_,
    columns=data.columns
  )
  st.write(centroids)

  st.subheader('Gráfica de los grupos generados')
  st.write('Para la generación de la gráfica de los grupos se utiliza PCA')
  scaler = StandardScaler()
  scaler.fit(data)
  dataStandarized = scaler.transform(data)
  pca = PCA(n_components=3)
  pca.fit(dataStandarized)
  dataReduced = pd.DataFrame(
    pca.transform(dataStandarized),
    columns=['PCA%s' % _ for _ in range(3)]
  )
  dataReduced['Cluster'] = kMeansModel.labels_
  fig2 = px.scatter(dataReduced, x="PCA0", y="PCA1", color="Cluster",
    color_continuous_scale="jet",title="Agrupamiento en 2D"
  )
  st.plotly_chart(fig2, use_container_width=True)

  fig3 = px.scatter_3d(dataReduced, x="PCA0", y="PCA1", z="PCA2",  color="Cluster",
    color_continuous_scale="jet",title="Agrupamiento en 3D"
  )
  st.plotly_chart(fig3, use_container_width=True)
