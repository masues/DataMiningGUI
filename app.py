import streamlit as st

# Custom imports
from multipage import MultiPage
from modules import upload_dataset,exploratory_data_analysis,logistic_predictor
from modules import principal_component_analysis,kmeans,logistic_regression

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title('Minado de datos ⛏️')

# Add all your application here
app.add_page("Carga de datos", upload_dataset.app)
app.add_page("EDA", exploratory_data_analysis.app)
app.add_page("PCA", principal_component_analysis.app)
app.add_page("k-medias",kmeans.app)
app.add_page("Entrenamiento del clasificador", logistic_regression.app)
app.add_page('Predicción de clases',logistic_predictor.app)

# The main app
app.run()
