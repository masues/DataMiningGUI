import streamlit as st
# https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
#import numpy as np


# Custom imports
from multipage import MultiPage
from modules import upload_dataset, exploratory_data_analysis

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title('Minado de datos ⛏️')

# Add all your application here
app.add_page("Carga de datos", upload_dataset.app)
app.add_page("Análisis exploratorio de datos", exploratory_data_analysis.app)
#app.add_page("Change Metadata", metadata.app)
#app.add_page("Machine Learning", machine_learning.app)
#app.add_page("Data Analysis",data_visualize.app)
#app.add_page("Y-Parameter Optimization",redundant.app)

# The main app
app.run()
