import streamlit as st
# https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
#import numpy as np


# Custom imports
from multipage import MultiPage
from modules import upload_dataset, exploratory_data_analysis
from modules import principal_component_analysis

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title('Minado de datos ⛏️')

# Add all your application here
app.add_page("Carga de datos", upload_dataset.app)
app.add_page("EDA", exploratory_data_analysis.app)
app.add_page("PCA", principal_component_analysis.app)

# The main app
app.run()
