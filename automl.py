import pandas as pd
import pycaret.regression
import streamlit as st 
import os,uuid 
import pycaret
#--------------Profiling-------------
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report
#--------ML-------------
from pycaret.classification import setup as setupc, compare_models as compare_modelsc, save_model as save_modelc, load_model as load_modelc, evaluate_model as evaluate_modelc

from pycaret.regression import setup as setupr, compare_models as compare_modelsr, pull, save_model as save_modelr, load_model as load_modelr, evaluate_model as evaluate_modelr

#-----------------page confirguration-------------------
st.set_page_config(page_title='AutoML', layout='wide')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Adding background image

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background-image: url({"https://images.unsplash.com/photo-1483794344563-d27a8d18014e"});
  background-size: 180%;
  background-position: center; /* Center the image */
  background-repeat: no-repeat;
  background-attachment: local;
  /* Apply a darkening overlay */
  background-color: rgba(0, 0, 0, 0.5);
}}

[data-testid="stHeader"] {{
  background: rgba(0, 0, 0, 0);
}}

[data-testid="stToolbar"] {{
  right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown('# :green[Automated] Machine Learning')
st.write("The **Automated Machine Learning System for Predictive Analytics** is designed to streamline the entire process of creating and deploying predictive models. This system leverages advanced algorithms and machine learning techniques to automate tasks such as data preprocessing, feature selection, model training, and evaluation. By minimizing the need for human intervention, it allows data scientists and analysts to focus on more strategic aspects of their work, such as interpreting results and making data-driven decisions. The system ensures consistency, reduces the potential for human error, and accelerates the time-to-insight, making it an invaluable tool for organizations looking to harness the power of predictive analytics efficiently and effectively.")

with st.sidebar:
  st.markdown('# **Auto** ML')
  st.write('Develop a :green[tailored machine learning model] fine-tuned to your specific data needs.')
  st.divider()
  
  with st.container(border=True):
    select_box = st.radio(label=' ', options=['Upload Data','Profiling','Edit Data','Model Creation','Model Download'], captions=['Make sure upload your feature engineered Data','Get Info about the upload data','Edit the data and drop uncessary Features','Automated model creation and evaluation','Final Model download'])
    
    
#----------functionality--------------

#-------------------------------------------------------------------------------
#--------------------------------Upload Data------------------------------------
#-------------------------------------------------------------------------------

if select_box=='Upload Data':
  c1,_ = st.columns([0.5,0.5])
  with c1:
    st.write('### Upload the Data')
    data = st.file_uploader(label='Upload the Data',accept_multiple_files=False, help='Upload')
  if data:
    df = pd.read_csv(data)
    df.to_csv('uploaded_data.csv',index=False)
    st.dataframe(df, use_container_width=True,hide_index=True)

if os.path.exists('uploaded_data.csv'):
  df = pd.read_csv('uploaded_data.csv')
  
#-------------------------------------------------------------------------------
#--------------------------------Profiling------------------------------------
#-------------------------------------------------------------------------------

if select_box=='Profiling':
  st.write("### Exploratory Data Analysis")
  pr = ProfileReport(df, minimal=True, orange_mode=True, explorative=True)
  st_profile_report(pr, navbar=True)
  
#-------------------------------------------------------------------------------
#--------------------------------Edit Data--------------------------------------
#-------------------------------------------------------------------------------
  
if select_box=='Edit Data':
  st.write("### Edit Data")
  c1,_ = st.columns([0.5,0.5])
  with c1:
    dropcol = st.multiselect(label='Select columns to drop', options=df.columns)
  button = st.button(label='Drop selected column', type='primary')
  if button:
    editdf = df.drop(dropcol,axis=1)
    editdf.to_csv('editdf.csv',index=False)
    st.success('Selected columns dropped and saved to editdf.csv')
    st.dataframe(editdf)
    
if os.path.exists('editdf.csv'):
  finaldf = pd.read_csv('editdf.csv')
elif os.path.exists('uploaded_data.csv'):
  finaldf = pd.read_csv('uploaded_data.csv')
  
#-------------------------------------------------------------------------------
#--------------------------------Model Creation---------------------------------
#-------------------------------------------------------------------------------

final_model = None
  
if select_box=='Model Creation':
  st.write("### Model Creation")
  c1,c2 = st.columns([0.5,0.5])
  with c1:
    target_button = st.selectbox('Select the Target variable', options=finaldf.columns, placeholder='Choose the Variable')
    run = st.button(label='Run')
  with c2:
    model_= st.toggle(label='Choose the Problem', help='Choose the Model basis on your problem. If Target variable is continous select Regression and For Discrete or binary variable select classification')
    if model_ is True:
      st.write('**Classification**', unsafe_allow_html=True)
    else:
      st.write('**Regression**', unsafe_allow_html=True)
      
#-------------------------------------------------------------------------------
#--------------------------------classification---------------------------------
#-------------------------------------------------------------------------------

  if model_ is True and run is True:
    st.info('Info')
    s_clf = setupc(finaldf, target=target_button, verbose=False)
    result = pycaret.classification.pull()
    st.write('##### Info', unsafe_allow_html=True)
    st.write(result)
    
    st.write('##### All Models', unsafe_allow_html=True)
    st.write('All trained models with metrics.')
    best_model_c= compare_modelsc()
    result_best = pycaret.classification.pull()
    st.write(result_best)
    final_model = compare_modelsc()
    
#-------------------------------------------------------------------------------
#--------------------------------Regression-------------------------------------
#-------------------------------------------------------------------------------

  elif model_ is False and run is True:
    st.info('Info')
    s_reg = setupr(finaldf, target=target_button, verbose=False)
    result = pycaret.regression.pull()
    st.write('##### Info', unsafe_allow_html=True)
    st.write(result)
    
    st.write('##### All Models', unsafe_allow_html=True)
    st.write('All trained models with metrics.')
    best_model_r= compare_modelsr()
    result_best = pycaret.regression.pull()
    st.write(result_best)
    final_model = compare_modelsr()

#-------------------------------------------------------------------------------
#--------------------------------Download---------------------------------------
#------------------------------------------------------------------------------- 

if select_box=='Model Download':
  down = st.button(label='Download the Model',type='primary')
  if down:
    save_modelc(final_model,f'models//{uuid.uuid1()}')
    st.success('your model is saved in Models folder')