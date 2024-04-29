import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
import pygwalker as pyg

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

#Setting up web app page
st.set_page_config(page_title='App de Prueba de Streamlit', page_icon=None, layout="wide")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Datos', 'Descripcion Datos', 'Graficos', 'Modelo', 'Descargar Datos'])

#Creating section in sidebar
st.sidebar.write("****A) File upload****")

#User prompt to select file type
ft = st.sidebar.selectbox("*What is the file type?*",["Excel", "csv"])

#Creating dynamic file upload option in sidebar
uploaded_file = st.sidebar.file_uploader("*Upload file here*")

if uploaded_file is not None:
    file_path = uploaded_file

    if ft == 'Excel':
        try:
            #User prompt to select sheet name in uploaded Excel
            sh = st.sidebar.selectbox("*Which sheet name in the file should be read?*",pd.ExcelFile(file_path).sheet_names)
            #User prompt to define row with column names if they aren't in the header row in the uploaded Excel
            h = st.sidebar.number_input("*Which row contains the column names?*",0,100)
        except:
            st.info("File is not recognised as an Excel file")
            sys.exit()
    
    elif ft == 'csv':
        try:
            #No need for sh and h for csv, set them to None
            sh = None
            h = None
        except:
            st.info("File is not recognised as a csv file.")
            sys.exit()

    #Caching function to load data
    @st.cache_data(experimental_allow_widgets=True)
    def load_data(file_path,ft,sh,h):
        
        if ft == 'Excel':
            try:
                #Reading the excel file
                data = pd.read_excel(file_path,header=h,sheet_name=sh,engine='openpyxl')
            except:
                st.infdo("File is not recognised as an Excel file.")
                sys.exit()
    
        elif ft == 'csv': 
            try:
                #Reading the csv file
                data = pd.read_csv(file_path)
            except:
                st.info("File is not recognised as a csv file.")
                sys.exit()
        
        return data

    data = load_data(file_path,ft,sh,h)

    @st.cache_data(experimental_allow_widgets=True)
    def preprocess(data):
        data.columns = data.columns.str.strip()
        data['Tipo de Cemento'] = data['Tipo de Cemento'].str.strip()
        data['Molino'] = data['Molino'].str.strip()      
        cemGUM1 = data[(data['Tipo de Cemento']=='Cemento GU')&(data['Molino']=='Molino 1')]
        cemGUM2 = data[(data['Tipo de Cemento']=='Cemento GU')&(data['Molino']=='Molino2')]
        cemHEM1 = data[(data['Tipo de Cemento']=='Cemento HE')&(data['Molino']=='Molino 1')]
        cemHEM2 = data[(data['Tipo de Cemento']=='Cemento HE')&(data['Molino']=='Molino2')]
        datasets = [cemGUM1, cemGUM2, cemHEM1, cemHEM2]
        return datasets

    cemGUM1, cemGUM2, cemHEM1, cemHEM2 = preprocess(data)
#=====================================================================================================
## 1. Overview of the data
    with tab1:
        st.write( '### 1. Dataset Preview ')
    
        try:
          #View the dataframe in streamlit
          st.dataframe(data, use_container_width=True)
    
        except:
          st.info("The file wasn't read properly. Please ensure that the input parameters are correctly defined.")
          sys.exit()

## 2. Understanding the data
    with tab2:
        st.write( '### 2. High-Level Overview ')
    
        #Creating radio button and sidebar simulataneously
        selected = st.sidebar.radio( "**B) What would you like to know about the data?**", 
                                    ["Data Dimensions",
                                     "Field Descriptions",
                                    "Summary Statistics", 
                                    "Value Counts of Fields"])
    
        #Showing field types
        if selected == 'Field Descriptions':
            fd = data.dtypes.reset_index().rename(columns={'index':'Field Name',0:'Field Type'}).sort_values(by='Field Type',ascending=False).reset_index(drop=True)
            st.dataframe(fd, use_container_width=True)
    
        #Showing summary statistics
        elif selected == 'Summary Statistics':
            ss = pd.DataFrame(data.describe(include='all').round(2).fillna(''))
            st.dataframe(ss, use_container_width=True)
    
        #Showing value counts of object fields
        elif selected == 'Value Counts of Fields':
            # creating radio button and sidebar simulataneously if this main selection is made
            sub_selected = st.sidebar.radio( "*Which field should be investigated?*",data.select_dtypes('object').columns)
            vc = data[sub_selected].value_counts().reset_index().rename(columns={'count':'Count'}).reset_index(drop=True)
            st.dataframe(vc, use_container_width=True)
    
        #Showing the shape of the dataframe
        else:
            st.write('###### The data has the dimensions :',data.shape)

#=====================================================================================================
## 3. Visualisation
    with tab3:
        
        #Selecting whether visualisation is required
        vis_select = st.sidebar.checkbox("**C) Is visualisation required for this dataset?**")
    
        if vis_select:
    
            st.write( '### 3. Visual Insights ')
            fig, axs = plt.subplots(2,2)
            fig.set_size_inches(10,6)
            axs[0,0].boxplot(cemGUM1['R1D'])
            axs[0,0].set_title("1 dia")
            axs[0,1].boxplot(cemGUM1['R3D'])
            axs[0,1].set_title("3 dias")
            axs[1,0].boxplot(cemGUM1['R7D'])
            axs[1,0].set_title("7 dias")
            axs[1,1].boxplot(cemGUM1['R28D'])
            axs[1,1].set_title("28 dias")
            st.pyplot(fig)

#=====================================================================================================
## 3. Visualisation
    with tab4:
        etapar = 0.08
        lambdapar = 5
        X = cemGUM1.drop(['Fecha','Tipo de Cemento','Molino','R1D','R3D','R7D','R28D'], axis=1)
        y = cemGUM1['R1D']
        X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1)
        modeloXGB = XGBRegressor(booster='gblinear', eta=etapar, reg_lambda=lambdapar)
        modeloXGB.fit(X_train, y_train)
        pred_test =  modeloXGB.predict(X_test)
        st.write(mt.mean_absolute_percentage_error(y_test, pred_test))
        fig2, axs2 = plt.subplots()
        fig2.set_size_inches(4,4)
        axs2.scatter(y_test, pred_test)
        st.pyplot(fig2)
        
            
            
