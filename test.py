import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io
import pygwalker as pyg
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb


from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

 

st.set_page_config(page_title='Modelo Predictivo Resistencia a la Compresión CEMPRO', page_icon=None, layout="wide")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Datos', 'Descripcion Datos', 'Graficos', 'Modelo', 'Descargar Datos'])

st.sidebar.write("****Cargar Archivo de Datos en Excel****")
uploaded_file = st.sidebar.file_uploader("*Upload file here*")

if uploaded_file is not None:
  sh = st.sidebar.selectbox("*Que hoja contiene los datos?*",pd.ExcelFile(uploaded_file).sheet_names)
  h = st.sidebar.number_input("*Que fila contiene los nombres de columnas?*",0,100)
  @st.cache_data(experimental_allow_widgets=True)
  def load_data(uploaded_file,sh,h):
    data = pd.read_excel(uploaded_file,header=h,sheet_name=sh,engine='openpyxl')
    data.columns = data.columns.str.strip()
    for col in data.columns:
      if data[col].dtype == 'O':
        data[col] = data[col].str.strip()    
    return data
  data = load_data(uploaded_file,sh,h)
 
  
  with tab1:
    st.write( '### 1. Datos Cargados ')
    st.dataframe(data, use_container_width=True)

  with tab2:
    st.write( '### 2. Descripción de los Datos ')
    selected = st.radio( "**B) Seleccione lo que desea ver de los datos?**", 
                                    ["Dimensiones",
                                     "Descripcion de las Variables",
                                    "Estadisticas Descriptivas", 
                                    "Tabulacion de Valores de las Columnas"])
   
    if selected == 'Descripcion de las Variables':
     fd = data.dtypes.reset_index().rename(columns={'index':'Field Name',0:'Field Type'}).sort_values(by='Field Type',ascending=False).reset_index(drop=True)
     st.dataframe(fd, use_container_width=True)
    
    elif selected == 'Estadisticas Descriptivas':
     ss = pd.DataFrame(data.describe(include='all').round(2).fillna(''))
     st.dataframe(ss, use_container_width=True)
    
    elif selected == 'Tabulacion de Valores de las Columnas':           
     sub_selected = st.radio( "*Columna a Investigar?*",data.select_dtypes('object').columns)
     vc = data[sub_selected].value_counts().reset_index().rename(columns={'count':'Count'}).reset_index(drop=True)
     st.dataframe(vc, use_container_width=True)
    
    else:
     st.write('###### Dimensiones de la Data :',data.shape)

  with tab3:
   molino = st.radio("** Seleccione Molino **", data['Molino'].unique())
   tipo = st.radio("** Seleccione Tipo de Cemento **", data['Tipo de Cemento'].unique())
   subdatos = data[(data['Tipo de Cemento']==tipo)&(data['Molino']==molino)]
   st.write( '### 3. Exploración Gráfica ')
   fig, axs = plt.subplots(2,2)
   fig.set_size_inches(10,6)
   axs[0,0].boxplot(subdatos['R1D'])
   axs[0,0].set_title("1 dia")
   axs[0,1].boxplot(subdatos['R3D'])
   axs[0,1].set_title("3 dias")
   axs[1,0].boxplot(subdatos['R7D'])
   axs[1,0].set_title("7 dias")
   axs[1,1].boxplot(subdatos['R28D'])
   axs[1,1].set_title("28 dias")
   st.pyplot(fig)

  with tab4:
   molino2 = st.radio("** Seleccione Molino a Modelar **", data['Molino'].unique())
   tipo2 = st.radio("** Seleccione Tipo de Cemento a Modelar **", data['Tipo de Cemento'].unique())

   edad =  st.radio("** Edad a Predecir **", ["1 dia", "3 dias", "7 dias", "28 dias"])
   
   subdatos2 = data[(data['Tipo de Cemento']==tipo2)&(data['Molino']==molino2)]
   
   def modelo(datos, quitar, respuesta):
    etapar = 0.08
    lambdapar = 5
    X = datos.drop(quitar, axis=1)
    y = datos[respuesta]
    modeloXGB = XGBRegressor(booster='gblinear', eta=etapar, reg_lambda=lambdapar)
    modeloXGB.fit(X, y)
    pred = modeloXGB.predict(X)
    return (X, y, pred)

   def desplegar():
    (X,y,pred) = modelo(subdatos2, quitar, respuesta)
    fig2, axs2 = plt.subplots()
    fig2.set_size_inches(6,6)
    axs2.scatter(y, pred)
    st.pyplot(fig2)

    st.write("Porcentaje de Error")
    st.write(mt.mean_absolute_percentage_error(y, pred))
    st.write("Coef. de Determinación")
    st.write(mt.r2_score(y,pred))
   
    datosprueba = pd.DataFrame({'ytest':y, 'pred':pred})
    st.dataframe(datosprueba)

    

   if edad == "1 dia":
    quitar = ['Fecha','Tipo de Cemento','Molino','R1D','R3D','R7D','R28D']
    respuesta = 'R1D'
    desplegar()
    
   if edad == "3 dias":
    quitar = ['Fecha','Tipo de Cemento','Molino','R3D','R7D','R28D']
    respuesta = 'R3D'
    desplegar()
    
   if edad == "7 dias":
    quitar = ['Fecha','Tipo de Cemento','Molino','R7D','R28D']
    respuesta = 'R7D'
    desplegar()
    
   
   if edad == "28 dias":
    quitar = ['Fecha','Tipo de Cemento','Molino','R7D','R28D']
    respuesta = 'R28D'
    desplegar()
   
  with tab5:
   
   def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data
    
   df_xlsx = to_excel(datosprueba)
   st.download_button(label='📥 Download Current Result',data=df_xlsx ,file_name= 'df_test.xlsx')
   

   
  
    
    
    
