import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
import sklearn.metrics as mt

if st.button("Cargar el archivo de datos para entrenar el modelo"):

  uploaded_file = st.file_uploader("Seleccionar el archivo de Excel con los datos")
  if uploaded_file is not None:
      if 'cemento' not in st.session_state:
        st.session_state['cemento'] = pd.read_excel(uploaded_file, sheet_name="DATOS 2")
        st.session_state['cemento'].columns = st.session_state['cemento'].columns.str.strip()
        st.session_state['cemento']["Tipo de Cemento"] = st.session_state['cemento']["Tipo de Cemento"].str.strip()
        st.session_state['cemento']['Molino'] = st.session_state['cemento']['Molino'].str.strip
        st.dataframe(st.session_state.cemento)

        if 'cemGUM1' not in st.session_state:    
          st.session_state['cemGUM1'] = st.session_state.cemento[(st.session_state.cemento['Tipo de Cemento']=="Cemento GU") & (st.session_state.cemento['Molino']=="Molino 1")]
        if 'cemGUM2' not in st.session_state:
          st.session_state['cemGUM2'] = st.session_state.cemento[(st.session_state.cemento['Tipo de Cemento']=="Cemento GU") & (st.session_state.cemento['Molino']=="Molino2")]
        if 'cemHEM1' not in st.session_state:
          st.session_state['cemHEM1'] = st.session_state.cemento[(st.session_state.cemento['Tipo de Cemento']=="Cemento HE") & (st.session_state.cemento['Molino']=="Molino 1")]
        if 'cemHEM2' not in st.session_state:
          st.session_state['cemHEM2'] = st.session_state.cemento[(st.session_state.cemento['Tipo de Cemento']=="Cemento HE") & (st.session_state.cemento['Molino']=="Molino2")]

if st.button("Visualizar los Boxplots de la Resistencia"):
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(10,6)
    axs[0,0].boxplot(st.session_state['cemGUM1']['R1D'])
    axs[0,0].set_title("1 dia")
    axs[0,1].boxplot(st.session_state['cemGUM1']['R3D'])
    axs[0,1].set_title("3 dias")
    axs[1,0].boxplot(st.session_state['cemGUM1']['R7D'])
    axs[1,0].set_title("7 dias")
    axs[1,1].boxplot(st.session_state['cemGUM1']['R28D'])
    axs[1,1].set_title("28 dias")
    st.pyplot(fig)

if st.button("Entrenar el modelo"):
  etapar = 0.08
  lambdapar = 5
    
  X = st.session_state['cemGUM1'].drop(['Fecha','Tipo de Cemento','Molino','R1D','R3D','R7D','R28D'], axis=1)
  y = st.session_state['cemGUM1']['R1D']
  X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1)
  modeloXGB = XGBRegressor(booster='gblinear', eta=etapar, reg_lambda=lambdapar)
  modeloXGB.fit(X_train, y_train)
  pred_test =  modeloXGB.predict(X_test)
  
  st.write(mt.mean_absolute_percentage_error(y_test, pred_test))

if st.button("Descargar los datos"):
  datospred = pd.DataFrame({'Real':np.array(y_test), 'Pred':pred_test})
  buffer = io.BytesIO()
  
  with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
    datospred.to_excel(writer, sheet_name="prueba", index=False)
    writer.close()

    download2 = st.download_button(
        label="Download data as Excel",
        data=buffer,
        file_name='prueba.xlsx',
        mime='application/vnd.ms-excel'
    )
    
