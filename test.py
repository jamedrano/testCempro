import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io

from xgboost import XGBRegressor
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
import sklearn.metrics as mt


uploaded_file = st.file_uploader("Choose a file")



if uploaded_file is not None:
  cemento = pd.read_excel(uploaded_file, sheet_name="DATOS 2")

  cemento.columns = cemento.columns.str.strip()
  cemento["Tipo de Cemento"] = cemento["Tipo de Cemento"].str.strip()
  cemento['Molino'] = cemento['Molino'].str.strip()
    
  cemGUM1 = cemento[(cemento['Tipo de Cemento']=="Cemento GU") & (cemento['Molino']=="Molino 1")]
  cemGUM2 = cemento[(cemento['Tipo de Cemento']=="Cemento GU") & (cemento['Molino']=="Molino2")]
  cemHEM1 = cemento[(cemento['Tipo de Cemento']=="Cemento HE") & (cemento['Molino']=="Molino 1")]
  cemHEM2 = cemento[(cemento['Tipo de Cemento']=="Cemento HE") & (cemento['Molino']=="Molino2")]
  
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
    
  etapar = 0.08
  lambdapar = 5
    
  X = cemGUM1.drop(['Fecha','Tipo de Cemento','Molino','R1D','R3D','R7D','R28D'], axis=1)
  y = cemGUM1['R1D']
  X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1)
  modeloXGB = XGBRegressor(booster='gblinear', eta=etapar, reg_lambda=lambdapar)
  modeloXGB.fit(X_train, y_train)
  pred_test =  modeloXGB.predict(X_test)
  
  st.write(mt.mean_absolute_percentage_error(y_test, pred_test))
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
    
