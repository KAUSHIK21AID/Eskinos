import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Pipeline
st.set_page_config(page_title="Graph Analysis",layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
conn = st.connection('mysql', type='sql')

# data= conn.query('SELECT * from ckdfinal_alldatap_reprocessed_noiseremoved',ttl=600)


bmis = conn.query('SELECT * from finaliti;', ttl=0)
bmis = Pipeline.AddingAttributes(bmis)
# st.write(bmis[bmis['id'] == 800])
data = bmis

@st.cache_data
def line_sys(given,n):
    y = np.array(given)
    x = list(range(1, n+1))
    fig,ax = plt.subplots()
    ax.plot(x,y)
    ax.set_xlabel("Number of Visit")
    # ax.xlabel("Number of Visit") 
    ax.set_ylabel("Systolic Pressure")  
    ax.set_title("Deviation in Systolic pressure") 
    st.pyplot(fig=fig)
    #   st.pyplot(x,y)
    
@st.cache_data
def line_dias(given,n):
        y = np.array(given)
        x = list(range(1, n+1))
        #   st.pyplot(x,y)
        fig,ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("Number of Visit")  # add X-axis label
        ax.set_ylabel("Diastolic Pressure")  # add Y-axis label
        ax.set_title("Deviaion in Diastolic pressure")  # add title
        st.pyplot(fig)

@st.cache_data
def bmi(df):
    fig,ax= plt.subplots()
    ax.bar(df["hospital_visited_date"],df["bmi"])
    num = np.arange(1,len(df["hospital_visited_date"])+1,1)
    ax.set_xticks(np.arange(0,len(df["hospital_visited_date"]),1),num)
    ax.set_xlabel("Visits")  # add X-axis label
    ax.set_ylabel("bmi")  # add Y-axis label
    ax.set_title("bmi Respect To Time")  # add title
    st.pyplot(fig=fig)

@st.cache_data
def creatine(df):
    fig,ax = plt.subplots()
    ax.bar(df["hospital_visited_date"],df["serum_creatinine_level"])
    num = np.arange(1,len(df["hospital_visited_date"])+1,1)
    ax.set_xticks(np.arange(0,len(df["hospital_visited_date"]),1),num)
    ax.set_xlabel("Visits")  # add X-axis label
    ax.set_ylabel("Creatinine")  # add Y-axis label
    ax.set_title("Creatinine Respect To Time")  # add title
    st.pyplot(fig)

st.markdown("<h2 style='text-align:center;colour:white;'><u>Health Metrics Graphical Analysis: Visualizing Diabetes, BMI, and Creatinine</u></h2>",unsafe_allow_html=True)

pid = data["id"].unique()
col1,col2 = st.columns([2,8])
with col1:
    i = st.selectbox("Patient ID",pid)
with col2:
    st.info("This page presents a graphical representation derived from patient data, illustrating the concurrent variations in BP,bmi,Creatinine.")
df1 = data[data['id']==i]
bps_ = list(df1['bp'])
sys = []
dias = []
d_sys = []
d_dias = []
# print(bps)
for j in bps_:
  k = j.split('/')
  sys.append(int(k[0]))
  dias.append(int(k[1]))
# print("PATIENT ",i)
# print(bps_)
# print(sys)
for j in range(0,len(sys)-1):
  d_sys.append(sys[j] - sys[j+1])
# print(dias)
for j in range(0,len(dias)-1):
  d_dias.append(dias[j] - dias[j+1])
# line_sys(d_sys,len(bps_)-1)
# print()
# line_dias(d_dias,len(bps_)-1)
# print()
# print()
col1,col2 = st.columns([5,5])
with col1:
    (line_sys(d_sys,len(bps_)-1))
    bmi(bmis.groupby("id").get_group(i))
with col2:
    (line_dias(d_dias,len(bps_)-1))
    creatine(bmis.groupby("id").get_group(i))
