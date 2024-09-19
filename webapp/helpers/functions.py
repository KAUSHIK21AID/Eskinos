import pandas as pd
import statsmodels.tsa.arima.model as sm
import statsmodels.api as sp
from sklearn.metrics import r2_score
import numpy as np
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# conn = st.connection('mysql', type='sql')

# tab = pd.read_csv("Files/GFR_15_vs_Meds.csv")
# tot = conn.query('SELECT * from final_final_v1;', ttl=600)
# df = pd.read_csv("C:/Users/vvhem/Downloads/FINAL_FINAL.csv")

# IDs = df["id"].unique()

# df["hospital_visited_date"] = pd.to_datetime(df["hospital_visited_date"],format="%d-%m-%Y")

def groupbyFunc(df,Id):
    return df.groupby("id").get_group(Id).reset_index()
 
def SARIMA(df):
    input_columns = ['serum_creatinine_level', 'total_count', 'random_blood_sugar', 'urea',
                    'haemoglobin_level', 'bmi', 'diabetes_encoded', 'living_area_encoded', 'worker_encoded', 'diet_encoded']
    forecast_column = 'glomerular_filtration'
    if len(df) < 12:
        warnings.warn(f"Insufficient data for a 12-month forecast. Only {len(df)} data points available.")
        if len(df) > 0:
            model = sm.ARIMA(df[forecast_column], order=(1,1,1))
            model_fit = model.fit()
            n = len(df)
            forecast = model_fit.get_forecast(steps=n)
            gfr_values_next_n_months = forecast.predicted_mean
        return pd.DataFrame(list(zip([i for i in range(1,n+1)],gfr_values_next_n_months)),columns=["Months","Predicted eGFR"])
    else:
        model = sm.ARIMA(df[forecast_column], exog=df[input_columns], order=(1,1,1))
        model_fit = model.fit()
        # df["counts"] = [i for i in range(1,len(df)+4)]
        exog_forecast = df[input_columns].tail(len(df))
        forecast = model_fit.get_forecast(steps=len(df), exog=exog_forecast)
        gfr_values_next_12_months = forecast.predicted_mean
        return pd.DataFrame(list(zip([i for i in range(1,len(df))],gfr_values_next_12_months)),columns=["Months","Predicted eGFR"])


def ARIMA(df):
    model = sm.ARIMA(df["glomerular_filtration"],order=(1,1,1)) 
    model1 = model.fit()
    df["Analysis"] = round(model1.predict(0,len(df)),2)
    def DepthAri(df):
        df["FForcast"] = round(model1.predict(len(df)-5,len(df)-1))
        return df   
    return df

def SARIMAX(df):
    model = sp.tsa.statespace.SARIMAX(df["glomerular_filtration"])
    model2 = model.fit()
    df["Analysis"] = round(model2.predict(0,len(df)),2)
    return df

def Aligner(df):
    df["Analysis"] = df["Analysis"].shift(-1)
    # df.iloc[len(df)-1,df.columns.get_loc("Analysis")] = df.iloc[len(df)-1,df.columns.get_loc("glomerular_filtration")]
    return df

def Accuracy(df):
    return round(r2_score(df["glomerular_filtration"],df["Analysis"]) * 100 , 2)

# def Graph(df):
#     plt.fig(figsize=(8,8))
#     plt.plot(df["Hosiptal Visited Date"],df[""])
#      res = df[[""]].plot(kind='bar',figsize=(20, 16), fontsize=26).get_figure()
#     return df[["glomerular_filtration","Analysis"]].plot()

def Graph(df):
    fig,ax = plt.subplots()
    # print(df)
    df["Forecast"] =df.iloc[len(df)-4:]["Analysis"]
    df["yr"] = [i for i in range(1,len(df)+1)]
    ax.plot(df["yr"],df["Analysis"],label="Analysis")
    # ax.legend("Analysis",loc="upper left")
    # ax.set_label("Analysis")
    ax.plot(df["yr"],df["Forecast"],label="Forecast")
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("eGFR")
    ax.set_title("Variation of GFR by Year")
    return fig
# def Graph(df):
#     df["Forecast"] =df.iloc[len(df)-4:]["Analysis"]
#     res = df[["Analysis","Forecast"]].plot(title = "Analysis and Forecasting of eGFR with Time(years)")
#     res.set_xlabel("Year")
#     res.set_ylabel("eGFR")
#     return res.get_figure().savefig("sample.png")
#     return df[["Analysis","f2"]].plot().get_figure().savefig("sample.png")
    

def OffsetCreator(df,n):
    future=[]
    df1 = df.resample("y",on="hospital_visited_date").mean().dropna().reset_index()
    last_year = int(df1["hospital_visited_date"].tail(1).dt.year)
    for i in range(last_year+1,int(last_year)+n,1):
        future.append(df1.tail(1)["hospital_visited_date"] + DateOffset(year=i))
    # print(future)
    id = [df1["id"][0] for i in range(len(future))]
    t=[np.nan for i in range(len(future))]
    w=[np.nan for i in range(len(future))]
    future_datest_df=pd.DataFrame(list(zip(id,future,t)),
                                columns=["id","hospital_visited_date","glomerular_filtration"])
    return pd.concat([df1,future_datest_df]).reset_index()

def Resampler(df):
    return round(df.resample("y",on="hospital_visited_date").mean(),0).dropna().reset_index()

def ret(given):
  nnn = given.split(',')
  medication = []
  for h in nnn:
    h = h.replace('"','')
    medication.append(h)
  return medication

