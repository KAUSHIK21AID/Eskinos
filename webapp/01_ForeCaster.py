import statsmodels.api as sm
import statsmodels.tsa.arima.model as sa
import statsmodels.tsa.statespace.sarimax as ss
import statsmodels
import numpy as np
import logging
import warnings
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys

from helpers.Pipeline import AddingAttributes
sys.path.append("helpers/")

from functions import *
from Pipeline import *
st.set_page_config(page_title="Forecaster",layout="wide")
#st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style='text-align:center;colour:white;'>Prediction of Dialysis in Patients with Early Kidney Disease</h1>",unsafe_allow_html=True)

# tab2,tab3 = st.tabs(["F2","Forcaster III"])
# tab3 = st.tabs(["Forecaster"])
# tab1.subheader("ARIMA Type")

# Define a custom Streamlit logger
logger = logging.getLogger("streamlit")

# Add a file handler to log to a file
file_handler = logging.FileHandler("streamlit.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

#Setup-connection-mysql
conn = st.connection('mysql', type='sql')

#For ARIMA-Multivariate
#picking the data from table
df_bh = conn.query('SELECT * from finaliti;', ttl=600)
# st.write(df_bh.head(5))
df_bh = AddingAttributes(df_bh)

#preprocessing
df_bh['hospital_visited_date'] = df_bh['hospital_visited_date'].apply(pd.Timestamp)
df_bh = df_bh.sort_values(by = 'hospital_visited_date')

# Suppressing warnings
warnings.filterwarnings("ignore")

def forecast_next_years_arima(patient_data, forecast_column, p, d, q, years=3):
    try:
        model = sa.ARIMA(patient_data[forecast_column], order=(p, d, q))
        model_fit = model.fit()

        # Forecast the next 3 years steps
        forecast = model_fit.get_forecast(steps=years)
        forecast_values = forecast.predicted_mean.values
        # logger.info(f"Sequential Forecasting for the patient {patient_data['id']} is sucessful.")
        # logger.info("Returning the values to make it visualized...")

        return forecast_values

    except Exception as e:
        logger.error(f'Error occured at the forecasting_next_years_arima module named{e}')
        st.write(f"Error: {e}")
        return None

def forecast_next_three_years_arima(patient_data, forecast_column, p, d, q):
    if len(patient_data) < 5:
        logger.error(f"The patient record for {patient_data['id'].iloc[0]} contains only {len(patient_data)} records. Insufficient data.")
        # Insufficient data
        raise Exception(f"The patient record for {patient_data['id'].iloc[0]} contains only {len(patient_data)} records. Insufficient data.")

    # Forecast the next three years using the ARIMA model
    forecast_values = forecast_next_years_arima(patient_data, forecast_column, p, d, q, years=3)

    return forecast_values

def tune_sarima_parameters(data, input_columns, forecast_column):
    p, d, q = 1, 1, 1
    best_s = 0
    best_aic = np.inf

    for s in range(2, 13):
        model = ss.SARIMAX(data[forecast_column], order=(p, d, q), seasonal_order=(0, 0, 0, s))
        model_fit = model.fit(disp=False)
        aic = model_fit.aic

        if aic < best_aic:
            best_aic = aic
            best_s = s

    return best_s

def forecast_next_years(patient_data, input_columns, forecast_column, years=3):
    p, d, q = 1, 1, 1

    s = tune_sarima_parameters(patient_data, input_columns, forecast_column)
    # logger.info("S-ARIMA tuned with the data provided")
    # st.write(s)

    try:
        model = ss.SARIMAX(patient_data[forecast_column], order=(p, d, q), seasonal_order=(0, 0, 0, s))
        model_fit = model.fit(disp=False)

        exog_mean_by_year = patient_data[input_columns].resample('Y').mean() # Calculating the mean for each year
        # st.write(exog_mean_by_year)
        exog_forecast = exog_mean_by_year.loc[exog_mean_by_year.index.max()]
        exog_forecast = pd.concat([exog_forecast] * years, ignore_index=True)

        forecast = model_fit.get_forecast(steps=years, exog=exog_forecast) # Forecasting for next 3 years
        gfr_values_next_years = forecast.predicted_mean.values
        # logger.info(f"The GFR has been forecasted for the three suceesfully for the patient {patient_data['id']}")
        return gfr_values_next_years

    except Exception as e:
        logger.error(f"Error araised from the module forecast_next_year , Exception named {e}")
        st.write(f"Error: {e}")
        return None
    
def forecast_next_three_years_sari(patient_data, input_columns, forecast_column):
    patient_data.index = pd.to_datetime(patient_data.index)
    # st.write(patient_data)

    if len(patient_data) < 5:
        logger.warning(f"The patient record for {patient_data['id'].iloc[0]} contains only {len(patient_data)} records. Insufficient data.")
        raise Exception(f"The patient record for {patient_data['id'].iloc[0]} contains only {len(patient_data)} records. Insufficient data.")

    # logger.info(f'Patient {patient_data["id"]} has adequent data to process further.. Sending the data to forecast_next_years.....')
    forecast_values = forecast_next_years(patient_data, input_columns, forecast_column, years=3)


    return forecast_values

# Example usage
# patient_name = 303

#FOR ARIMA
df = conn.query('SELECT * from finaliti',ttl=200)

#SARIMA-TB3
sari = (Interpolating(AddingVisits(Staging(AddingAttributes(df)))))
sari['hospital_visited_date'] = sari["hospital_visited_date"].apply(pd.Timestamp)
sari = sari.sort_values(by = 'hospital_visited_date')
sari = sari.sort_values(by=['id', 'hospital_visited_date'], ascending=[True, True])
sari = sari[['id', 'hospital_visited_date','diabetes','glomerular_filration','serum_creatinine_level','haemoglobin_level','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi', 'total_water_intake']]
label_encoder = LabelEncoder()
categorical_columns = ['diabetes', 'living_area', 'worker', 'diet']
for column in categorical_columns:
    sari[column + '_encoded'] = label_encoder.fit_transform(sari[column])
sari = sari.drop(['diabetes','living_area','worker','diet'], axis = 1)


df["hospital_visited_date"] = df['hospital_visited_date'].apply(pd.Timestamp)
# st.write(df)
# st.write(df.dtypes)
# df.drop(df[df.columns[0:7]],axis=1,inplace=True)
df["id"] = df["id"].astype(int)


# with tab2:
#     tab2.subheader("ARIMA MULTIVARIATE TYPE")
#     # st.write(AddingVisits(df=df))
#     # st.write(df[df['id']==307])
#     col1,col2 = st.columns([2,4])

#     with col1:
#         # st.write(df.head(10))
#         grpId = st.selectbox("PID",df["id"].unique())
#         st.write("YearWise Details Of Patients:")
#         df1 = df[["id","hospital_visited_date","glomerular_filration"]]
#         df2 = df1
#         df2["glomerular_filration"] = df2["glomerular_filration"].astype(float)
#         df2 = df2.groupby("id").get_group(grpId).resample("y",on="hospital_visited_date").mean().dropna().reset_index()
#         df2["glomerular_filration"] = round(df2["glomerular_filration"],0)
#         lastrecord = df2[["hospital_visited_date","glomerular_filration"]]
#         st.write(lastrecord)
    
#     with col2:
#         patient_data = df[df['id'] == grpId].dropna()

#         forecast_column = 'glomerular_filration'

#         # Specify ARIMA order (p, d, q)
#         p, d, q = 1, 1, 1

#         try:
#             forecast_values = forecast_next_three_years_arima(patient_data, forecast_column, p, d, q)
#             forecast_values = np.insert(forecast_values,0,lastrecord["glomerular_filration"][len(lastrecord)-1],axis=0)
#             fv = pd.DataFrame(forecast_values,columns=["glomerular_filration"])
#             temp = pd.concat([lastrecord[["glomerular_filration"]],fv]).reset_index()
#             # temp["0"] =temp.iloc[len(df)-4:]["glomerular_filration"]
#             temp["Forecasted"] = temp.iloc[len(temp)-4:]["glomerular_filration"]
#             temp["yr"] = [i for i in range(len(temp))]
#             # st.write(temp)
#             fig,ax = plt.subplots()
#             ax.plot(temp["yr"],temp["glomerular_filration"],label="Analysis")
#             ax.plot(temp["yr"],temp["Forecasted"],label="Forecasted")
#             ax.legend()
#             ax.set_xlabel("Year")
#             ax.set_ylabel("eGFR")

#             # res = temp[["glomerular_filration","Forecasted"]].plot(title = "Analysis and Forecasting of eGFR with Time(years)")
#             st.pyplot(fig)
#             if forecast_values is not None:
#                 st.write(f'Mean GFR values for PID-{grpId} for the Next 3 Years (ARIMA):')
#                 for year in range(1, 4):
#                     st.write(f'GFR value for Year {year}: {round(forecast_values[year],3)}')
#                 sum = (forecast_values[2] - forecast_values[3])
#                 measure = forecast_values[2] *(3.3/100)
#                 st.write("The change of difference of GFR ",round(sum,5))
#                 st.write("The 3.3% of their last Year 3 is ",round(measure,2))
#                 if(sum>measure or forecast_values[3]<25):
#                    st.write("Patient is in Risk Condition")
#                 else:
#                    st.write("Patient is not in Risk Condition") 
                
#             else:
#                 st.write("Forecasting failed.")
#         except Exception as e:
#             st.write(str(e))

# with tab3:
st.subheader("S-ARIMA")
col1,col2 = st.columns([2,4])
# st.write(df.head(5))
# st.write(AddingAttributes(df[df['id'] == 800]))
with col1:
    grpIdS = st.selectbox("Patient-ID",sari["id"].unique().astype(int))
    st.write("YearWise Details Of Patients:")
    df2 = sari
    df2["glomerular_filration"] = df2["glomerular_filration"].astype(float)
    df2 = df2.groupby("id").get_group(grpIdS).resample("y",on="hospital_visited_date").mean().dropna().reset_index()
    df2["hospital_visited_date"] = df2["hospital_visited_date"].dt.year
    df2["hospital_visited_date"] = df2["hospital_visited_date"].astype(str)
    df2["glomerular_filration"] = round(df2["glomerular_filration"],0)
    lastrecord = df2[["hospital_visited_date","glomerular_filration"]]
    st.dataframe(lastrecord,width=800,height=650)
    logger.info(f'Processing the forecasting for the patient {grpIdS}')
with col2:
    patient_data = sari[sari['id'] == grpIdS].dropna()

    input_columns = ['serum_creatinine_level', 'total_count', 'random_blood_sugar', 'urea',
                    'haemoglobin_level', 'bmi', 'diabetes_encoded', 'living_area_encoded', 'worker_encoded', 'diet_encoded']
    forecast_column = 'glomerular_filration'

    try:
        logging.info("Started forecating for the next three years...")
        forecast_values = forecast_next_three_years_sari(patient_data, input_columns, forecast_column)
        # st.write(forecast_values)
        forecast_values = np.insert(forecast_values,0,lastrecord["glomerular_filration"][len(lastrecord)-1],axis=0)
        fv = pd.DataFrame(forecast_values,columns=["glomerular_filration"])
        temp = pd.concat([lastrecord[["glomerular_filration"]],fv]).reset_index()
        # temp["0"] =temp.iloc[len(df)-4:]["glomerular_filration"]
        temp["Forecasted"] = temp.iloc[len(temp)-4:]["glomerular_filration"]
        temp["yr"] = [i for i in range(len(temp))]
        fig,ax = plt.subplots()
        ax.plot(temp["yr"],temp["glomerular_filration"],label="Analysis")
        ax.plot(temp["yr"],temp["Forecasted"],label="Forecasted")
        ax.legend()
        ax.set_xlabel("Year")
        ax.set_ylabel("eGFR")
        # st.write(temp)
        # st.write(forecast_values)
        st.pyplot(fig)

        if forecast_values is not None:
            st.write(f'Forecasted GFR values for Patient "{grpIdS}" for the Next 3 Years:')
            for year in range(1, 4):
                st.write(f'GFR value for Year {year}: {forecast_values[year]}')
            sum = (forecast_values[2] - forecast_values[3])
            measure = forecast_values[2] *(4.416/100)
            #st.write("The change of difference of GFR ",round(sum,5))
            #st.write("The threshold value for Indian patient as per research ",round(measure,2))
            st.write("Based on the research as per the threshold value for Indian patient")

            if(sum>measure or forecast_values[3]<25):
                st.write("Patient is in Risk Condition")
            else:
                st.write("Patient is not in Risk Condition") 
            logger.info(f'Forecasting for the patient {grpIdS} is sucessfull')
        else:
            st.write("Forecasting failed.")
            logging.info(f'Internal error occured at the model, main file at forecasting')
    except Exception as e:
        st.write(str(e))
        logging.error("Exception araises after the failer of the forecasting fopr the patient",grpIdS)
