import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import multiprocessing as mp

def Data_modulations(df):
   return



@st.cache_data
def AddingAttributes(df):
    # df['bmi'] = round(((df["weight"] / (df["height"] * df["height"]))*703),2)
    df['bmi'] = round((df['weight']/(df['height']*df['height']))*10000,2)
    df[["systolic","diastolic"]] = df["bp"].str.split('/',expand=True)
    df["systolic"] = df["systolic"].astype(int)
    df["diastolic"] = df["diastolic"].astype(int)
    return df

@st.cache_data  
def LabelConverter(df):
    for i in df.columns:
        if((i != 'bp') & (i!= 'medications')):
            if df[i].dtypes == 'object':
                le=LabelEncoder()
                df[i] = le.fit_transform(df[i])
    return df
    
@st.cache_data
def Staging(df):
    st = []
    for i in df["glomerular_filration"]:
        if(i>90):
            st.append(1)
        elif ((60 <= i) & (i <=90)):
            st.append(2)
        elif ((30 <= i) & (i <=50)):
            st.append(3)
        elif((15<= i) &  (i<= 29)):
            st.append(4)
        else:
            st.append(5)
    df["stage"] = st
    return df

@st.cache_data
def AddingVisits(df):
    samp = pd.DataFrame()
    for i in df["id"].unique():
        temp = df[df["id"] == i]
        temp["No.of visits"] = len(temp)
        samp = pd.concat([samp,temp])
    return samp

@st.cache_data
def interpolate_patient_data(patient_df):
    patient_df = patient_df.sort_values(by='No.of visits')
    numerical_columns = ['age', 'height', 'weight', 'glomerular_filration',
                         'serum_creatinine_level', 'haemoglobin_level',
                         'total_count', 'random_blood_sugar', 'urea', 'bmi', 'systolic', 'diastolic'] # Numerical columns

    patient_df[numerical_columns] = patient_df[numerical_columns].interpolate(method='linear', axis=0)

    interpolated_rows = []
    for i in range(len(patient_df) - 1):
        mean_values = patient_df.iloc[i:i + 2][numerical_columns].mean()

        new_row = pd.Series({                          # Creating new rows between 2 visits
            'id': patient_df.iloc[i]['id'],
            'age': mean_values['age'],
            'height': mean_values['height'],
            'weight': mean_values['weight'],
            'glomerular_filration': mean_values['glomerular_filration'],
            'serum_creatinine_level': mean_values['serum_creatinine_level'],
            'haemoglobin_level': mean_values['haemoglobin_level'],
            'total_count': mean_values['total_count'],
            'random_blood_sugar': mean_values['random_blood_sugar'],
            'urea': mean_values['urea'],
            'bmi': mean_values['bmi'],
            'systolic': mean_values['systolic'],
            'diastolic': mean_values['diastolic'],
            'No.of visits': patient_df.iloc[i]['No.of visits'] + 0.5
        })

        interpolated_rows.append(new_row)

    patient_df = pd.concat([patient_df, pd.DataFrame(interpolated_rows)], ignore_index=True)

    # ffil for categorical columns
    categorical_columns = ['id', 'hospital_visited_date', 'gender', 'medications', 'bp', 'diabetes',
                            'hypertension', 'family_history', 'previous_kidney_disease','urine_albumin',
                            'cardiovascular_disease', 'urinary_tract_infection','pus_cells',
                            'kidney_stone', 'kidney_injury_history', 'total_water_intake',
                            'living_area', 'worker', 'diet', 'other_medical_condition', 'stage']

    patient_df[categorical_columns] = patient_df[categorical_columns].fillna(method='ffill')

    return patient_df

@st.cache_data
def Interpolating(df):
    df_less_than_10_visits = df[df['No.of visits'] < 10]
    df_interpolated = df_less_than_10_visits.groupby('id', group_keys=False).apply(interpolate_patient_data)
    df_combined = pd.concat([df, df_interpolated], ignore_index=True)

    df_combined = df_combined.sort_values(by=['id', 'No.of visits']).reset_index(drop=True)
    return df_combined

@st.cache_data
def gfr_meds(data):
  def ret(given):
    medication = []
    medic = []
    for i in given:
      a = i.split(',')
      # print(a)
      for j in a:
        j = j.replace('"','')
        medication.append(j)
    for i in medication:
      if i not in medic:
        medic.append(i)
    return(medic)
  patients = list(data['id'].unique())
  gfr_first_last = []
  df = pd.DataFrame()
  for i in patients:
    d_each = data[data['id']==i]
    d_each['hospital_visited_date'] = d_each ['hospital_visited_date'].apply(pd.Timestamp)
    d_each = d_each.sort_values(by='hospital_visited_date')

    gfr = list(d_each['glomerular_filration'])

    gfr_first_last.append(gfr[len(gfr)-1]-gfr[0])

  d_bmi_gfr = pd.DataFrame()

  d_bmi_gfr['id'] = patients

  d_bmi_gfr['Deviation in GFR'] = gfr_first_last

  p = list(d_bmi_gfr[d_bmi_gfr['Deviation in GFR']>=15]['id'].unique())

  meds_gfr_increase = []
  for i in p:
    meds = []
    df2 = data[data['id']==i]
    # print(len(df2))
    df2['hospital_visited_date'] = df2['hospital_visited_date'].apply(pd.Timestamp)
    df2 = df2.sort_values('hospital_visited_date')
    m = list(df2['medications'])
    n = int(0.8*len(m))
    # print(n)
    for j in range(n,len(m)):
      # print(j)
    #   print(m[j])
      meds.append(m[j])
    final = ret(meds)
    # print(final)
    meds_gfr_increase.append(final)
#   print(meds_gfr_increase)

  df['id'] = p
  df['Meds_helped'] = list(meds_gfr_increase)
#   st.write(df["Meds_helped"])
  df = df.sort_values('id')
  # print(df)
  return df

@st.cache_data
def convert_to_list(df):
    df["Meds_helped"] = df["Meds_helped"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df["Meds_helped"] = df["Meds_helped"].astype(str)
    return df

@st.cache_data
def convert_to_string(input_list):
    output = ",".join(input_list)
    return output

@st.cache_data
def Model(df):
    # df["total_water_intake"] = df["total_water_intake"].astype(float)
    unique_patients = len(df['id'].unique())
    x = df[['bmi', 'serum_creatinine_level', 'haemoglobin_level', 'age','random_blood_sugar', 'total_count', 'systolic','diastolic', 'diabetes', 'total_water_intake','living_area','cardiovascular_disease','previous_kidney_disease','kidney_injury_history']]
    y = df['glomerular_filration']
    le = LabelEncoder()
    x['living_area'] = le.fit_transform(x['living_area'])
    x['cardiovascular_disease'] = le.fit_transform(x['cardiovascular_disease'])
    x['previous_kidney_disease'] = le.fit_transform(x['previous_kidney_disease'])
    x['kidney_injury_history'] = le.fit_transform(x['kidney_injury_history'])

    x['diabetes'] = le.fit_transform(x['diabetes'])
    rfr  = RandomForestRegressor(n_jobs=mp.cpu_count())
    model = rfr.fit(x,y)
    score = rfr.score(x,y)
    return model,score,unique_patients
