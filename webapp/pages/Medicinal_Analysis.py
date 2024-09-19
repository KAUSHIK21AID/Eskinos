import logging
import time
import streamlit as st
import pandas as pd
import sys
sys.path.append("helpers/")
from Pipeline import Staging
#from streamlit_extras.metric_cards import style_metric_card


import streamlit as st
st.set_page_config(page_title="Metrics",layout="centered")
#streamlit-extras-code
css = '''
<style>
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    [data-testid="stMetricDelta"] > div::before {
        content:"♦";
        font-weight: bold;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)
box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;")
st.markdown(
        f"""
        <style>
            div[data-testid="stMetric"],
            div[data-testid="metric-container"] {{
                background-color: black;
                border: 2px "";
                padding: 5% 5% 5% 10%;
                border-radius: 10px;
                border-left: 0.5rem solid maroon;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data
def convert_to_list(df):
    df["Meds that helped regression of Stage"] = df["Meds that helped regression of Stage"].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
    df["Meds that helped regression of Stage"] = df["Meds that helped regression of Stage"].astype(str)
    return df

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
      # print(m[j])
      meds.append(m[j])
    final = ret(meds)
    # print(final)
    meds_gfr_increase.append(final)

  df['id'] = p
  df['Meds_helped'] = meds_gfr_increase
  df = df.sort_values('id')
  return df

@st.cache_data
def stages_alone(data,risk):
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


  lr_p = list(risk[risk['Risk Factor']=='Low Risk']['Patient id'])
  gfr = list(data['glomerular_filration'])
  s = []
  patients = list(data['id'].unique())

  med_vs_patient = []
  id = []
  for i in patients:
    df1 = data[data['id']==i]
    df1['hospital_visited_date'] = df1['hospital_visited_date'].apply(pd.Timestamp)
    df1 = df1.sort_values('hospital_visited_date')
    st = list(df1['stage'])
    m = list(df1['medications'])
    # print(st)
    ind = []
    med = []
    c = 0
    for j in range(0,len(st)-1):
      if(st[j]>st[j+1]):
        c+=1
        ind.append(j)
    if(c==0):
      id.append(i)
      med_vs_patient.append('')
    else:
      # print(ind)
      for k in ind:
        med.append(m[k])
      # print(med)
      final = ret(med)
      # print(final)
      id.append(i)
      med_vs_patient.append(final)

  meds_vs_stage = pd.DataFrame()
  meds_vs_stage['Patient id'] = id
  meds_vs_stage['Meds that helped regression of Stage'] = med_vs_patient
  meds_vs_stage = meds_vs_stage.sort_values('Patient id')

  index_null = meds_vs_stage[meds_vs_stage['Meds that helped regression of Stage']== ''].index
  meds_vs_stage.drop(index_null , inplace=True)

  patients = []
  for i in list(meds_vs_stage['Patient id']):
    if i in lr_p:
      patients.append(i)
  ourdata = pd.DataFrame()
  for _,j in meds_vs_stage.iterrows():
    if(j['Patient id'] in patients):
      ourdata = pd.concat([ourdata,j],axis=1)
      # ourdata = ourdata.T
  ourdata=ourdata.T
  ourdata= convert_to_list(ourdata)
  whole = [ ]
  for _,i in meds_vs_stage.iterrows():
    j = i['Meds that helped regression of Stage']
    for k in j:
      whole.append(k)

  lowrisk_patients = list(ourdata['Patient id'].unique())
  highrisk_patients = []
  all_patients = list(data['id'].unique())
  for i in all_patients:
    if i not in lowrisk_patients:
      highrisk_patients.append(i)

  meds_high = []
  for i in highrisk_patients:
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
      # print(m[j])
      meds.append(m[j])
    final = ret(meds)
    # print(final)
    for j in final:
      meds_high.append(j)

  fdf = pd.DataFrame()
  fdf['MM'] = meds_high

  meds_high_unique = list(fdf['MM'].unique())
  meds_not_used_by_low_risk = []
  for i in meds_high_unique:
    if i not in whole:
      meds_not_used_by_low_risk.append(i)

  meds_unused_much = pd.DataFrame()

  meds_unused_much['Number'] = [i for i in range(1, len(meds_not_used_by_low_risk)+1)]

  meds_unused_much['Medicine id'] = meds_not_used_by_low_risk

  meds_only_taken_by_low = []
  for i in whole:
    if i not in meds_high:
      meds_only_taken_by_low.append(i)

  fdddd = pd.DataFrame()

  fdddd['Medicine id'] = meds_only_taken_by_low
  unique_meds = list(fdddd['Medicine id'].unique())

  doja = pd.DataFrame()
  doja['Number'] = [i for i in range(1, len(unique_meds)+1)]
  doja['Medicine id'] = unique_meds

  return meds_unused_much,doja

@st.cache_data
def gfr_alone(data,df):

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


  whole = []
  for i in list(df['Meds_helped']):
    for j in i:
      whole.append(j)
  lowrisk_patients = list(df['id'].unique())
  highrisk_patients = []
  all_patients = list(data['id'].unique())
  for i in all_patients:
    if i not in lowrisk_patients:
      highrisk_patients.append(i)

  meds_high = []
  for i in highrisk_patients:
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
      # print(m[j])
      meds.append(m[j])
    final = ret(meds)
    # print(final)
    for j in final:
      meds_high.append(j)

  meds_only_taken_by_low = []
  for i in whole:
    if i not in meds_high:
      meds_only_taken_by_low.append(i)

  fdddd = pd.DataFrame()
  fdddd['Medicine id'] = meds_only_taken_by_low
  unique_meds = list(fdddd['Medicine id'].unique())

  doja = pd.DataFrame()
  doja['Number'] = [i for i in range(1, len(unique_meds)+1)]
  doja['Medicine id'] = unique_meds

  ddd = pd.DataFrame()
  ddd['M'] = meds_high
  meds_high_unique = list(ddd['M'].unique())
  meds_not_used_by_low_risk = []
  for i in meds_high_unique:
    if i not in whole:
      meds_not_used_by_low_risk.append(i)

  meds_unused_much = pd.DataFrame()

  meds_unused_much['Number'] = [i for i in range(1, len(meds_not_used_by_low_risk)+1)]

  meds_unused_much['Medicine id'] = meds_not_used_by_low_risk
  # meds_unused_much.to_csv('Meds Unused_GFR alone.csv')
  # doja.to_csv('GFR___Meds only taken by Low Risk patients.csv')

  return meds_unused_much,doja

#st.cache_data
def analyze_risk(grouped, risk_threshold=0.5):
    unique_patient_names = set()
    risk_analysis = pd.DataFrame(columns=['Patient id', 'Risk Factor'])

    for id, group in grouped:
        total_visits = len(group)
        stage_counts = group['stage'].value_counts()

        # Checking Stage Conditions
        stage_4_count = stage_counts.get('Stage 4', 0)
        stage_5_count = stage_counts.get('Stage 5', 0)

        # Check if either Stage 4 or Stage 5 counts exceed the threshold or difference in GFR
        first_gfr = group.iloc[0]['glomerular_filration']
        last_gfr = group.iloc[-1]['glomerular_filration']
        gfr_difference = abs(last_gfr - first_gfr)

        last_stage = group.iloc[-1]['stage']

        if ((stage_4_count + stage_5_count) > (total_visits * risk_threshold)) and (last_stage in ['Stage 4', 'Stage 5']) or (gfr_difference >= 15):
            temp = pd.DataFrame(data ={'Patient id': id, 'Risk Factor': 'High Risk'},index=[0])
            risk_analysis = pd.concat([temp,risk_analysis], ignore_index=True)
            unique_patient_names.add(id)  # Add the unique patient id
        else:
            temp = pd.DataFrame(data ={'Patient id': id, 'Risk Factor': 'Low Risk'},index=[0])
            risk_analysis = pd.concat([temp,risk_analysis], ignore_index=True)
            unique_patient_names.add(id)  # Add the unique patient id

    return risk_analysis


conn = st.connection('mysql', type='sql')

df = conn.query('SELECT * from finaliti', ttl=600)
meds = conn.query('SELECT * from new_table', ttl=600)
meds["id"] = meds.index


df = Staging(df=df)
data = gfr_meds(df)

# import streamlit as st


# Set up logging
# logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Define a custom Streamlit logger
logger = logging.getLogger("streamlit")

# Add a file handler to log to a file
file_handler = logging.FileHandler("streamlit.log")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Example usage of logging
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")


# df.to_csv("Finaliti.csv")
grouped = df.groupby("id")
res_df=analyze_risk(grouped=grouped)
# st.write(res_df["Risk Factor"].value_counts())
risk_patients_count = len(res_df[res_df["Risk Factor"] == "High Risk"])
low_patients_count = len(res_df[res_df["Risk Factor"] == "Low Risk"])
risk_percentage = str(round(risk_patients_count / len(res_df) *100 , 2))+"%"
low_percentage = str(round(low_patients_count / len(res_df) *100 , 2))+"%"
col3,col4 = st.columns([5,5])
# col1.metric("High-Risk Patient Count",risk_patients_count,delta=risk_percentage,delta_color="inverse")
# col2.metric("Low-Risk Patient Count",low_patients_count,delta=low_percentage,delta_color="normal")
col3.metric("Meds in DB",len(meds['id']),help="Total medicines count in DB",delta="Counts",delta_color="off")
col4.metric("Patients in DB",len(df['id'].unique()),help="Total medicines count in DB",delta="Counts",delta_color="off")


MedsNotByGfr,MedsByGfr = gfr_alone(data=df,df=data)
MedsNotByStages,MedsByStages = stages_alone(data=df,risk=res_df)
# samp = stages_alone(data=df,risk=res_df)
# st.write(samp)

#need to add stages
###############################
st.write("---")
c1,c3 = st.columns([5,5])
c1.write("<u>Medicines used in terms <b>GFR - Low Risk</b></u> ("+str(len(MedsByGfr))+")",unsafe_allow_html=True)
c1.dataframe(MedsByGfr,width=400,height=450,hide_index=True)
c3.write("<u>Medicines used in terms <b>Stages - Low Risk</b></u> ("+str(len(MedsByStages))+")",unsafe_allow_html=True)
c3.dataframe(MedsByStages,width=400,height=450,hide_index=True)
c1.write("<u>Medicines used in terms <b>GFR - High Risk</b></u> ("+str(len(MedsNotByGfr))+")",unsafe_allow_html=True)
c1.dataframe(MedsNotByGfr,width=400,height=450,hide_index=True)
c3.write("<u>Medicines used in terms <b>Stages - High Risk</b></u> ("+str(len(MedsNotByStages))+")",unsafe_allow_html=True)
c3.dataframe(MedsNotByStages,width=400,height=450,hide_index=True)
#set data
#set input
#set integration
