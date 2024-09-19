import streamlit as st
import pandas as pd
from datetime import datetime
cell_hover = {
    "selector": "td:hover",
    "props": [("background-color", "#FFFFE0")]
}
index_names = {
    "selector": ".index_name",
    "props": "font-style: italic; color: darkgrey; font-weight:normal;"
}
headers = {
    "selector": "th:not(.index_name)",
    "props": "background-color: #800000; color: white;"
}
st.set_page_config(page_title="Data Visits",layout="wide")


st.markdown("<h2 style='text-align:center;colour:white;'><u>Key Metrics of Patients</u></h2>",unsafe_allow_html=True)



def month_diff(date1, date2):
    return (date2.year - date1.year) * 12 + date2.month - date1.month



conn = st.connection('mysql', type='sql')
df = conn.query('SELECT * from finaliti;', ttl=600)
gfr1,gfr2,scr1,scr2,gen,diab,hyper,months,visits,age = list(),list(),list(),list(),list(),list(),list(),list(),list(),list()
attr = ['glomerular_filration','serum_creatinine_level']

patients = df["id"].unique()
for patient in patients:
    temp_grp = df[df['id'] == patient].reset_index()
    temp_grp["hospital_visited_date"] = temp_grp["hospital_visited_date"].apply(pd.Timestamp)
    temp_grp = temp_grp.sort_values(by="hospital_visited_date")
    # st.write(temp_grp[attr[0]])
    # break
    gfr1.append(temp_grp[attr[0]][0])
    scr1.append(temp_grp[attr[1]][0])
    gfr2.append(temp_grp[attr[0]][len(temp_grp)-1])
    scr2.append(temp_grp[attr[1]][len(temp_grp)-1])
    gen.append(temp_grp['gender'][0])
    age.append(temp_grp['age'][len(temp_grp)-1])
    diab.append(temp_grp['diabetes'][0])
    hyper.append(temp_grp['hypertension'][0])
    months.append(month_diff(temp_grp['hospital_visited_date'][0],temp_grp['hospital_visited_date'][len(temp_grp)-1]))
    visits.append(len(temp_grp))

datas = [patients,age,gen,diab,hyper,gfr1,scr1,gfr2,scr2,months,visits]
columns = ["ID","Age","Gender","Diabetes","Hypertension","GFR first visit","Creatinine first Visit","GFR last visit","Creatinine last visit","Months","visits"]
dff = {}
for dat,col in zip(datas,columns):
    dff[col] = dat

# for i in datas:
#     st.write(len(i))

# data = pd.DataFrame([patients,age,gen,diab,hyper,gfr1,scr1,gfr2,scr2,months,visits],columns = ["ID","Age","Gender","Diabetes","Hypertension","GFR first visit","Creatinine first Visit","GFR last visit","Creatinine last visit","Months","visits"])
dff = pd.DataFrame(dff).sort_values(by='ID').reset_index().drop('index',axis=1)

dff['Frequency of visits'] = round(dff['Months'] / dff['visits'] , 3)
dff['Change in GFR'] = dff['GFR last visit'] - dff['GFR first visit']

dff.style.set_table_styles([cell_hover, index_names, headers])
dff.style.hide()

st.dataframe(dff,height=700)
