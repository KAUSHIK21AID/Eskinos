import streamlit as st
import time
import numpy as np
import pandas as pd
import multiprocessing as mp
# import streamlit_scrollable_textbox as st.write
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append("helpers/")
from Pipeline import AddingAttributes,LabelConverter,gfr_meds,convert_to_list
st.set_page_config(page_title="Recommander")
st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
.vbig-font {
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def ret(given):
    nnn = given.split(',')
    medication = []
    for h in nnn:
        h = h.replace('"','')
        medication.append(h)
    return medication

@st.cache_data
def retrieve_sugg_meds(patient_id):
    compare_meds = (tab1['Meds_helped'])

    real_compare_meds = []
    for mee in compare_meds:
        real_compare_meds.append((mee))
    # st.write(real_compare_meds)
    meds_taken_visit = (d1['medications'])
    # st.write(d1)
    # st.write("Heelllo",meds_taken_visit)
    # st.write(d1['medications'])
    # st.write(data.tail(5))
    final = ret(meds_taken_visit)
    # st.write(final[0])

    # List to compare
    target_list = final

    # Set of lists to compare to
    lists_to_compare = real_compare_meds
    # st.write(real_compare_meds)

    # Convert lists to strings for vectorization
    target_str = ' '.join(target_list)
    # st.write(target_str[0])

    # Vectorize the target list
    vectorizer = CountVectorizer().fit_transform([target_str] + lists_to_compare)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    similarities = cosine_similarity(vectors)

    # Find the index of the most similar list
    most_similar_index = similarities[0, 1:].argmax()

    # Get the most similar list
    most_similar_list = lists_to_compare[most_similar_index]

    # Convert most_similar_list to a list of strings
    most_similar_list = most_similar_list.split(', ')

    only_in_list1 = [item for item in most_similar_list if item not in final]
    list_of_meds = []
    # st.write("Suggested Tablets")
    for item in only_in_list1:
        item = item.replace('[','')
        item = item.replace(']','')
        item = item.replace("'",'')
        # st.write(item)
        list_of_meds.append(item)
    final_recomm = [item for item in list_of_meds if item not in final]

    return final_recomm

@st.cache_data
def random_value_generator(high_risk_patient,recursion_count,max_recursions):
    if recursion_count > max_recursions:
        return None
    hypothetical_patient_data = {}

    for field in data.columns:
        if field in changable:
            # Ensure the field is numeric, excluding non-numeric fields like "Medications"
            if data[field].dtype == np.number:
                # Calculate the range from low-risk patients
                # Generate a random value within the range
                min_range = low[field].min()
                max_range = low[field].max()
                if(field=='bmi'):
                    min_range = 18.5
                    max_range = 24.9
                if(field=='haemoglobin_level'):
                    min_range = 12.1
                    max_range = 17.2
                if(field=='random_blood_sugar'):
                    min_range = 100.0
                    max_range = 150.0
                if(field=='systolic'):
                    min_range = 110.0
                    max_range = 130.0
                if(field=='diastolic'):
                    min_range = 60.0
                    max_range = 80.0
                hypothetical_patient_data[field] = np.random.uniform(min_range, max_range)
            else:
                # If the field is non-numeric, take unique values and generate a random choice
                unique_values = low[field].unique()
                hypothetical_patient_data[field] = np.random.choice(unique_values)
        else:
            # Keep the original value from high-risk patients for other fields
            hypothetical_patient_data[field] = high_risk_patient[field]
    my_input = []
    for i in input_columns:
        if i=='random_blood_sugar':
            hypothetical_patient_data[i] = np.random.uniform(70, 150)
        my_input.append(hypothetical_patient_data[i])
    hypogfr = model.predict([my_input])
    if(hypogfr>hypothetical_patient_data['glomerular_filration']+3.0):
        return hypothetical_patient_data
    else:
        return random_value_generator(high_risk_patient, recursion_count=recursion_count + 1, max_recursions=max_recursions)

conn = st.connection( name="mysql", type="sql")


# low = conn.query('SELECT * from whatif_train;',ttl=0)
data = conn.query('SELECT * from finaliti', ttl=0)
# st.write(data.columns)
tab1 = convert_to_list(gfr_meds(data=data))
tab2 = convert_to_list(gfr_meds(data=data))['id']
# st.write(tab1)
low = pd.DataFrame()
for id in tab2:
    d = data[data['id']==id]
    low = pd.concat([low,d])
# low = data[data['id']==tab1['id']]
# st.write(low)
low = AddingAttributes(low)

# st.write(low)

data = LabelConverter(data)
# st.write(data['hypertension'].dtypes =='object')
# st.write(data)

data = AddingAttributes(data)
sys = []
dias = []
for i in list(data['bp']):
    a = i.split('/')
    sys.append(int(a[0]))
    dias.append(int(a[1]))

data['systolic'] = sys
data['diastolic'] = dias
# tab
# low
patients = list(data['id'].unique())
# len(patients)
high_id = []
for i in patients:
    if i not in list(low['id']):
        high_id.append(i)
# len(high_id)
high = pd.DataFrame()
# st.write(high_id)
for i in high_id:
    df = data[data['id']==i]
    # st. write(df)
    high = pd.concat([high,df.tail(1)])
    # st.write(high)
    # break
# 

# st.write(high)
# data = data.drop(['Unnamed: 0.7', 'Unnamed: 0.6', 'Unnamed: 0.5', 'Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'index'], axis = 1)
data['hospital_visited_date'] = data['hospital_visited_date'].apply(pd.Timestamp)
data = data.sort_values(by = 'hospital_visited_date')
# data
# high = high.drop(['Unnamed: 0.8', 'Unnamed: 0.7', 'Unnamed: 0.6', 'Unnamed: 0.5', 'Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'index'], axis = 1)
# high
sys = []
dias = []
for i in list(high['bp']):
    a = i.split('/')
    sys.append(int(a[0]))
    dias.append(int(a[1]))
high['systolic'] = sys
high['diastolic'] = dias
# high.columns
# low = low.drop('Unnamed: 0',axis=1)
# low
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
col = ['diabetes', 'hypertension', 'family_history', 'previous_kidney_disease', 'cardiovascular_disease', 'urinary_tract_infection', 'kidney_stone', 'kidney_injury_history', 'pus_cells', 'worker', 'diet', 'living_area','urine_albumin']
for i in col:
    low[i] = label.fit_transform(low[i])
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
col = ['diabetes', 'hypertension', 'family_history', 'previous_kidney_disease', 'cardiovascular_disease', 'urinary_tract_infection', 'kidney_stone', 'kidney_injury_history', 'pus_cells', 'worker', 'diet', 'living_area','urine_albumin']
for i in col:
    high[i] = label.fit_transform(high[i])
low = (LabelConverter(low))
x = low[['gender','age','diabetes','serum_creatinine_level','haemoglobin_level','total_water_intake','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi','systolic','diastolic','urine_albumin']]
# x = low[['Gender','cal_age','Diabetes','Serum Creatinine Level','haemoglobin_level','Tota Water Intake noise removed','Living Area','Total Count','random_blood_sugar','Urea','Worker','Diet','bmi','systolic','diastolic','Urine Albumin']]
y = low['glomerular_filration']

from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42,n_jobs=mp.cpu_count())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)

model = rf_regressor.fit(x_train, y_train)
# st.write(x_train)
y_pred = model.predict(x_test)
changable = ['haemoglobin_level','total_water_intake',
            'random_blood_sugar','systolic','diastolic','urine_albumin','living_area' , 'bmi','worker']
input_columns = ['gender','age','diabetes','serum_creatinine_level','haemoglobin_level','total_water_intake','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi','systolic','diastolic','urine_albumin']
hypothetical_data = pd.DataFrame()
for _, high_risk_patient in high.iterrows():
# For each high-risk patient, generate hypothetical data
    # Create a dictionary to store hypothetical patient data
    recursion_count = 0
    max_recursions = 20
    hypothetical_patient_data = random_value_generator(high_risk_patient,recursion_count,max_recursions)
    # st.write(hypothetical_patient_data)
    hypothetical_patient_data = pd.DataFrame(hypothetical_patient_data,index=[0])
    # st.write(hypothetical_patient_data)
    hypothetical_data = pd.concat([hypothetical_patient_data,hypothetical_data], ignore_index=True)
hypothetical_data = hypothetical_data.dropna()


# st.write(hypothetical_data)
hypogfr = model.predict(hypothetical_data[['gender','age','diabetes','serum_creatinine_level','haemoglobin_level','total_water_intake','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi','systolic','diastolic','urine_albumin']])
hypothetical_data['hypogfr'] = hypogfr
check = pd.DataFrame()
# check = []
for _, i in hypothetical_data.iterrows():
    if i['hypogfr'] > i['glomerular_filration'] + 3.0:
        temp = pd.DataFrame()
        temp = pd.concat([temp,i])
        # st.write(temp)
        check = pd.concat([check,temp.T])
        # st.write(check)
        # break
# st.write(check)

# st.markdown("<h1 style='text-align:center;colour:white;'>KIDNEY ANALYSIS</h1>",unsafe_allow_html=True)

st.markdown("<h2 style='text-align:center;colour:white;'><u>Recommender-Guiding Kidney Patients Through Essential Health Metrics</u></h2>",unsafe_allow_html=True)

st.info("It'll suggest recommandation to the patients with their previous data")
col1,col2 = st.columns([5,3])

# st.write(low["id"].unique())
list_of_Patients = set(data["id"])
with col1:
    patient_id = st.selectbox(label="Patient ID",options=list_of_Patients)
    with st.spinner('Please wait...'):
        time.sleep(1)
    checkp = list(check["id"].unique())
    not_recomm = []
    for i in patients:
        if i not in checkp:
            if i not in list(low['id']):
                not_recomm.append(i)    
    lowrisk = tab2  
    flag=0
    if patient_id in checkp:
        flag=1
        dc = high[high['id']==patient_id]
        # dc = dc.reset_index()
        hd = check[check['id']==patient_id]
        # hd = hd.reset_index()
        # st.write(dc)
        # st.write(low['name'].unique())
        # st.write(dc.head(5))
        d1 = dc.iloc[0]
        # st.write(hd)
        d2 = hd.iloc[0]
        # st.write(d2)
        for k in changable:
            if(k!='diet' and k!='worker' and k!='living_area'):
                if(k=='random_blood_sugar'):
                    d2[k] = round(d2[k], 0)
                if(k=='systolic'):
                    if(d2[k]>120):
                        d2[k]=120
                if(k=='diastolic'):
                    if(d2[k]>100):
                        d2[k]=80
                    if(d2[k]<60):
                        d2[k]=60
                dif = d1[k] - d2[k]
                st.write("<p class='vbig-font'><u><b>"+k+" : </b></u>",unsafe_allow_html=True)
                st.write("<p class='big-font'>Current - "+str(d1[k]),unsafe_allow_html=True)
                #   st.write(f"Current - {k}",str(d1[k]))
                st.write("<p class='big-font'>Suggestion - "+str(d2[k]),unsafe_allow_html=True)
                #   st.write(f"Suggestion - {k}",str(d2[k]))
                if(dif>0):
                    st.write("<p class='big-font'>Status - Decrease it",unsafe_allow_html=True)
                else:
                    st.write("<p class='big-font'>Status - Increase it",unsafe_allow_html=True)
            else:
                if(k=='worker' or k=='living_area' or k=='diet'):
                    if(d1[k]!=d2[k]):
                        st.write("<u><p class='vbig-font'><b>"+k+"</b></p></u> should be changed from the existing.",unsafe_allow_html=True)
                    else:
                        st.write("<u><p class='vbig-font'><b>"+k+"</b></p></u> should remain the same.",unsafe_allow_html=True)
                else:
                    if(d1[k]>d2[k]):
                        st.write("<u><p class='vbig-font'><b>"+k+"</b></p></u> should be increased.",unsafe_allow_html=True)
                    else:
                        st.write("<u><p class='vbig-font'><b>"+k+"</b></p></u> should be decreased.",unsafe_allow_html=True)
            # st.write()
        st.write("<p class='vbig-font'><b>"+str(d1['glomerular_filration'])+"</b> is the Actual GFR",unsafe_allow_html=True)
        # st.write()
        if(d2['hypogfr']>d1['glomerular_filration']+15.0):
            st.write("<p class='vbig-font'><b>"+str(d1['glomerular_filration']+15.0)+"</b> is the hypothetical GFR",unsafe_allow_html=True)
        else:
            st.write("<p class='vbig-font'><b>"+str(d2['hypogfr'])+"</b> is the hypothetical GFR",unsafe_allow_html=True)
        # st.write()
    elif patient_id in not_recomm:
        flag=1
        d = high[high['id']==patient_id]
        d1=d.iloc[0]
        h=0
        sys =0
        dias = 0
        rbs = 0
        bmi = 0
        st.write("<u><b>This Patient is healthy and good health metrics on:",unsafe_allow_html=True)
        if(list(d['haemoglobin_level'])[0]>=12.1 and list(d['haemoglobin_level'])[0]<=17.2):
            st.write("Haemoglobin level ✅")
        else:
            h=1

        if(list(d['random_blood_sugar'])[0]>=110 and list(d['haemoglobin_level'])[0]<=160):
            st.write("Random Blood Sugar ✅")
        else:
            rbs=1

        if(list(d['systolic'])[0]>=110 and list(d['systolic'])[0]<=120):
            st.write("Systolic pressure ✅")
        else:
            sys=1

        if(list(d['diastolic'])[0]>=60 and list(d['diastolic'])[0]<=80):
            st.write("Diastolic pressure ✅")
        else:
            dias=1

        if(list(d['bmi'])[0]>=18.5 and list(d['bmi'])[0]<=24.9):
            st.write("Body Mass Index ✅")
        else:
            bmi=1


        if(bmi==0 and sys==0 and dias==0 and rbs==0 and h==0):
            st.write("Every single attribute suggest you are on the right path of diagnosis and treatment")

        else:
            st.write("<b><u>Values that need to be concentrated on:",unsafe_allow_html=True)
            if(h==1):
                st.write("🔴 Haemoglobin level")
            if(sys==1 or dias==1):
                st.write("🔴 Blood Pressure")
            if(rbs==1):
                st.write("🔴 Random Blood Sugar")
            if(bmi==1):
                st.write("🔴 Body Mass Index")

    elif patient_id in list(low["id"].unique()):
        st.write("This patient is Progressing: LOW RISK PATIENT")
with col2:
    if(flag!=0):
        st.write("###")
        st.write("<p class='big-font'><b><u>Medicines that could help the GFR increase for this patient:<b><u>",unsafe_allow_html=True)
        mmm = retrieve_sugg_meds(patient_id)
        for qq in mmm:
            st.write("* "+qq)
