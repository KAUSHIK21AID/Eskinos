import streamlit as st
import numpy as np
import pandas as pd
import logging
import sys
sys.path.append("helpers/")
from Pipeline import AddingAttributes,convert_to_list, convert_to_string,gfr_meds
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(filename='logs.log')
st.set_page_config(page_title="Suggester")
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

conn = st.connection('mysql', type='sql')
logging.debug("this is a log message go the file")
logging.basicConfig(filename="logs.log")


# tab = pd.read_csv('Files/GFR_15_vs_Meds.csv')
low = conn.query('SELECT * from whatif_train;', ttl=600)
data = conn.query('SELECT * from finaliti;', ttl=600)
meds = conn.query('SELECT * from new_table;', ttl=600)
# tab = convert_to_list(gfr_meds(data=data))
hypothetical_patient_data = {}
tab1 = convert_to_list(gfr_meds(data=data))

@st.cache_data
def ret(given):
    nnn = given.split(',')
    medication = []
    for h in nnn:
        h = h.replace('"','')
        medication.append(h)
    return medication

@st.cache_data
def retrieve_sugg_meds(med):
    compare_meds = (tab1['Meds_helped'])

    real_compare_meds = []
    for mee in compare_meds:
        real_compare_meds.append((mee))
    # st.write(real_compare_meds)
    meds_taken_visit = med
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

def validate_inputs(age, diabetes, scr, haem, twi, area, tc, rbs, urea, worker, diet, height, weight, sys, dias, ualb, gfr):
    errors = []

    # Perform validation for each input field
    if not 0 <= int(age) <= 99:
        errors.append("Age must be between 0 and 99.")

    if not 0 <= int(diabetes) <= 1:
        errors.append("Diabetes must be either 0 or 1.")

    if not 0 <= float(scr) <= 30:
        errors.append("Creatinine must be between 0 and 30.")

    if not 0 <= float(haem) <= 20:
        errors.append("Haemoglobin must be between 0 and 20.")

    if not 1 <= float(twi) <= 4:
        errors.append("Water Intake must be between 1 and 4.")

    if not 0 <= int(area) <= 1:
        errors.append("Area must be either 0 or 1.")

    if not 1000 <= float(tc) <20000:
        errors.append("Total Count must be between 1000 and 20000")

    if not 0 <= float(rbs) <= 500:
        errors.append("Random Blood Sugar must be between 0 and 500.")

    if not 0 <= float(urea) <= 300:
        errors.append("Urea must be between 0 and 300.")

    if not 0 <= int(worker) <= 1:
        errors.append("Worker must be either 0 or 1.")

    if not 0 <= int(diet) <= 1:
        errors.append("Diet must be either 0 or 1.")

    if not 100 <= float(height) <= 195: # remove bmi instead calc
        errors.append("height must be between 100cm and 195cm")

    if not 20 <= float(weight) <= 210: # remove bmi instead calc
        errors.append("weight must be between 20kg and 210")

    if not 0 <= int(sys) <= 210:
        errors.append("Systolic must be between 0 and 210.")

    if not 0 <= int(dias) <= 130:
        errors.append("Diastolic must be between 0 and 130.")

    if not 0 <= int(ualb) <= 6:
        errors.append("Urine Albumin must be between 1 and 6.")

    if not 0 <= float(gfr) <=120:
        errors.append("GFR must be must be between 0 and 120.")

    return errors

@st.cache_data
def random_value_generator(high_risk_patient,recursion_count,max_recursions):
  if recursion_count > max_recursions:
    # st.write("hello there")
    # st.write(recursion_count)
    # st.write(max_recursions)
    return hypothetical_patient_data
  input_columns = ['gender','age','diabetes','serum_creatinine_level','haemoglobin_level','total_water_intake','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi','Systolic','Diastolic','urine_albumin']
  for field in high.columns:
      if field in changable:
          # Ensure the field is numeric, excluding non-numeric fields like "Medications"
          if high[field].dtype == np.number:
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
                min_range = 70
                max_range = 150
              if(field=='Systolic'):
                min_range = 110
                max_range = 130
              if(field=='Diastolic'):
                min_range = 60
                max_range = 80
              hypothetical_patient_data[field] = np.random.uniform(min_range, max_range)
          else:
              # If the field is non-numeric, take unique values and generate a random choice
              unique_values = low[field].unique()
              hypothetical_patient_data[field] = np.random.choice(unique_values)
      else:
          # Keep the original value from high-risk patients for other fields
          hypothetical_patient_data[field] = high_risk_patient[field]
        #   st.write(hypothetical_patient_data)
  my_input = []
  for i in input_columns:
    if i=='random_blood_sugar':
      hypothetical_patient_data[i] = np.random.uniform(70, 150)
    my_input.append(hypothetical_patient_data[i])
  hypogfr = model.predict([my_input])
  if(hypogfr>hypothetical_patient_data['glomerular_filtration']+3.0):
    return hypothetical_patient_data
  else:
    return random_value_generator(high_risk_patient, recursion_count=recursion_count + 1, max_recursions=max_recursions)
sys = []
dias = []
for i in list(data['bp']):
  a = i.split('/')
  sys.append(int(a[0]))
  dias.append(int(a[1]))
data['Systolic'] = sys
data['Diastolic'] = dias
# data = data.drop(['Unnamed: 0.7', 'Unnamed: 0.6', 'Unnamed: 0.5', 'Unnamed: 0.4', 'Unnamed: 0.3', 'Unnamed: 0.2', 'Unnamed: 0.1', 'Unnamed: 0', 'index'], axis = 1)
data['hospital_visited_date'] = data['hospital_visited_date'].apply(pd.Timestamp)
data = data.sort_values(by = 'hospital_visited_date')
# low = low.drop('Unnamed: 0',axis=1)
from sklearn.preprocessing import LabelEncoder
# Assuming 'low' is your DataFrame and 'col' is the list of columns to encode
label = LabelEncoder()
col = ['diabetes', 'hypertension', 'family_history', 'previous_kidney_disease', 'cardiovascular_disease', 'urinary_tract_infection', 'kidney_stone', 'kidney_injury_history', 'pus_cells', 'worker', 'diet', 'living_area','urine_albumin']
# Iterate through each column and encode the labels
for i in col:
    low[i] = label.fit_transform(low[i])

# st.write the mapping of original labels to encoded labels
# st.write(f"Column: {i}")
# st.write(f"Original Labels: {label.classes_}")
# st.write(f"Encoded Labels: {list(range(len(label.classes_)))}")
# st.write()


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
col = ['diabetes', 'hypertension', 'family_history', 'previous_kidney_disease', 'cardiovascular_disease', 'urinary_tract_infection', 'kidney_stone', 'kidney_injury_history', 'pus_cells', 'worker', 'diet', 'living_area','urine_albumin']
for i in col:
  low[i] = label.fit_transform(low[i])
x = low[['gender','age','diabetes','serum_creatinine_level','haemoglobin_level','total_water_intake','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi','Systolic','Diastolic','urine_albumin']]
y = low['glomerular_filration']
# x.columns
from sklearn.ensemble import RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 42)
model = rf_regressor.fit(x_train, y_train)

st.markdown("<h2 style='text-align:center;colour:white;'><u>New Patient Health Metric Guide: Tailored Suggestions for Optimal Wellness</u></h2>",unsafe_allow_html=True)

st.info("It'll make suggestions for the patients with their current data")

col1,col2 = st.columns([5,5])
st.markdown(
    """
<style>
button {
    height: auto;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)
with col1:
    # st.write('Enter gender 0/1')
    gender = (st.selectbox(label="gender",options=[0,1],help="1 for male,0 for female"))
    # a = st.text_st.text_input(label=)
    # st.write('Enter Age')
    age = (st.text_input(label="Age"))

    # st.write('Enter Diabtetes 0/1')
    diabetes = (st.selectbox(label="diabetes",options=[0,1],help="1 - diabetes,0 - No diabetes"))

    # st.write('Enter SCR')
    scr = (st.text_input(label="Creatinine"))

    # st.write('Enter haemoglobin_level')
    haem = (st.text_input(label="Haemoglobin"))

    # st.write('Enter Water Intake')
    twi = (st.text_input(label="Water Intake"))

    # st.write('Enter living_area 0/1')
    area = (st.selectbox(label="Area",options=[0,1],help="1 - Urban,0 - Rural"))

    # st.write('Enter total_count')
    tc =(st.text_input(label="total_count"))

          
with col2:
    # st.write('Enter random_blood_sugar')
    rbs = (st.text_input(label="random_blood_sugar"))

    # st.write('Enter urea')
    urea = (st.text_input(label="urea"))

    # st.write('Enter worker 0/1')
    worker = (st.selectbox(label="worker",options=[0,1],help="0 - Field worker,1 - Manual worker"))

    # st.write('Enter diet 0/1')
    diet = (st.selectbox(label="diet",options=[0,1],help="0 - Vegan,1 - Non-Vegan"))

    col1_1,col1_2 = st.columns([5,5])
    with col1_1:
        height = st.text_input("height")
    with col1_2:
        weight = st.text_input("weight")

    # st.write('Enter Systolic')
    sys = (st.text_input(label="Systol"))

    # st.write('Enter Diastolic')
    dias = (st.text_input(label="Diastol"))

    # st.write('Enter urine_albumin 0/1/2/3/4')
    ualb = (st.selectbox(label="urine_albumin",options=[0,1,2,3,4],help="Choose your Urine Alb rate"))

col3,col4 = st.columns([7,3])
with col3:
    col5,col6 = st.columns([5,5])
    with col5:
        gfr = (st.text_input(label="GFR"))
    with col6:
       med = st.multiselect("Meds",options=meds['dosage'])
with col4:
    st.write("###")
    sub = st.button("Get Suggestion",use_container_width=True,type="primary",help="It may take sometime to get suggestion")
try:    
    if sub:
        colv1,colv2= st.columns([6,4])
        with colv1:
            flag=False
            correct = False
            cols1 = [age,scr,haem,twi,tc,rbs,urea,height,weight,sys,dias]
            for i in cols1:
                if(len(i)<1):
                    flag=True
                    break
            if(flag):
                st.error("Check All the Fields have filled?")
            elif (not flag):
                errors = validate_inputs(age, diabetes, scr, haem, twi, area, tc, rbs, urea, worker, diet, height, weight, sys, dias, ualb, gfr)
                if errors:
                    st.error("Validation failed:")
                    for error in errors:
                        st.error(error)
                else:
                    correct = True
            if(correct):
                high = pd.DataFrame()
                gender = int(gender)
                high['gender'] = [int(gender)]
                high['age'] = int(age)
                high['diabetes'] = int(diabetes)
                high['gender'] = float(gender)
                high['serum_creatinine_level'] = float(scr)
                high['haemoglobin_level'] = float(haem)
                high['total_water_intake'] = float(twi)
                high['living_area'] = int(area)
                high['total_count'] = int(tc)
                high['random_blood_sugar'] = int(rbs)
                high['urea'] = float(urea)
                high['worker'] = int(worker)
                high['diet'] = int(diet)
                high['bmi'] = round((float(weight) / (float(height)*float(height))) * 10000 ,2)
                high['Systolic'] = int(sys)
                high['Diastolic'] = int(dias)
                high['urine_albumin'] = int(ualb)
                # st.write(high["urine_albumin"])
                high['glomerular_filtration'] = int(gfr)
                changable = ['haemoglobin_level','total_water_intake','random_blood_sugar','Systolic','Diastolic','urine_albumin','living_area' , 'bmi','worker']

                hypothetical_data = pd.DataFrame()
                # recursion_count = 0
                # max_recursions = 20
                # # st.write(high.head(1))
                # hypothetical_patient_data = random_value_generator(high.iloc[0],recursion_count,max_recursions)
                # # st.write(hypothetical_patient_data)
                # hypothetical_data = hypothetical_data.append(hypothetical_patient_data, ignore_index=True)

                recursion_count = 0
                max_recursions = 20
                hypothetical_patient_data1 = random_value_generator(high.iloc[0],recursion_count=recursion_count,max_recursions=max_recursions)
                # st.write(hypothetical_patient_data1)
                hypothetical_patient_data = pd.DataFrame(hypothetical_patient_data1,index=[0])
                # hypothetical_data = hypothetical_data.append(hypothetical_patient_data, ignore_index=True)
                hypothetical_data = pd.concat([hypothetical_data,hypothetical_patient_data],ignore_index=True)
                # st.write(hypothetical_data)
                hypogfr = model.predict(hypothetical_data[['gender','age','diabetes','serum_creatinine_level','haemoglobin_level','total_water_intake','living_area','total_count','random_blood_sugar','urea','worker','diet','bmi','Systolic','Diastolic','urine_albumin']])

                hypothetical_data['hypogfr'] = hypogfr
                d1 = high.iloc[0]
                # st.write(d1)
                d2 = hypothetical_data.iloc[0]
                # st.write(d2)
                for k in changable:
                    #   st.write(k)
                    if(k!='diet' and k!='worker' and k!='living_area'):
                        if(k=='Systolic'):
                            if(d2[k]>140):
                                d2[k]=140
                        if(k=='Diastolic'):
                            if(d2[k]>100):
                                d2[k]=80
                            if(d2[k]<60):
                                d2[k]=60
                        dif = d1[k] - d2[k]
                        st.write("<p class='vbig-font'><u><b>"+k+" : </b></u>",unsafe_allow_html=True)
                        st.write("<p class='big-font'>Current - "+str(d1[k]),unsafe_allow_html=True)
                        #   st.write(f"Current - {k}",str(d1[k]))
                        st.write("<p class='big-font'>Approximate - "+str(d2[k]),unsafe_allow_html=True)
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

                st.write("<b>"+str(d1['glomerular_filtration'])+"</b> is the Actual GFR",unsafe_allow_html=True)
                # st.write()
                if(d2['hypogfr']>d1['glomerular_filtration']+15.0):
                    st.write("<b>"+str(d1['glomerular_filtration']+15.0)+"</b> is the hypothetical GFR",unsafe_allow_html=True)
                else:
                    st.write("<b>"+str(d2['hypogfr'])+"</b> is the hypothetical GFR",unsafe_allow_html=True)
            else:
                
                st.error("Check all fields")
        with colv2:
            # if(flag!=0):
            st.write("###")
            st.write("<p class='big-font'><b><u>Medicines that could help the GFR increase for this patient:<b><u>",unsafe_allow_html=True)
            mmm = retrieve_sugg_meds(convert_to_string(med))
            for qq in mmm:
                st.write("* "+qq)
    # st.write("Medicines that could help the GFR increase for this patient:")
    # mmm = retrieve_sugg_meds(patient_id)
    # for qq in mmm:
    #   st.write("* "+qq)
except:
    st.warning("(⊙_⊙)❗ Check the Input contains whether any Alphabetical Letters")
