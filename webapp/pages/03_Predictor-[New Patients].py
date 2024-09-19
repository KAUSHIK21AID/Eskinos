import time
import datetime
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sys
sys.path.append("helpers/")
from Pipeline import AddingAttributes,Model
from sklearn.preprocessing import PolynomialFeatures
st.set_page_config(page_title="Predictor")

# model = pkl.load(open('model.pkl','rb'))

conn = st.connection('mysql', type='sql')
@st.cache_data
def listing(df,age):
    temp = df[(df["age"] == age) ]
    # if(len(temp) == 1): return -1
    bmi = []
    scl = []
    hl = []
    rbs = []
    tc = []
    sys = []
    dias=[]
    #print(temp["id"].unique())
    for i in temp["id"].unique():
        diff_bmi=[]
        diff_scl = []
        diff_hl=[]
        diff_rbs = []
        diff_tc=[]
        diff_sys =[]
        diff_dias =[]
        temp_df = temp.groupby("id").get_group(i).reset_index()
        for ind in range(len(temp_df)-1):
            diff_bmi.append(round(temp_df["bmi"][ind+1] - temp_df["bmi"][ind],2))
            diff_scl.append(round(temp_df["serum_creatinine_level"][ind+1] - temp_df["serum_creatinine_level"][ind],2))
            diff_hl.append(round(temp_df["haemoglobin_level"][ind+1] - temp_df["haemoglobin_level"][ind],2))
            diff_rbs.append(round(temp_df["random_blood_sugar"][ind+1] - temp_df["random_blood_sugar"][ind],2))
            diff_tc.append(round(temp_df["total_count"][ind+1] - temp_df["total_count"][ind],2))
            diff_sys.append(round(temp_df["systolic"][ind+1] - temp_df["systolic"][ind],2))
            diff_dias.append(round(temp_df["diastolic"][ind+1] - temp_df["diastolic"][ind],2))
        # print("bmi ",i,"=",diff_bmi);print("\n")
        # print("Creatinine ",i,"=",diff_scl);print("\n")
        # print("Haemoglobin ",i,"=",diff_hl);print("\n")
        # print("random_blood_sugar ",i,"=",diff_rbs);print("\n")
        # print("total_count ",i,"=",diff_tc);print("\n")
        # print("systolic ",i,"=",diff_sys);print("\n")
        # print("diastolic ",i,"=",diff_dias);print("\n")
        # st.write(diff_bmi)
        if(len(diff_bmi)>=1):
            bmi.append(round(sum(diff_bmi)/len(diff_bmi),2))
            scl.append(round(sum(diff_scl)/len(diff_scl),2))
            hl.append(round(sum(diff_hl)/len(diff_hl),2))
            rbs.append(round(sum(diff_rbs)/len(diff_rbs),2))
            tc.append(round(sum(diff_tc)/len(diff_tc),2))
            sys.append(round(sum(diff_sys)/len(diff_sys),2))
            dias.append(round(sum(diff_dias)/len(diff_dias),2))
        # st.write(diff_bmi)
    # st.write("first")
    # st.write([bmi,scl,hl,rbs,tc,sys,dias])
    bmi1 = round(sum(bmi)/len(bmi),2)
    # st.write(bmi1)
    scl1 = round(sum(scl)/len(scl),2)
    hl1 = round(sum(hl)/len(hl),2)
    rbs1 = round(sum(rbs)/len(rbs),2)
    tc1 = round(sum(tc)/len(tc),2)
    sys1 = round(sum(sys)/len(sys),2)
    dias1 = round(sum(dias)/len(dias),2)
    # st.write([bmi1,scl1,hl1,rbs1,tc1,sys1,dias1])
    return [bmi1,scl1,hl1,rbs1,tc1,sys1,dias1]

def validate_inputs(age, height , weight, t_hb, t_scr, t_tc, t_sys, t_di, t_rbs, t_tw, t_la, t_cvd, t_pkd, t_kih, t_db):
    errors = []

    # Perform validation for each input field
    if not 0 <= int(age) <= 99:
        errors.append("Age must be between 0 and 99.")

    if not 100 <= float(height) <= 195: # remove bmi instead calc
        errors.append("height must be between 100cm and 195cm")

    if not 20 <= float(weight) <= 210: # remove bmi instead calc
        errors.append("weight must be between 20kg and 210kg")

    if not 0 <= float(t_hb) <= 16:
        errors.append("Haemoglobin Level must be between 0 and 16.")

    if not 0 <= float(t_scr) <= 30:
        errors.append("Serum Creatinine Level must be between 0 and 30.")

    if not 1000 <= float(t_tc) <= 20000:
        errors.append("Total Count must be between 1000 and 20000.")

    if not 0 <= int(t_sys) <= 210:
        errors.append("Systolic must be between 0 and 210.")

    if not 0 <= int(t_di) <= 130:
        errors.append("Diastolic must be between 0 and 130.")

    if not 0 <= float(t_rbs) <= 500:
        errors.append("Random Blood Sugar must be between 0 and 500.")

    if not 1 <= float(t_tw) <= 6:
        errors.append("Total Water Intake must be between 1 and 6.")

    if not 0 <= int(t_la) <= 1:
        errors.append("Living Area must be either 0 or 1.")

    # You can continue adding validation for the remaining fields...

    return errors

st.markdown(
        """
        <style>
            div[data-testid=stToast] {
                padding:  20px 10px;
                margin: 10px 400px;
                width: 30%;
            }
             
            [data-testid=toastContainer] [data-testid=stMarkdownContainer] > p {
                font-size: 20px; font-style: normal; font-weight: 350;
                foreground-color: #ffffff;
            }
        </style>
        """, unsafe_allow_html=True
    )
df = conn.query('SELECT * from finaliti',ttl=0)
df = AddingAttributes(df)
age_l = sorted(list(df['age'].unique()))
agedf = df.groupby('age')
st.markdown("<h2 style='text-align:center;colour:white;'><u>Renal Prognosticator: Predicting GFR for New Patient</u></h2>",unsafe_allow_html=True)

st.info("It'll predict the GFR for next 3 years with the patient's current test results")

col1,col2 = st.columns([5,5])
with col1:
    col2_1,col2_2 = st.columns([5,5])
    with col2_1:
        age = st.text_input(label="Age")
    with col2_2:
        gfr = st.text_input(label="GFR")
    col1_1,col1_2 = st.columns([5,5])
    with col1_1:
        height = st.text_input("height")
    with col1_2:
        weight = st.text_input("weight")
    t_hb = (st.text_input('haemoglobin_level'))
    t_scr = (st.text_input('serum_creatinine_level'))
    t_tc = (st.text_input('total_count'))
    t_sys = (st.text_input('systolic'))
    t_di = (st.text_input('diastolic'))
with col2:
    t_rbs = (st.text_input('random_blood_sugar'))
    t_tw = (st.text_input('Total Water Intake'))
    t_la = (st.selectbox('living_area',options=[0,1],help="0 - Rular,1 - Urban"))
    t_cvd = (st.selectbox(options=[0,1],help="0 - No , 1 - Yes",label='Cardio Vascular Disease'))
    t_pkd = (st.selectbox(options=[0,1],help="0 - No , 1 - Yes",label='previous_kidney_disease'))
    t_kih = (st.selectbox(options=[0,1],help="0 - No , 1 - Yes",label='kidney_injury_history'))
    t_db = (st.selectbox(options=[0,1],help="0 - No , 1 - Yes",label='diabetes'))
col3,col4,col5 = st.columns([4,2,4])
with col4:
    result = st.button(label="Predict",type='primary',help="It will predict the GFR for Next year",use_container_width=True)

# with open('model.pkl', 'rb') as f:
#     u = pkl._Unpickler(f)
#     u.encoding = 'latin1'
#     p=u.load()
# model = np.load('model.pkl',allow_pickle=True)
ml,score,unique_patients = Model(df=df)
#st.write(unique_patients)
cols1 = [age,height ,weight,t_hb,t_scr,t_tc,t_sys,t_di,t_rbs,t_tw,t_la,t_cvd,t_pkd,t_kih,t_db]
try:
    if result:
        correct = False
        flag=False
        i=""
        if i in cols1:
            flag=True
            # break
        if(flag):
            st.error("Check All the Fields have filled?")
        elif(not flag):
            errors = validate_inputs(age, height , weight, t_hb, t_scr, t_tc, t_sys, t_di, t_rbs, t_tw, t_la, t_cvd, t_pkd, t_kih, t_db)
            if errors:
                st.error("Validation failed:")
                for error in errors:
                    st.error(error)
            else:
                correct = True
        if(correct):
            age = float(age)
            t_bmi = round((float(weight) / (float(height)*float(height)))*10000,2)
            t_hb = float(t_hb)
            t_scr = float(t_scr)
            t_tc  = float(t_tc)
            t_sys = float(t_sys)
            t_di  = float(t_di)
            t_rbs = float(t_rbs)
            t_tw  = float(t_tw)
            t_la  = float(t_la)
            t_cvd = float(t_cvd)
            t_pkd = float(t_pkd)
            t_kih = float(t_kih)
            t_db  = float(t_db)
            while(age not in age_l):
                age = age+1
            # temp = agedf.get_group(age)
            try:
                while(True): 
                    if(not(len(df[df["age"]==age]) > 2)):
                        age+=1
                    else:break
                    # st.write(age)
                fl = listing(df,age)
                bmi = fl[0]
                scr = fl[1]
                hl = fl[2]
                rbs = fl[3]
                tc = fl[4]
                sys = fl[5]
                dias = fl[6]

                newbmi = bmi + t_bmi
                newhb =  hl  + t_hb
                newscr = scr + t_scr
                newtc = tc + t_tc
                newsys = sys + t_sys
                newdi = dias + t_di
                newrbs = rbs + t_rbs
                sample = pd.DataFrame({
                    'bmi':[newbmi],
                    'serum_creatinine_level':[newscr],
                    'haemoglobin_level':[newhb],
                    'age':[age],
                    'random_blood_sugar':[newrbs],
                    'total_count':[newtc],
                    'systolic':[newsys],
                    'diastolic':[newdi],
                    'diabetes':[t_db],
                    'total_water_intake':[t_tw],
                    'living_area':[t_la],
                    'cardiovascular_disease':[t_cvd],
                    'previous_kidney_disease':[t_pkd],
                    'kidney_injury_history':[t_kih]
                    })
                pred = ml.predict(sample)
                # st.write("Before",age)
                age = age+1
                while(age not in age_l):
                    age = age+1
                # st.write("/After",age)
                # st.write(age_l)
                while(True): 
                    if(not(len(df[df["age"]==age]) > 2)):
                        age+=1
                    else:break
                    # st.write(age)
                fl_2 = listing(df,age)
                # if (fl_2 == -1):
                #     st.write("From 2")
                #     temp = df[df["age"]==age]
                #     st.write(temp)
                #     samp = pd.DataFrame({
                #         'bmi':temp["bmi"],
                #         'serum_creatinine_level':temp["serum_creatinine_level"],
                #         'haemoglobin_level':temp["haemoglobin_level"],
                #         'age':[age],
                #         'random_blood_sugar':temp["random_blood_sugar"],
                #         'total_count':temp["total_count"],
                #         'systolic':temp["systolic"],
                #         'diastolic':temp["diastolic"],
                #         'diabetes':[t_db],
                #         'total_water_intake':temp["total_water_intake"],
                #         'living_area':[t_la],
                #         'cardiovascular_disease':[t_cvd],
                #         'previous_kidney_disease':[t_pkd],
                #         'kidney_injury_history':[t_kih]
                #         })
                #     pred2 = ml.predict(samp)
                # else:
                bmi2 = fl_2[0]
                scr2 = fl_2[1]
                hl2 = fl_2[2]
                rbs2 = fl_2[3]
                tc2 = fl_2[4]
                sys2 = fl_2[5]
                dias2 = fl_2[6]
                newbmi2 = bmi2 + sample['bmi'][0]
                newhb2 =  hl2  + sample['haemoglobin_level'][0]
                newscr2 = scr2 + sample['serum_creatinine_level'][0]
                newtc2 = tc2 + sample['total_count'][0]
                newsys2 = sys2 + sample['systolic'][0]
                newdi2 = dias2 + sample['diastolic'][0]
                newrbs2 = rbs2 + sample['random_blood_sugar'][0]
                sample2 = pd.DataFrame({
                    'bmi':[newbmi2],
                    'serum_creatinine_level':[newscr2],
                    'haemoglobin_level':[newhb2],
                    'age':[age],
                    'random_blood_sugar':[newrbs2],
                    'total_count':[newtc2],
                    'systolic':[newsys2],
                    'diastolic':[newdi2],
                    'diabetes':[t_db],
                    'total_water_intake':[t_tw],
                    'living_area':[t_la],
                    'cardiovascular_disease':[t_cvd],
                    'previous_kidney_disease':[t_pkd],
                    'kidney_injury_history':[t_kih]
                    })
                pred2 = ml.predict(sample2)
                age=age+1
                # st.write(pred2)
                # st.write("Before",age)
                while(age not in age_l):
                    age = age+1
                # st.write("After",age)
                # st.write(age_l)
                while(True): 
                    if(not(len(df[df["age"]==age]) > 2)):
                        age+=1
                    else:break
                    # st.write(age)
                fl_3 = listing(df,age)
                # if(fl_3 == -1):
                #     while()
                    # temp = df[df["age"]==age]
                    # st.write("From 3")
                    # st.write(temp)
                    # samp = pd.DataFrame({
                    #     'bmi':temp["bmi"],
                    #     'serum_creatinine_level':temp["serum_creatinine_level"],
                    #     'haemoglobin_level':temp["haemoglobin_level"],
                    #     'age':[age],
                    #     'random_blood_sugar':temp["random_blood_sugar"],
                    #     'total_count':temp["total_count"],
                    #     'systolic':temp["systolic"],
                    #     'diastolic':temp["diastolic"],
                    #     'diabetes':[t_db],
                    #     'total_water_intake':temp["total_water_intake"],
                    #     'living_area':[t_la],
                    #     'cardiovascular_disease':[t_cvd],
                    #     'previous_kidney_disease':[t_pkd],
                    #     'kidney_injury_history':[t_kih]
                    #     })
                    # pred3 = ml.predict(samp)
                # else:
                bmi3 = fl_3[0]
                scr3 = fl_3[1]
                hl3 = fl_3[2]
                rbs3 = fl_3[3]
                tc3 = fl_3[4]
                sys3 = fl_3[5]
                dias3 = fl_3[6]
                newbmi3 = bmi3 + sample2['bmi'][0]
                newhb3 =  hl3  + sample2['haemoglobin_level'][0]
                newscr3 = scr3 + sample2['serum_creatinine_level'][0]
                newtc3 = tc3 + sample2['total_count'][0]
                newsys3 = sys3 + sample2['systolic'][0]
                newdi3 = dias3 + sample2['diastolic'][0]
                newrbs3 = rbs3 + sample2['random_blood_sugar'][0]
                sample3 = pd.DataFrame({
                    'bmi':[newbmi3],
                    'serum_creatinine_level':[newscr3],
                    'haemoglobin_level':[newhb3],
                    'age':[age+1],
                    'random_blood_sugar':[newrbs3],
                    'total_count':[newtc3],
                    'systolic':[newsys3],
                    'diastolic':[newdi3],
                    'diabetes':[t_db],
                    'total_water_intake':[t_tw],
                    'living_area':[t_la],
                    'cardiovascular_disease':[t_cvd],
                    'previous_kidney_disease':[t_pkd],
                    'kidney_injury_history':[t_kih]
                    })
                pred3 = ml.predict(sample3)
                
                today = datetime.date.today()
                current_year = today.year
                st.write(f"Predicted GFR for Year {current_year+1} - ",pred[0])
                st.write(f"Predicted GFR for  Year {current_year+2}- ",pred2[0])
                st.write(f"Predicted GFR for Year {current_year+3} - ",pred3[0])
                PercentDiff = int(gfr) * 0.044 * 3  # grf * (4.416/100)
                DiffInGfr = int(gfr) - pred3
                #st.write(PercentDiff)
                #st.write(f"The threshold value for this patient as per research {DiffInGfr[0]}")
                #st.write("Note : As per research , if the threshold value for Indian patients is 4.416% of initial GFR")
                st.write("Based on the research as per the threshold value for Indian patients ")
                #st.write("")
                #st.write(DiffInGfr)
                if PercentDiff < DiffInGfr : st.write("Patient is in Risk Condition") 
                else:st.write("Patient is not in Risk Condition")
                st.toast(f"Model Accuracy : {round(score,2)*100}% ",icon="✨")
                time.sleep(5)
                st.toast(f"The Predictor will evolve it's performances, when new data were added.",icon="🌟")
            except:
                st.toast("No data found in DB",icon="❗")
                st.info("Sorry! There is no data related to this age category, It will be resolved in the mean time while adding the data on this age category",icon="ℹ")
except:
    st.warning("(⊙_⊙)❗ Check the Input contains whether any Alphabetical Letters")
