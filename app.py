import streamlit as st
import pandas as pd
import pickle




st.title("Logistic Regression")
st.markdown("<hr style=margin-top:-10px margin-bottom:20px>",unsafe_allow_html = True)
st.markdown("### Titanic Crash Survival Chances")
st.markdown("#### Passenger Details")
col1,col2 = st.columns([0.15,0.85])
with col1:
	title = st.selectbox("Title",["Mr","Mrs","Miss","Master","Dr","Sir","Major","Lady","Capt"])
with col2:
	name = st.text_input("Name")
col3,col4,col5 = st.columns([0.33,0.33,0.34])
with col3:
	age = st.number_input("Age",format = "%d",step = 1,min_value = 0,max_value = 120)
with col4:
	sex = st.selectbox("Gender",[" ","Male","Female"])
with col5:
	pclass = st.selectbox("P-Class",[" ",1,2,3])
col6,col7 = st.columns([0.5,0.5])
with col6:
	sibsp = st.number_input("Siblings/Spouse",format = "%d",step = 1)
with col7:
	parch = st.number_input("Parent/Children",format = "%d",step = 1)
col8,col9,col10 = st.columns([0.20,0.20,0.6])
with col8:
	cabin = st.selectbox("Cabin Code",[" ","Unknown","A","B","C","D","E","F","G","T"])
with col9:
	embarked = st.selectbox("Embarked",[" ","S","C","Q"])
with col10:
	fare = st.number_input("Fare",format = "%.2f")
col11,col12 = st.columns([0.88,0.12])
with col12:
	submit = st.button("Submit")

features = {"title":title,"Age":age,"Sex":sex,"Pclass":pclass,"SibSp":sibsp,"Parch":parch,"cabin code":cabin,"Embarked":embarked,"Fare":fare}
df = pd.DataFrame(features,index= [0])


st.write(f"""Description: 

Survived - 1, 
Not Survived - 0 """)
if(submit == True):
	model = pickle.load(open("lr.pkl","rb"))
	res = model.predict(df)
	if(res == 0):
		st.write(f"{title}.{name} will not be Survived")
	else:
		st.write(f"{title}.{name} will be Survived")


