#Streamlit app

import pandas as pd
import streamlit as st
import pickle 

#page configuration
st.set_page_config(page_title="Loan Classification",page_icon="logo.jpg",layout ="wide")

with st.sidebar:
    st.title("Loan Classification")
    st.image("logo.jpg")

#load the model
with open ("best_model_pipeline.pkl","rb") as file:
    best_pipeline=pickle.load(file)

#load cleaned dataframe
df= pd.read_csv("cleaned_data.csv")



#user input
col1,col2,col3,col4=st.columns(4,border=True)
with col1:
    person_age=st.slider("Age:",min_value=1,max_value=110)
    person_income=st.slider("Income:",min_value=10000,max_value=1000000)
    person_emp_exp=st.slider("Experience:",min_value=0,max_value=100)
    loan_amnt=st.slider("Loan_amount:",min_value=30000,max_value=1000000)
with col2:
        person_gender=st.radio("Gender:",options=df["person_gender"].unique())
        loan_intent=st.selectbox("Loan_intent:",options=df["loan_intent"].unique())
        loan_int_rate=st.slider("Loan_int_rate:",min_value=6.0,max_value=200.0)
        loan_percent_income=st.slider("loan_percent_income:",min_value=0.0,max_value=1.0)
        cb_person_cred_hist_length=st.slider("credit_history_length:",min_value=1.0,max_value=30.0)
        credit_score=st.slider("credit_score:",min_value=300,max_value=600)
with col3:
        person_education=st.selectbox("Education:",options=df["person_education"].unique())
        person_home_ownership=st.selectbox("Home Ownership:",options=df["person_home_ownership"].unique())
        previous_loan_defaults_on_file=st.selectbox("loan_default:",options=df["previous_loan_defaults_on_file"].unique())


#new data
new_data={'person_age':person_age,
          'person_income':person_income,
            'person_emp_exp':person_emp_exp,
            'loan_amnt':loan_amnt,
            'person_gender':person_gender,
            'loan_intent':loan_intent,
            'loan_int_rate':loan_int_rate,
            'loan_percent_income':loan_percent_income,
            'cb_person_cred_hist_length':cb_person_cred_hist_length,
            'credit_score':credit_score,
            'person_education':person_education,
            'person_home_ownership':person_home_ownership,
            'previous_loan_defaults_on_file':previous_loan_defaults_on_file
           }
new_data=pd.DataFrame(new_data,index=[0])
with col4:
      if st.button("Predict loan status"):
            prediction=best_pipeline.predict(new_data)[0]
            if prediction == 1:
                st.success("The loan is approved.")
            else:
                st.error("The loan is not approved.")