import streamlit as st
import pickle
import numpy as np

with open(f"./model/model.pkl", "rb") as f:
    model = pickle.load(f)

def get_yes_no_question(yes_no):
    yes_no_mapping = {
        "No" : 0,
        "Yes" : 1
    }
    return yes_no_mapping.get(yes_no, -1)

def get_gender(gender):
    gender_mapping = {
        "Female" : 0,
        "Male" : 1
    }
    return gender_mapping.get(gender, -1)

status = {
    0 : "Dropout",
    1 : "Graduate"
}

def predict_status(_sem_enrolled, _scholarship_holder, _sem_approved, _sem_credited, 
                _tuition_fees, _sem_evaluations, _gender, _debt):
    scholarship_holder = get_yes_no_question(_scholarship_holder)
    tuition_fees = get_yes_no_question(_tuition_fees)
    gender = get_gender(_gender)
    debt = get_yes_no_question(_debt)
    data = np.array[[debt, _sem_approved, _sem_evaluations, _sem_credited, 
                    _sem_enrolled, scholarship_holder, 
                    tuition_fees, gender]]
    prediction = model.predict(data)[0]
    prediction_proba = model.predict_proba(data)[0][prediction] * 100
    if prediction == 0:
        return f"The student might {status[prediction]}, model confidence is {prediction_proba:.2f}%"
    if prediction == 1:
        return f"The student should {status[prediction]}, model confidence is {prediction_proba:.2f}%"



st.title("Student Status Prediction")

st.markdown("""
            # 🎒 Student Status Prediction
            # Dicoding - Solving Educational Institution Problem
            ## Made by : Muhammad Hafizh Dzaki
            ## Gihub Repo : [Here](https://github.com/hfzdzakii/Dicoding-SolvingEducationIntsituteProblem)
            """)

col1, col2 = st.columns(2)

with col1:
    st.header("Input Variables")
    sem_approved = st.number_input(label="Sum of 2nd Semester Curricular Units Approved:",
                                   value=0, min_value=0, max_value=24)
    sem_evaluations = st.number_input(label="Sum of 2nd Semester Curricular Units Evalutions:",
                                      value=0, min_value=0)
    sem_credited = st.number_input(label="Sum of 2nd Semester Curricular Units Credited:",
                                   value=0, min_value=0, max_value=24)
    sem_enrolled = st.number_input(label="Sum of 2nd Semester Curricular Units Enrolled:",
                                   value=0, min_value=0, max_value=24)
    debt = st.radio(label="Having Debt?",
                    options=["No", "Yes"], index=0)
    scholarship_holder = st.radio(label="Scholarship Holder?",
                                  options=["No", "Yes"], index=0)
    tuition_fees = st.radio(label="Tuition Fees Payed?",
                            options=["No", "Yes"], index=0)
    gender = st.radio(label="Gender:",
                      options=["Male", "Female"], index=0)
    
with col2:
    st.header("Predict and Result")
    if st.button("Predict", type="primary"):
        result = predict_status(sem_enrolled, scholarship_holder,
                                sem_approved,
                                sem_credited, tuition_fees,
                                sem_evaluations, gender, debt)
        st.text_area("Prediction", value=result, height=100)
        