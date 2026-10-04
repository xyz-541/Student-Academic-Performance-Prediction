import streamlit as st
import pandas as pd
import joblib
import numpy as np

PROJECT = "/home/mayank/Desktop/Student_Performance_ML"
ARTIFACT = joblib.load(PROJECT + "/student_performance_model.joblib")
PREPROCESSOR = ARTIFACT["preprocessor"]
SELECTOR = ARTIFACT["selector"]
MODEL = ARTIFACT["model"]

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Student Performance Predictor")
st.write("Enter basic student details, attendance and subject marks to predict academic performance.")

st.header("Student Details")
c1, c2 = st.columns(2)
with c1:
    student_name = st.text_input("Student Name", "Rahul Sharma")
    father_name = st.text_input("Father Name", "Rajesh Sharma")
    mother_name = st.text_input("Mother Name", "Sunita Sharma")
with c2:
    age = st.number_input("Age", 15, 25, 18)
    total_classes = st.number_input("Total Classes", 1, 300, 100)
    present_classes = st.number_input("Present Classes", 0, 300, 90)

st.header("Subject Marks")
st.caption("Each subject has a total of 100 marks. Total marks for 5 subjects = 500.")
m1, m2 = st.columns(2)
with m1:
    maths = st.number_input("Mathematics (100)", 0, 100, 70)
    physics = st.number_input("Physics (100)", 0, 100, 70)
    chemistry = st.number_input("Chemistry (100)", 0, 100, 70)
with m2:
    english = st.number_input("English (100)", 0, 100, 70)
    computer = st.number_input("Computer Science (100)", 0, 100, 75)

if st.button("🔮 Predict Student Performance", type="primary"):
    attendance = (present_classes / total_classes) * 100 if total_classes else 0
    total_obtained = maths + physics + chemistry + english + computer
    total_possible = 500
    percentage = (total_obtained / total_possible) * 100
    average_marks = total_obtained / 5
    estimated_grade = average_marks / 5
    G1 = estimated_grade
    G2 = estimated_grade
    absences = max(0, int(total_classes - present_classes))

    data = {
        "school": "GP", "sex": "M", "age": age, "address": "U",
        "famsize": "GT3", "Pstatus": "T", "Medu": 2, "Fedu": 2,
        "Mjob": "other", "Fjob": "other", "reason": "course",
        "guardian": "mother", "traveltime": 1, "studytime": 2,
        "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes",
        "internet": "yes", "romantic": "no", "famrel": 4, "freetime": 3,
        "goout": 3, "Dalc": 1, "Walc": 1, "health": 4,
        "absences": absences, "G1": G1, "G2": G2
    }

    X = pd.DataFrame([data])
    X_processed = PREPROCESSOR.transform(X)
    X_selected = SELECTOR.transform(X_processed)
    prediction = float(MODEL.predict(X_selected)[0])
    prediction = float(np.clip(prediction, 0, 20))
    predicted_marks = prediction * 5

    predicted_percentage = predicted_marks

    if predicted_percentage >= 80:
        grade = "A"
    elif predicted_percentage >= 60:
        grade = "B"
    elif predicted_percentage >= 40:
        grade = "C"
    elif predicted_percentage >= 20:
        grade = "D"
    else:
        grade = "E"

    st.success(f"Predicted Performance: {predicted_percentage:.1f} / 100")
    st.metric("Predicted Grade", grade)
    st.metric("Entered Total Marks", f"{total_obtained} / {total_possible}")
    st.metric("Total Percentage", f"{percentage:.1f}%")
    st.metric("Attendance", f"{attendance:.1f}%")
    st.write(f"**Student:** {student_name}")
    st.write(f"**Father:** {father_name}  |  **Mother:** {mother_name}")
    st.write(f"**Average entered marks:** {average_marks:.1f} / 100")

    with st.expander("ML Prediction (Research Model)"):
        st.metric("ML Predicted Performance", f"{predicted_percentage:.1f}%")
        st.caption("This is the research model prediction. The student report grade above is calculated directly from the entered marks.")
        st.caption("Grade scale: A = 80-100, B = 60-79, C = 40-59, D = 20-39, E = below 20.")
        st.caption("Names are displayed for the student record only. The ML prediction uses academic/attendance inputs, not names.")
