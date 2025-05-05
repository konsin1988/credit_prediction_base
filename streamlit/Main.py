import streamlit as st
import requests as rq

st.title('German credit prediction')
st.subheader('Paste your data here:')

col1, col2, col3 = st.columns(3)

age = col1.selectbox('Age', [i for i in range(18, 70)])

sex = col1.selectbox('Gender', ['male', 'female'])
job = col2.selectbox('Job skilled', ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled'])
housing = col2.selectbox('Housing owner', ['own', 'rent', 'free'])
credit_amount = col3.text_input('Loan amount', 2000)
duration = col3.selectbox('Duration', [i for i in range(1, 72)])

input_data = {
    'age': age,
    'sex': sex,
    'job': job,
    'housing': housing,
    'credit_amount': credit_amount,
    'duration': duration
}

# st.text(rq.post('http://fastapi:8000/predict', json=input_data))

prediction = rq.post('http://fastapi:8000/predict', json=input_data).json()['prediction']

st.subheader('ML prediction:')
st.markdown(f"Person **{age}** years old, **{sex}**, **{job}**, have **{housing}** house who wants to get loan amount **{credit_amount}** USD on **{duration}** month, - is **{prediction}**")