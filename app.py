import pickle
import string
import streamlit as st
import webbrowser



global Lrdetect_Model

LrdetectFile = open('model.pkl','rb')
Lrdetect_Model = pickle.load(LrdetectFile)
LrdetectFile.close()
st.title("Yoruba Language Detection Tool")
input_test = st.text_input("Provide your text input here", 'fẹ́ pààrọ̀ gílóòbù iná tó wà')

button_clicked = st.button("Get Language Name")
if button_clicked:
    st.text(Lrdetect_Model.predict([input_test]))
    
voice_button_clicked = st.button("Start Voice")
  