import pickle
import string
import streamlit as st
import webbrowser


global Lrdetect_Model

LrdetectFile = open('model.pkl','rb')
Lrdetect_Model = pickle.load(LrdetectFile)
LrdetectFile.close()

st.title("Yoruba Language Detection Tool")
input_test = st.text_input("Provide your text input here", 'Mofẹ́ pààrọ̀ gílóòbù iná')


# if model_pipe.predict([input_test]) != 'Yoruba' or model_pipe.predict(['input_test']) != 'English':
#     st.text('Error: Not a yoruba or English text')

button_clicked = st.button("Get Language Name")
if button_clicked:
    st.text(Lrdetect_Model.predict([input_test]))
    
# if Lrdetect_Model.predict([input_test]) != 'Yoruba':
#     error = "Not a Yoruba Text, Please try another"
#     st.text(error)
    
    
# voice_button_clicked = st.button("Start Voice")
  