import torch 
import streamlit as st
from streamlit_chat import message


col1, col2 = st.columns( [0.25, 0.75])

def get_image():
    uploaded_img = st.file_uploader("Choose an image", type = ['png', 'jpg'])
    if uploaded_img is not None:
        img = uploaded_img.read()
        return img 
    else:
        return None

def generate_image_caption(img):    
    pass

def get_question():
    input_text = st.text_input("Enter your question here","how many items are there", key="input")
    return input_text 

def get_answer(img, qn):
    return "IMGGGG"



with col2:
    message("Welcome to VQA chatbot!! Upload image")
    img = get_image()
if img:
    st.image(img)

    captions_generated = generate_image_caption(img)

    with col1:
        qn = get_question()

    with col2:
        message(qn, is_user=True)
        if qn:
            output = get_answer(img, qn)
            message(output)




