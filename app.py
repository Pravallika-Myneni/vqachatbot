import torch 
import streamlit as st
from streamlit_chat import message
from captions import infer
# !pip install transformers --q
MAX_LEN = 200

from transformers import T5ForConditionalGeneration, T5Tokenizer
from QA import QAModel
col1, col2 = st.columns( [0.25, 0.75])
tokenizer = T5Tokenizer.from_pretrained('t5-base')
trained_model = QAModel.load_from_checkpoint('/content/checkpoints/best-checkpoint.ckpt')

def get_image():

    uploaded_img = st.file_uploader("Choose an image", type = ['png', 'jpg', 'jpeg'])
    if uploaded_img is not None:
        img = uploaded_img.read()
        return img 
    else:
        return None

def generate_image_caption(img):
    caption = infer(img)
    return caption

def get_question():
    input_text = st.text_input("Enter your question here","how many items are there", key="input")
    return input_text 

def get_answer(question, context):
  input = f"question: {question} context: {context}"
  source_encoding = tokenizer(input, max_length = MAX_LEN, pad_to_max_length = True, padding = "max_length", return_attention_mask = True, add_special_tokens = True, return_tensors = "pt")
  generated_ids = trained_model.model.generate(
      input_ids = source_encoding['input_ids'],
      attention_mask = source_encoding['attention_mask'],
      num_beams = 1,
      max_length = 80, 
      repetition_penalty = 2.5, 
      early_stopping = True,
      use_cache = True
  )
  preds = [tokenizer.decode(gen_id, skip_special_tokens = True, clean_up_tokenization_spaces = True) for gen_id in generated_ids]
  return " ".join(preds)


with col2:
    message("Welcome to VQA chatbot!! Upload image")
    img = get_image()
if img:
    st.image(img)

    captions_generated = generate_image_caption(img)
    print(captions_generated)

    with col1:
        qn = get_question()

    with col2:
        message(qn, is_user=True)
        if qn:
            output = get_answer(img, qn)
            message(output)




