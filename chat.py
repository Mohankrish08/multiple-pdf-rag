import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "CohereForAI/c4ai-command-r-plus"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, num_beams=1, early_stopping=False):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        early_stopping=early_stopping,
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Streamlit app
st.title("Custom LLM Model")
st.write("This app uses a pre-trained GPT-2 model to generate text.")

prompt = st.text_area("Enter a prompt:", height=100)
if st.button("Generate Text"):
    generated_text = generate_text(prompt)
    st.write("Generated text:", generated_text)

