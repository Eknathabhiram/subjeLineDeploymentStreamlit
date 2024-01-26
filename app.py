import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer,AutoModelForCausalLM

# Load the model and tokenizer
model_path = "/AbhiramPemmaraju/gpt_model"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
extra_tokens = ["<email>", "<subject>"]
tokenizer.add_tokens(extra_tokens)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = AutoModelForCausalLM.from_pretrained("AbhiPemmaraju/gpt_model")


# Set up the device (use GPU if available, otherwise use CPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# Streamlit app
st.title("GPT-2 Text Generation")

# Input prompt from user
prompt = st.text_area("Enter your prompt:")

# Generate button
if st.button("Generate"):
    # Tokenize input and generate output
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        input_ids,
        max_length=1024,
        num_return_sequences=1,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
    )

    # Decode and display the generated text
    if output_ids is not None and output_ids[0] is not None:
        # Decode and display the generated text
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.text("Generated Text:")
        st.write(generated_text)
    else:
        st.text("Error: Failed to generate text. Please try again.")
