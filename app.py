import streamlit as st
from transformers import pipeline
import torch

@st.cache_resource
def load_generator(model_path: str):
    return pipeline(
        "text-generation",
        model=model_path,
        tokenizer=model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device=0 if torch.cuda.is_available() else -1,
    )

st.set_page_config(page_title="MovieBot", page_icon="🎬")
st.title("🎬 MovieBot – LLM IMDB Recommendation")
st.write("Un manque d'inspiration ? Demande à MovieBot de te suggérer un film en fonction de tes envies !")

model_options = {
    "distilgpt2 – corpus brut": "./distilgpt2-imdb-finetuned",
    "distilgpt2 – instructions": "./distilgpt2-imdb-instructions-finetuned",
    "gpt2-medium – LoRA": "./gpt2-imdb-lora",
}

model_label = st.sidebar.selectbox(
    "Choisis le modèle",
    list(model_options.keys()),
    index=1, 
)

MODEL_PATH = model_options[model_label]
st.sidebar.write(f"Modèle chargé : `{MODEL_PATH}`")

generator = load_generator(MODEL_PATH)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Écris ta question ...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            conditioned_prompt = f"Instruction: {prompt}\nRéponse:"
            outputs = generator(
                conditioned_prompt,
                max_length=256,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                repetition_penalty=1.2,
                num_return_sequences=1,
                pad_token_id=generator.tokenizer.eos_token_id,
            )
            full_text = outputs[0]["generated_text"]
            answer = full_text.split("Réponse:")[-1].strip()
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
