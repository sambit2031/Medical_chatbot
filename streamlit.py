"""
Streamlit frontend for the Generative AI Medical Chatbot.

It talks to the FastAPI backend (app.py) over HTTP.

Run the backend first:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then run this UI:
    streamlit run streamlit_app.py
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺")
st.title("🩺 Medical Chatbot")
st.caption("Ask a medical question. Answers are generated from the knowledge base.")

# Keep chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if user_input := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"message": user_input},
                    timeout=60,
                )
                resp.raise_for_status()
                answer = resp.json().get("answer", "No answer returned.")
            except requests.exceptions.RequestException as e:
                answer = f"Error contacting backend: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

with st.sidebar:
    st.markdown("### Settings")
    st.write(f"**Backend:** {API_URL}")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.warning("This chatbot is for informational purposes only and is not a "
               "substitute for professional medical advice.")
