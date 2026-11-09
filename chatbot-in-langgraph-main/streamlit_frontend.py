import uuid

import streamlit as st
from langchain_core.messages import HumanMessage

from langgraph_backend import chatbot

MAX_INPUT_LENGTH = 2_000
MAX_REQUESTS_PER_SESSION = 30

st.set_page_config(page_title="Chatbot", page_icon="💬")
st.title("Chatbot")
st.caption("A personal chatbot built with Streamlit and LangGraph")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "request_count" not in st.session_state:
    st.session_state.request_count = 0

for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message")

if user_input:
    if len(user_input) > MAX_INPUT_LENGTH:
        st.error(f"Please limit your message to {MAX_INPUT_LENGTH:,} characters.")
        st.stop()

    if st.session_state.request_count >= MAX_REQUESTS_PER_SESSION:
        st.warning("This demo has reached its message limit for this session.")
        st.stop()

    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.write(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )
                ai_message = response["messages"][-1].content
                st.write(ai_message)

        st.session_state.message_history.append(
            {"role": "assistant", "content": ai_message}
        )
        st.session_state.request_count += 1
    except Exception:
        st.error("The chatbot is temporarily unavailable. Please try again later.")
