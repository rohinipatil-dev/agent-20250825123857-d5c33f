import os
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # stores {"role": "user"|"assistant", "content": str}
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2


def build_system_messages() -> List[Dict[str, str]]:
    # First system message as required by spec
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    # Additional instruction to focus on Python programming
    messages.append({
        "role": "system",
        "content": (
            "You specialize in answering Python programming questions. "
            "Provide clear explanations and concise examples when helpful. "
            "If a question is unrelated to Python programming, politely steer the user back to Python topics."
        ),
    })
    return messages


def build_chat_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    # Combines system messages with conversation history
    messages = build_system_messages()
    messages.extend(history)
    return messages


def generate_response(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    response = client.chat.completions.create(
        model=model,  # "gpt-4" or "gpt-3.5-turbo"
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def sidebar_controls() -> None:
    st.sidebar.header("Settings")
    st.session_state.model = st.sidebar.selectbox(
        "Model",
        options=["gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="Choose the OpenAI model used to answer your Python questions.",
    )
    st.session_state.temperature = st.sidebar.slider(
        "Creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values make answers more precise and deterministic.",
    )
    if st.sidebar.button("Start new chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.sidebar.markdown("Examples:")
    st.sidebar.button("How do I use list comprehensions?", on_click=append_example, args=("How do I use list comprehensions in Python?",))
    st.sidebar.button("What is a Python generator?", on_click=append_example, args=("What is a generator in Python and when should I use one?",))
    st.sidebar.button("Explain async/await", on_click=append_example, args=("Explain how async and await work in Python with an example.",))


def append_example(example: str) -> None:
    st.session_state.messages.append({"role": "user", "content": example})


def render_chat(history: List[Dict[str, str]]) -> None:
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def main() -> None:
    st.set_page_config(page_title="Python Programming Chatbot", page_icon="🐍", layout="centered")
    st.title("🐍 Python Programming Chatbot")
    st.caption("Ask questions about Python programming. This app uses OpenAI's chat models via the official SDK.")

    init_session_state()
    sidebar_controls()

    # Validate API key presence
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Please set the OPENAI_API_KEY environment variable to use this app.")
        st.stop()

    client = OpenAI()

    # Render existing conversation
    if st.session_state.messages:
        render_chat(st.session_state.messages)
    else:
        with st.chat_message("assistant"):
            st.markdown(
                "Hello! Ask me anything about Python programming: syntax, libraries, best practices, debugging, and more."
            )

    # Chat input
    user_input = st.chat_input("Type your Python question here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    messages = build_chat_messages(st.session_state.messages)
                    answer = generate_response(
                        client=client,
                        model=st.session_state.model,
                        messages=messages,
                        temperature=st.session_state.temperature,
                    )
                    st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error while generating response: {e}")


if __name__ == "__main__":
    main()