import streamlit as st


DEFAULT_SESSION_KEYS = {
    "messages": [],
    "chat_session": None,
    "last_result": None,
    "explanations": {},
    "saved_file_path": None,
    "uploaded_file_name": None,
    "last_loaded_path": None,
    "df": None,
    "query_suggestions": [],
    "suggestions_generated": False,
    "processing_suggestion": False,
}


def ensure_defaults() -> None:
    for key, default_value in DEFAULT_SESSION_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


