import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai


def configure_environment() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path)

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if not api_key:
        try:
            api_key = (
                st.secrets.get("GEMINI_API_KEY")
                or st.secrets.get("GOOGLE_API_KEY")
            )
        except Exception:
            api_key = None

    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error(
            "No API key found. Please set GEMINI_API_KEY (or GOOGLE_API_KEY) in a .env at the project root or in Streamlit secrets."
        )


def configure_page() -> None:
    st.set_page_config(page_title="🧠 Data Analyzer Agent", layout="wide")


def get_gemini_model(model_name: str = "gemini-1.5-flash"):
    return genai.GenerativeModel(model_name)


