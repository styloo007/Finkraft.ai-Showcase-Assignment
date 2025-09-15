import streamlit as st


CUSTOM_CSS = """
<style>
    .stAppViewContainer {background: #0b1220;}
    .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stText, .stHeader, .stSubheader, .stCode, .stTable { color: #e5e7eb; }
    h1, h2, h3, h4, h5, h6 { color: #e8eaf1 !important; }

    .suggestion-button {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; padding: 10px 15px; margin: 5px; border: none; cursor: pointer; transition: all 0.3s ease;}
    .suggestion-button:hover {transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15);}    
    .chat-container {background: #0f172a; border-radius: 14px; padding: 16px 18px; margin: 12px 0; border: 1px solid rgba(255,255,255,0.06);}    
    .suggestion-container {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; color: white; margin-bottom: 20px;}
    .metric-card {background: #ffffff; color: #1f2937; border-radius: 14px; padding: 20px; box-shadow: 0 10px 28px rgba(0,0,0,0.12);}    
    .metric-card h2 { color: #111827 !important; font-size: 28px !important; font-weight: 800 !important; letter-spacing: -0.02em; }
    .metric-card h3 { color: #4f46e5 !important; font-weight: 700 !important; }
    .stExpander > div:first-child {background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: white; border-radius: 10px;}
    .export-dropdown {position: relative; display: inline-block;}
    .export-dropdown-content {display: none; position: absolute; background-color: #f9f9f9; min-width: 160px; box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2); z-index: 1; border-radius: 8px;}
    .export-dropdown:hover .export-dropdown-content {display: block;}
    .export-dropdown-content a {color: black; padding: 12px 16px; text-decoration: none; display: block; border-radius: 8px;}
    .export-dropdown-content a:hover {background-color: #f1f1f1;}
    .chart-container {background: #0f172a; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.25); margin: 16px 0; border: 1px solid rgba(255,255,255,0.06);} 
    .chart-title {font-size: 18px; font-weight: bold; color: #e5e7eb; margin-bottom: 10px; text-align: center;}
    .suggestions-section {background: transparent; padding: 6px 0 0 0; margin: 0 0 8px 0; border: none;}

    .stChatMessage[data-testid="stChatMessage"] {border-radius: 14px; padding: 12px 16px; border: 1px solid rgba(255,255,255,0.06);} 
    .stChatMessage[data-testid="stChatMessage"] p { margin: 0; color: #e5e7eb; }
    .stChatMessage[data-testid="stChatMessage"] pre, .stChatMessage[data-testid="stChatMessage"] code { background: #0b1220 !important; color: #e5e7eb !important; border-radius: 8px; }

    .stButton>button { background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%); color: white; border: 0; border-radius: 12px; padding: 0.6rem 1rem; box-shadow: 0 8px 22px rgba(124, 58, 237, 0.25);} 
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(124, 58, 237, 0.35);}    

    div[data-baseweb="select"]>div { background: #0f172a; color: #e5e7eb; border-radius: 10px; border: 1px solid rgba(255,255,255,0.08);} 
    .stTextInput>div>div>input, .stChatInput input { background: #0f172a; color: #e5e7eb; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; }

    .stDataFrame { filter: drop-shadow(0 10px 24px rgba(0,0,0,0.25)); }
    .stDataFrame table { border-radius: 10px; overflow: hidden; }

    .stDownloadButton>button { background: linear-gradient(90deg, #10b981 0%, #06b6d4 100%); color: white; border: 0; border-radius: 10px; }
</style>
"""


def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


