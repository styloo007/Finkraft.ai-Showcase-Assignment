from pathlib import Path
import time
import pandas as pd
import streamlit as st


def save_uploaded_once(uploaded_file) -> Path | None:
    if uploaded_file is None:
        return None

    if (st.session_state.saved_file_path is None) or (
        st.session_state.uploaded_file_name != uploaded_file.name
    ):
        try:
            try:
                app_dir = Path(__file__).parent.resolve()
            except NameError:
                app_dir = Path.cwd()

            orig_filename = Path(uploaded_file.name).name
            timestamp = int(time.time())
            safe_filename = f"{Path(orig_filename).stem}_{timestamp}{Path(orig_filename).suffix}"
            save_path = app_dir / safe_filename

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.saved_file_path = str(save_path)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.suggestions_generated = False
            st.session_state.query_suggestions = []
            st.success(f"File '{orig_filename}' uploaded and saved to: {save_path.name}")
            return save_path
        except Exception as e:
            st.error(f"Failed to save file: {e}")
            st.stop()
    else:
        save_path = Path(st.session_state.saved_file_path)
        st.info(f"Using previously saved file: {save_path.name}")
        return save_path


def load_dataframe_from_saved() -> pd.DataFrame:
    try:
        save_path = Path(st.session_state.saved_file_path)
        if st.session_state.last_loaded_path != str(save_path):
            if save_path.suffix.lower() == ".csv":
                df = pd.read_csv(save_path)
            elif save_path.suffix.lower() in [".xls", ".xlsx"]:
                df = pd.read_excel(save_path)
            else:
                df = pd.read_csv(save_path)

            st.session_state.df = df
            st.session_state.last_loaded_path = str(save_path)
        else:
            df = st.session_state.df
        return df
    except Exception as e:
        st.error(f"Could not read the uploaded file into a DataFrame: {e}")
        st.stop()


