import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
from pathlib import Path
from io import StringIO
import time
import re
import altair as alt
import numpy as np
import json

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="CSV/Excel Uploader with Gemini Chat", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .suggestion-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .suggestion-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .suggestion-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 20px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stExpander > div:first-child {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 10px;
    }
    .export-dropdown {
        position: relative;
        display: inline-block;
    }
    .export-dropdown-content {
        display: none;
        position: absolute;
        background-color: #f9f9f9;
        min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        z-index: 1;
        border-radius: 8px;
    }
    .export-dropdown:hover .export-dropdown-content {
        display: block;
    }
    .export-dropdown-content a {
        color: black;
        padding: 12px 16px;
        text-decoration: none;
        display: block;
        border-radius: 8px;
    }
    .export-dropdown-content a:hover {
        background-color: #f1f1f1;
    }
    
    /* Enhanced chart styling */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 15px 0;
        border: 1px solid #e0e0e0;
    }
    
    .chart-title {
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
        text-align: center;
    }

    .suggestions-section {
        position: sticky;
        top: 0;
        background: white;
        z-index: 100;
        padding: 20px 0;
        border-bottom: 2px solid #f0f0f0;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 CSV/Excel Uploader with Gemini AI Assistant")

# --- Session state defaults ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "explanations" not in st.session_state:
    st.session_state.explanations = {}
if "saved_file_path" not in st.session_state:
    st.session_state.saved_file_path = None
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "last_loaded_path" not in st.session_state:
    st.session_state.last_loaded_path = None
if "df" not in st.session_state:
    st.session_state.df = None
if "query_suggestions" not in st.session_state:
    st.session_state.query_suggestions = []
if "suggestions_generated" not in st.session_state:
    st.session_state.suggestions_generated = False
if "processing_suggestion" not in st.session_state:
    st.session_state.processing_suggestion = False


# --- File uploader (CSV / Excel) ---
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"]) 

if uploaded_file is not None:
    # Save the uploaded file only once per upload (prevent duplicate saves on every rerun)
    if (st.session_state.saved_file_path is None) or (st.session_state.uploaded_file_name != uploaded_file.name):
        try:
            try:
                app_dir = Path(__file__).parent.resolve()
            except NameError:
                app_dir = Path.cwd()

            orig_filename = Path(uploaded_file.name).name
            # use a timestamped filename to avoid accidental overwrites while still saving only once per upload
            timestamp = int(time.time())
            safe_filename = f"{Path(orig_filename).stem}_{timestamp}{Path(orig_filename).suffix}"
            save_path = app_dir / safe_filename

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.session_state.saved_file_path = str(save_path)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.suggestions_generated = False  # Reset suggestions when new file is uploaded
            st.session_state.query_suggestions = []
            st.success(f"File '{orig_filename}' uploaded and saved to: {save_path.name}")
        except Exception as e:
            st.error(f"Failed to save file: {e}")
            st.stop()
    else:
        save_path = Path(st.session_state.saved_file_path)
        st.info(f"Using previously saved file: {save_path.name}")

    # Load the DataFrame (and cache it in session_state so we don't re-read unnecessarily)
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
    except Exception as e:
        st.error(f"Could not read the uploaded file into a DataFrame: {e}")
        st.stop()

    # --- Enhanced Basic file / data info ---
    st.markdown("### 📈 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📊 Rows</h3>
            <h2 style="margin: 5px 0;">{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📋 Columns</h3>
            <h2 style="margin: 5px 0;">{}</h2>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📁 File</h3>
            <h2 style="margin: 5px 0; font-size: 16px;">{}</h2>
        </div>
        """.format(Path(st.session_state.saved_file_path).name), unsafe_allow_html=True)

    # --- Dropdown sections for Data View and Column Info ---
    st.markdown("---")
    
    with st.expander("📊 Data View", expanded=False):
        st.markdown("### 📋 Data Preview")
        st.dataframe(df, use_container_width=True)

    with st.expander("ℹ️ Column Information", expanded=False):
        st.markdown("### 📊 Column Information")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values,
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.markdown("### 📈 Numeric Column Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

    # --- Initialize Gemini chat session ---
    if st.session_state.chat_session is None:
        model = genai.GenerativeModel('gemini-1.5-flash')
        data_context = f"""You are a helpful data analyst assistant specialized in analyzing tabular data.

The user has uploaded a file with the following information:
- File name: {Path(st.session_state.saved_file_path).name}
- Number of rows: {len(df)}
- Number of columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}
- Data types: {df.dtypes.to_dict()}
- Sample data (first 5 rows):
{df.head().to_string()}

IMPORTANT INSTRUCTIONS:

1. When asked for data analysis, filtering, grouping, or summaries:
   - Perform the analysis conceptually.
   - Produce a tabular result (CSV/JSON friendly) inside a ```result``` block.
   - Produce a natural-language explanation inside an ```explanation``` block.
   - Optionally produce the best visualization suggestion inside a ```viz``` block (e.g., "bar chart of Department vs salary").

2. Example response format (AI should return):
   ```result
   <CSV text, or header+rows>
   ```
   ```explanation
   <Natural language explanation>
   ```
   ```viz
   <Best Visualization suggestion>
   ```
"""
        st.session_state.chat_session = model.start_chat(history=[])
        st.session_state.chat_session.send_message(data_context)

    # Helper to extract blocks
    def get_block(text, tag):
        m = re.search(rf"```{re.escape(tag)}\s*(.*?)\s*```", text, re.S)
        return m.group(1).strip() if m else ""

    # Enhanced auto visualization with better logic and styling
    def create_enhanced_chart(data, viz_hint, title="Data Visualization"):
        cols = list(data.columns)
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Clean data first
        data_clean = data.dropna()
        
        # Detect mentioned columns in the hint
        mentioned_cols = [c for c in cols if re.search(rf"\b{re.escape(c)}\b", viz_hint, re.I)]
        hint = viz_hint.lower()
        
        try:
            # Bar chart logic
            if any(word in hint for word in ['bar', 'column', 'count', 'frequency']):
                if mentioned_cols and mentioned_cols[0] in categorical_cols:
                    cat_col = mentioned_cols[0]
                    if len(mentioned_cols) > 1 and mentioned_cols[1] in numeric_cols:
                        # Categorical vs Numeric
                        num_col = mentioned_cols[1]
                        grouped_data = data_clean.groupby(cat_col)[num_col].sum().reset_index()
                        chart = alt.Chart(grouped_data).mark_bar(
                            color=alt.Gradient(
                                gradient='linear',
                                stops=[alt.GradientStop(color='#667eea', offset=0),
                                      alt.GradientStop(color='#764ba2', offset=1)]
                            ),
                            cornerRadiusTopLeft=3,
                            cornerRadiusTopRight=3
                        ).encode(
                            x=alt.X(f'{cat_col}:N', 
                                  sort=alt.EncodingSortField(field=num_col, op='sum', order='descending'),
                                  title=cat_col.replace('_', ' ').title()),
                            y=alt.Y(f'{num_col}:Q', 
                                  title=f'Total {num_col.replace("_", " ").title()}'),
                            tooltip=[alt.Tooltip(f'{cat_col}:N'), 
                                   alt.Tooltip(f'{num_col}:Q', format='.2f')]
                        ).properties(
                            width=600,
                            height=400,
                            title=alt.TitleParams(
                                text=f'{num_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}',
                                fontSize=16,
                                fontWeight='bold'
                            )
                        ).resolve_scale(color='independent')
                        return chart
                    else:
                        # Count by category
                        count_data = data_clean[cat_col].value_counts().reset_index()
                        count_data.columns = [cat_col, 'count']
                        chart = alt.Chart(count_data).mark_bar(
                            color='#667eea',
                            cornerRadiusTopLeft=3,
                            cornerRadiusTopRight=3
                        ).encode(
                            x=alt.X(f'{cat_col}:N', sort='-y', title=cat_col.replace('_', ' ').title()),
                            y=alt.Y('count:Q', title='Count'),
                            tooltip=[alt.Tooltip(f'{cat_col}:N'), alt.Tooltip('count:Q')]
                        ).properties(
                            width=600,
                            height=400,
                            title=alt.TitleParams(
                                text=f'Count by {cat_col.replace("_", " ").title()}',
                                fontSize=16,
                                fontWeight='bold'
                            )
                        )
                        return chart
            
            # Line chart logic
            if any(word in hint for word in ['line', 'trend', 'time', 'over time']):
                if date_cols:
                    date_col = date_cols[0]
                elif categorical_cols:
                    date_col = categorical_cols[0]
                else:
                    date_col = cols[0]
                
                if numeric_cols:
                    num_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0]
                    
                    # Try to parse date column
                    try:
                        data_clean[date_col] = pd.to_datetime(data_clean[date_col])
                        sort_field = date_col
                        x_type = 'T'
                    except:
                        sort_field = date_col
                        x_type = 'N'
                    
                    chart = alt.Chart(data_clean).mark_line(
                        point=alt.OverlayMarkDef(filled=False, fill='white', size=50),
                        color='#667eea',
                        strokeWidth=3
                    ).encode(
                        x=alt.X(f'{date_col}:{x_type}', title=date_col.replace('_', ' ').title()),
                        y=alt.Y(f'{num_col}:Q', title=num_col.replace('_', ' ').title()),
                        tooltip=[alt.Tooltip(f'{date_col}:{x_type}'), 
                               alt.Tooltip(f'{num_col}:Q', format='.2f')]
                    ).properties(
                        width=600,
                        height=400,
                        title=alt.TitleParams(
                            text=f'{num_col.replace("_", " ").title()} Trend',
                            fontSize=16,
                            fontWeight='bold'
                        )
                    )
                    return chart
            
            # Histogram/Distribution
            if any(word in hint for word in ['hist', 'distribution', 'spread']):
                if numeric_cols:
                    num_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0]
                    chart = alt.Chart(data_clean).mark_bar(
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='#4facfe', offset=0),
                                  alt.GradientStop(color='#00f2fe', offset=1)]
                        ),
                        cornerRadiusTopLeft=2,
                        cornerRadiusTopRight=2
                    ).encode(
                        x=alt.X(f'{num_col}:Q', bin=alt.Bin(maxbins=30), 
                               title=num_col.replace('_', ' ').title()),
                        y=alt.Y('count()', title='Frequency'),
                        tooltip=['count()']
                    ).properties(
                        width=600,
                        height=400,
                        title=alt.TitleParams(
                            text=f'Distribution of {num_col.replace("_", " ").title()}',
                            fontSize=16,
                            fontWeight='bold'
                        )
                    )
                    return chart
            
            # Scatter plot
            if any(word in hint for word in ['scatter', 'correlation', 'relationship']):
                if len(numeric_cols) >= 2:
                    x_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0]
                    y_col = mentioned_cols[1] if len(mentioned_cols) > 1 and mentioned_cols[1] in numeric_cols else numeric_cols[1]
                    
                    chart = alt.Chart(data_clean).mark_circle(
                        size=60,
                        opacity=0.7,
                        color='#667eea',
                        stroke='white',
                        strokeWidth=1
                    ).encode(
                        x=alt.X(f'{x_col}:Q', title=x_col.replace('_', ' ').title()),
                        y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
                        tooltip=[alt.Tooltip(f'{x_col}:Q', format='.2f'), 
                               alt.Tooltip(f'{y_col}:Q', format='.2f')]
                    ).properties(
                        width=600,
                        height=400,
                        title=alt.TitleParams(
                            text=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}',
                            fontSize=16,
                            fontWeight='bold'
                        )
                    )
                    return chart
            
            # Default: Smart chart based on data
            if categorical_cols and numeric_cols:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                if data_clean[cat_col].nunique() <= 10:  # Bar chart for few categories
                    grouped_data = data_clean.groupby(cat_col)[num_col].mean().reset_index()
                    chart = alt.Chart(grouped_data).mark_bar(
                        color='#667eea',
                        cornerRadiusTopLeft=3,
                        cornerRadiusTopRight=3
                    ).encode(
                        x=alt.X(f'{cat_col}:N', sort='-y', title=cat_col.replace('_', ' ').title()),
                        y=alt.Y(f'{num_col}:Q', title=f'Average {num_col.replace("_", " ").title()}'),
                        tooltip=[alt.Tooltip(f'{cat_col}:N'), 
                               alt.Tooltip(f'{num_col}:Q', format='.2f')]
                    ).properties(
                        width=600,
                        height=400,
                        title=alt.TitleParams(
                            text=f'Average {num_col.replace("_", " ").title()} by {cat_col.replace("_", " ").title()}',
                            fontSize=16,
                            fontWeight='bold'
                        )
                    )
                    return chart
            
            # Fallback: First numeric column histogram
            if numeric_cols:
                num_col = numeric_cols[0]
                chart = alt.Chart(data_clean).mark_bar(
                    color='#764ba2',
                    cornerRadiusTopLeft=2,
                    cornerRadiusTopRight=2
                ).encode(
                    x=alt.X(f'{num_col}:Q', bin=alt.Bin(maxbins=20), 
                           title=num_col.replace('_', ' ').title()),
                    y=alt.Y('count()', title='Count'),
                    tooltip=['count()']
                ).properties(
                    width=600,
                    height=400,
                    title=alt.TitleParams(
                        text=f'Distribution of {num_col.replace("_", " ").title()}',
                        fontSize=16,
                        fontWeight='bold'
                    )
                )
                return chart
                
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None
        
        return None

    # Enhanced export dropdown component
    def create_export_dropdown(displayed_df, button_key_prefix):
        if displayed_df is not None:
            st.markdown("### 📥 Export Data")
            
            # Create two columns for the dropdown effect
            col_export, col_spacer = st.columns([2, 8])
            
            with col_export:
                export_option = st.selectbox(
                    "Choose export format:",
                    ["Select format...", "Export as CSV", "Export as JSON"],
                    key=f"export_select_{button_key_prefix}"
                )
                
                if export_option == "Export as CSV":
                    csv_bytes = displayed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_bytes,
                        file_name="analysis_result.csv",
                        mime="text/csv",
                        key=f"csv_{button_key_prefix}",
                        use_container_width=True
                    )
                elif export_option == "Export as JSON":
                    json_bytes = displayed_df.to_json(orient="records", indent=2).encode('utf-8')
                    st.download_button(
                        "⬇️ Download JSON",
                        data=json_bytes,
                        file_name="analysis_result.json",
                        mime="application/json",
                        key=f"json_{button_key_prefix}",
                        use_container_width=True
                    )

    # Function to render response content (unified for both current and history)
    def render_response_content(response_text, message_index, is_current=False):
        # Extract blocks
        result_part = get_block(response_text, 'result')
        explanation_part = get_block(response_text, 'explanation')
        viz_part = get_block(response_text, 'viz')

        displayed_df = None

        # Try to parse result block as CSV first, then JSON
        if result_part:
            parsed = None
            try:
                parsed = pd.read_csv(StringIO(result_part))
            except Exception:
                try:
                    parsed = pd.read_json(result_part)
                except Exception:
                    parsed = None

            if isinstance(parsed, pd.DataFrame):
                displayed_df = parsed
                if is_current:
                    st.session_state.last_result = displayed_df
                
                st.markdown("### 📊 Analysis Results")
                st.dataframe(displayed_df, use_container_width=True)

                # Enhanced export dropdown
                button_key_prefix = "current" if is_current else f"hist_{message_index}"
                create_export_dropdown(displayed_df, button_key_prefix)
                
            else:
                # If we couldn't parse the result as table, show the raw text
                st.text(result_part)

        # If no result block, show preview for current message
        if displayed_df is None and result_part == "" and is_current:
            st.dataframe(df.head(50), use_container_width=True)
            st.info("Showing a preview of your uploaded data because the assistant didn't return a tabular `result`. Ask the assistant to return a CSV-compatible `result` block for downloadable tables.")

        # Enhanced Visualization
        data_for_viz = displayed_df if displayed_df is not None else df
        if viz_part and len(data_for_viz) > 0:
            chart = create_enhanced_chart(data_for_viz, viz_part)
            if chart is not None:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown("### 📈 Data Visualization")
                st.altair_chart(chart, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info(f"💡 Visualization suggestion: {viz_part}")

        # Single Explain Button
        button_key = f"explain_btn_current_{message_index}" if is_current else f"explain_btn_hist_{message_index}"
        if st.button("📖 Get Detailed Explanation", key=button_key, use_container_width=False):
            with st.spinner("Generating detailed explanation..."):
                try:
                    explanation_model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    if is_current and displayed_df is not None:
                        # Convert DataFrame to a list of dictionaries for more detailed context
                        data_records = displayed_df.to_dict('records')
                        
                        # Build comprehensive context with actual data
                        context_parts = [
                            f"ANALYSIS QUERY RESPONSE DATA:",
                            f"Raw Result Block Content:\n{result_part}",
                            f"\nParsed Data as Records:\n{json.dumps(data_records, indent=2, default=str)}",
                            f"\nDataFrame Shape: {displayed_df.shape[0]} rows × {displayed_df.shape[1]} columns",
                            f"Column Names: {list(displayed_df.columns)}",
                            f"Column Data Types: {displayed_df.dtypes.to_dict()}"
                        ]
                        
                        # Add statistical information for numeric columns
                        numeric_cols_exp = displayed_df.select_dtypes(include=['number']).columns
                        if len(numeric_cols_exp) > 0:
                            context_parts.append(f"\nNUMERIC COLUMN STATISTICS:")
                            for col in numeric_cols_exp:
                                stats = displayed_df[col].describe()
                                context_parts.append(f"{col}: mean={stats.get('mean', 'N/A'):.3f}, min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}, sum={displayed_df[col].sum():.3f}")
                        
                        # Add categorical column information
                        categorical_cols_exp = displayed_df.select_dtypes(include=['object', 'category']).columns
                        if len(categorical_cols_exp) > 0:
                            context_parts.append(f"\nCATEGORICAL COLUMN VALUES:")
                            for col in categorical_cols_exp:
                                unique_vals = displayed_df[col].unique()[:10]  # Limit to first 10 unique values
                                context_parts.append(f"{col}: {list(unique_vals)}")
                        
                        if viz_part:
                            context_parts.append(f"\nVISUALIZATION SUGGESTION: {viz_part}")
                        
                        full_context = "\n".join(context_parts)
                        
                        explanation_prompt = f"""You are analyzing REAL DATA from a business analysis. Below is the complete context including the actual data values and records. I need you to provide a detailed explanation using the SPECIFIC NUMBERS, VALUES, and EXACT DATA from this analysis.

{full_context}

INSTRUCTIONS:
1. Reference EXACT values, numbers, and specific data entries from the records shown above
2. Mention specific row data, compare actual values between different entries
3. Use the real numbers to calculate percentages, ratios, and comparisons  
4. Point out the highest and lowest specific values with their exact amounts
5. Reference specific categorical values and their associated numeric values
6. Use the actual data records to support every statement you make
7. Calculate and mention totals, averages, and other metrics from the real data
8. Compare specific entries (e.g., "Entry X has value Y while Entry Z has value W")

Do NOT give generic explanations. Use the actual data values and records shown above to create a data-driven explanation with specific numbers and comparisons!"""
                    else:
                        # Simple explanation for history messages
                        explanation_prompt = f"""Please provide a clear, detailed explanation of this data analysis response:

{response_text}

Focus on:
1. What analysis was performed
2. What the results mean
3. Any important insights or patterns
4. How to interpret the data shown

Provide only the explanation, no code blocks."""

                    explanation_response = explanation_model.generate_content(explanation_prompt)
                    st.session_state.explanations[message_index] = explanation_response.text
                    st.rerun()
                except Exception as e:
                    st.error(f"Error getting explanation: {str(e)}")
        
        # Show explanation if available
        if message_index in st.session_state.explanations:
            with st.expander("💡 Detailed Explanation", expanded=True):
                st.write(st.session_state.explanations[message_index])

    # Function to process a query
    def process_query(query_text):
        try:
            with st.spinner("🔍 Analyzing your data..."):
                response = st.session_state.chat_session.send_message(query_text)
                response_text = response.text

                # Save assistant response into conversation history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Render the response content
                current_index = len(st.session_state.messages) - 1
                render_response_content(response_text, current_index, is_current=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # --- Query Suggestions Section (Moved to top, after data overview) ---
    st.markdown("---")
    st.markdown('<div class="suggestions-section">', unsafe_allow_html=True)
    st.markdown("## 💡 AI Query Suggestions")
    st.markdown("*Get AI-powered suggestions for analyzing your data*")
    
    with st.container():
        # Generate suggestions button
        col_suggest, col_refresh = st.columns([3, 1])
        
        with col_suggest:
            if st.button("🔍 Generate Query Suggestions", disabled=st.session_state.suggestions_generated, use_container_width=True):
                with st.spinner("🤖 Generating intelligent query suggestions..."):
                    try:
                        suggestion_model = genai.GenerativeModel('gemini-1.5-flash')
                        
                        # Get column info for context
                        numeric_cols_list = df.select_dtypes(include=['number']).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                        
                        # Get sample values for better context
                        sample_data = {}
                        for col in df.columns[:10]:  # Limit to first 10 columns to avoid token limits
                            sample_values = df[col].dropna().head(3).tolist()
                            sample_data[col] = sample_values
                        
                        suggestion_prompt = f"""Generate 8-10 diverse and interesting query suggestions for data analysis based on this dataset structure:

Dataset Info:
- Total rows: {len(df)}
- Total columns: {len(df.columns)}
- Column names: {', '.join(df.columns.tolist())}
- Numeric columns: {', '.join(numeric_cols_list) if numeric_cols_list else 'None'}
- Categorical columns: {', '.join(categorical_cols) if categorical_cols else 'None'}
- Date columns: {', '.join(date_cols) if date_cols else 'None'}

Sample data values:
{json.dumps(sample_data, indent=2, default=str)}

Generate practical queries that would provide business insights. Include a mix of:
1. Aggregations and grouping
2. Filtering and sorting
3. Statistical analysis
4. Comparisons between categories
5. Trend analysis (if date columns exist)
6. Top/bottom N analysis

Format each suggestion as a natural language query that a business user might ask.
Return ONLY the queries, one per line, no numbering or bullets."""

                        suggestion_response = suggestion_model.generate_content(suggestion_prompt)
                        suggestions_text = suggestion_response.text.strip()
                        
                        # Parse suggestions (split by lines and clean up)
                        suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
                        suggestions = [re.sub(r'^\d+[\.\)]\s*', '', s) for s in suggestions]  # Remove numbering
                        suggestions = [s for s in suggestions if len(s) > 10]  # Filter out very short suggestions
                        
                        st.session_state.query_suggestions = suggestions[:10]  # Limit to 10 suggestions
                        st.session_state.suggestions_generated = True
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error generating suggestions: {str(e)}")
        
        with col_refresh:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.suggestions_generated = False
                st.session_state.query_suggestions = []
                st.rerun()
        
        # Display suggestions if available
        if st.session_state.query_suggestions:
            st.markdown("**💬 Click any suggestion to analyze:**")
            
            # Create columns for better layout
            cols_per_row = 2
            for i in range(0, len(st.session_state.query_suggestions), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(st.session_state.query_suggestions):
                        suggestion = st.session_state.query_suggestions[i + j]
                        with col:
                            if st.button(
                                f"💬 {suggestion[:80]}{'...' if len(suggestion) > 80 else ''}", 
                                key=f"suggestion_{i+j}",
                                help=suggestion,
                                use_container_width=True
                            ):
                                st.session_state.selected_suggestion = suggestion
                                st.rerun()
        
        elif st.session_state.suggestions_generated:
            st.info("No suggestions generated. Try refreshing or check your data structure.")
        else:
            st.info("👆 Click 'Generate Query Suggestions' to get AI-powered analysis ideas for your data.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Enhanced Gemini chat section ---
    st.markdown("---")
    st.markdown("# 🤖 Conversation History")

    # Handle suggestion selection (before chat input)
    if hasattr(st.session_state, 'selected_suggestion') and not st.session_state.processing_suggestion:
        st.session_state.processing_suggestion = True
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": st.session_state.selected_suggestion})
        
        # Process the suggestion
        with st.chat_message("user"):
            st.markdown(f"**You:** {st.session_state.selected_suggestion}")

        with st.chat_message("assistant"):
            st.markdown("**Assistant:**")
            process_query(st.session_state.selected_suggestion)
        
        # Clean up
        del st.session_state.selected_suggestion
        st.session_state.processing_suggestion = False

    # Display chat history
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown("**Assistant:**")
                    render_response_content(message["content"], i, is_current=False)

    # --- Query Input Section (Moved to bottom) ---
    st.markdown("---")
    st.markdown("## ✏️ Ask a Question")
    st.markdown("*Type your question about the data below*")
    
    if prompt := st.chat_input("Ask me anything about your data (e.g., 'Show top 10 rows by salary', 'Group by department')..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")

        with st.chat_message("assistant"):
            st.markdown("**Assistant:**")
            process_query(prompt)

    # Clear chat & resets (moved to complete bottom)
    st.markdown("---")
    st.markdown("## 🗑️ Reset Options")
    col_clear, col_spacer = st.columns([2, 8])
    with col_clear:
        if st.button("🗑️ Clear Chat History", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_session = None
            st.session_state.last_result = None
            st.session_state.explanations = {}
            st.session_state.query_suggestions = []
            st.session_state.suggestions_generated = False
            st.session_state.processing_suggestion = False
            st.rerun()

else:
    st.info("📁 Please upload a CSV or Excel file to get started")
    st.markdown("""
    ## 🚀 Features
    - **🔍 Smart Data Analysis**: Upload CSV/Excel files and get instant insights
    - **🤖 AI-Powered Chat**: Ask questions about your data in natural language
    - **💡 Query Suggestions**: Get intelligent suggestions for data exploration
    - **📊 Interactive Visualizations**: Automatic enhanced charts based on your queries
    - **📥 Export Results**: Download analysis results in CSV/JSON format with dropdown selection
    - **📖 Detailed Explanations**: Get comprehensive explanations of analysis results
    - **🎨 Enhanced UI**: Beautiful, modern interface with improved user experience
    """)