import streamlit as st
import pandas as pd
from pathlib import Path
from io import StringIO
import re
import json
from core.config import configure_environment, configure_page, get_gemini_model
from ui.styles import inject_custom_css
from services.state import ensure_defaults
from services.data_io import save_uploaded_once, load_dataframe_from_saved
from services.viz import create_enhanced_chart

configure_environment()
configure_page()

inject_custom_css()

st.title("🧠 Data Analyzer Agent")

ensure_defaults()


# --- File uploader (CSV / Excel) ---
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"]) 

if uploaded_file is not None:
    save_path = save_uploaded_once(uploaded_file)
    df = load_dataframe_from_saved()

    # --- Enhanced Basic file / data info ---
    st.markdown("## 📈 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📊 Rows</h3>
            <h2 style="margin: 5px 0; color:#111827;">{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">📋 Columns</h3>
            <h2 style="margin: 5px 0; color:#111827;">{}</h2>
        </div>
        """.format(len(df.columns)), unsafe_allow_html=True)
    

    # --- Dropdown sections for Data View and Column Info ---
    st.markdown("")
    
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
        model = get_gemini_model('gemini-1.5-flash')
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

    # Visualization is provided by Client.services.viz.create_enhanced_chart

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
                    explanation_model = get_gemini_model('gemini-1.5-flash')
                    
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
    st.markdown('<div class="suggestions-section">', unsafe_allow_html=True)
    st.markdown("## 💡 AI Query Suggestions")
    st.markdown("*Click to auto-run a suggested query*")
    
    with st.container():
        # Generate suggestions button
        col_suggest, col_refresh = st.columns([3, 1])
        
        with col_suggest:
            if st.button("🔍 Generate Query Suggestions", disabled=st.session_state.suggestions_generated, use_container_width=True):
                with st.spinner("🤖 Generating intelligent query suggestions..."):
                    try:
                        suggestion_model = get_gemini_model('gemini-1.5-flash')
                        
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
    st.markdown("## 🤖 Conversation History")

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