import re
import pandas as pd
import altair as alt
import streamlit as st


def create_enhanced_chart(data: pd.DataFrame, viz_hint: str, title: str = "Data Visualization"):
    cols = list(data.columns)
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = data.select_dtypes(include=['datetime64']).columns.tolist()

    data_clean = data.dropna()
    mentioned_cols = [c for c in cols if re.search(rf"\b{re.escape(c)}\b", viz_hint, re.I)]
    hint = viz_hint.lower()

    try:
        if any(word in hint for word in ['bar', 'column', 'count', 'frequency']):
            if mentioned_cols and mentioned_cols[0] in categorical_cols:
                cat_col = mentioned_cols[0]
                if len(mentioned_cols) > 1 and mentioned_cols[1] in numeric_cols:
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
                        y=alt.Y(f'{num_col}:Q', title=f"Total {num_col.replace('_', ' ').title()}"),
                        tooltip=[alt.Tooltip(f'{cat_col}:N'), alt.Tooltip(f'{num_col}:Q', format='.2f')]
                    ).properties(width=600, height=400).resolve_scale(color='independent')
                    return chart
                else:
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
                    ).properties(width=600, height=400)
                    return chart

        if any(word in hint for word in ['line', 'trend', 'time', 'over time']):
            if date_cols:
                date_col = date_cols[0]
            elif categorical_cols:
                date_col = categorical_cols[0]
            else:
                date_col = cols[0]

            if numeric_cols:
                num_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0]

                try:
                    data_clean[date_col] = pd.to_datetime(data_clean[date_col])
                    x_type = 'T'
                except:
                    x_type = 'N'

                chart = alt.Chart(data_clean).mark_line(
                    point=alt.OverlayMarkDef(filled=False, fill='white', size=50),
                    color='#667eea',
                    strokeWidth=3
                ).encode(
                    x=alt.X(f'{date_col}:{x_type}', title=date_col.replace('_', ' ').title()),
                    y=alt.Y(f'{num_col}:Q', title=num_col.replace('_', ' ').title()),
                    tooltip=[alt.Tooltip(f'{date_col}:{x_type}'), alt.Tooltip(f'{num_col}:Q', format='.2f')]
                ).properties(width=600, height=400)
                return chart

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
                    x=alt.X(f'{num_col}:Q', bin=alt.Bin(maxbins=30), title=num_col.replace('_', ' ').title()),
                    y=alt.Y('count()', title='Frequency'),
                    tooltip=['count()']
                ).properties(width=600, height=400)
                return chart

        if any(word in hint for word in ['scatter', 'correlation', 'relationship']):
            if len(numeric_cols) >= 2:
                x_col = mentioned_cols[0] if mentioned_cols and mentioned_cols[0] in numeric_cols else numeric_cols[0]
                y_col = mentioned_cols[1] if len(mentioned_cols) > 1 and mentioned_cols[1] in numeric_cols else numeric_cols[1]

                chart = alt.Chart(data_clean).mark_circle(
                    size=60, opacity=0.7, color='#667eea', stroke='white', strokeWidth=1
                ).encode(
                    x=alt.X(f'{x_col}:Q', title=x_col.replace('_', ' ').title()),
                    y=alt.Y(f'{y_col}:Q', title=y_col.replace('_', ' ').title()),
                    tooltip=[alt.Tooltip(f'{x_col}:Q', format='.2f'), alt.Tooltip(f'{y_col}:Q', format='.2f')]
                ).properties(width=600, height=400)
                return chart

        if categorical_cols and numeric_cols:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            if data_clean[cat_col].nunique() <= 10:
                grouped_data = data_clean.groupby(cat_col)[num_col].mean().reset_index()
                chart = alt.Chart(grouped_data).mark_bar(
                    color='#667eea', cornerRadiusTopLeft=3, cornerRadiusTopRight=3
                ).encode(
                    x=alt.X(f'{cat_col}:N', sort='-y', title=cat_col.replace('_', ' ').title()),
                    y=alt.Y(f'{num_col}:Q', title=f"Average {num_col.replace('_', ' ').title()}"),
                    tooltip=[alt.Tooltip(f'{cat_col}:N'), alt.Tooltip(f'{num_col}:Q', format='.2f')]
                ).properties(width=600, height=400)
                return chart

        if numeric_cols:
            num_col = numeric_cols[0]
            chart = alt.Chart(data_clean).mark_bar(
                color='#764ba2', cornerRadiusTopLeft=2, cornerRadiusTopRight=2
            ).encode(
                x=alt.X(f'{num_col}:Q', bin=alt.Bin(maxbins=20), title=num_col.replace('_', ' ').title()),
                y=alt.Y('count()', title='Count'),
                tooltip=['count()']
            ).properties(width=600, height=400)
            return chart

    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

    return None


