"""Analytics Dashboard - Interactive data exploration and visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analytics.database import DataStore
from src.analytics.metrics import calculate_ltv, calculate_cac, calculate_roi
from src.analytics.ab_testing import ABTestAnalyzer

st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("Analytics Dashboard")
    st.caption("Upload your data and explore insights with interactive charts")
    
    # Initialize data store in session
    if "data_store" not in st.session_state:
        st.session_state.data_store = DataStore()
    
    data_store = st.session_state.data_store
    
    # Sidebar: Data upload
    with st.sidebar:
        st.header("Data Source")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            help="Upload any CSV file to analyze"
        )
        
        if uploaded_file:
            result = data_store.load_csv(
                uploaded_file.getvalue(),
                table_name="uploaded_data"
            )
            
            if result["success"]:
                st.success(f"Loaded {result['rows']} rows, {len(result['columns'])} columns")
                st.session_state.current_table = "uploaded_data"
            else:
                st.error(f"Error: {result['error']}")
        
        # Show available tables
        tables = data_store.get_tables()
        if tables:
            st.divider()
            st.subheader("Available Tables")
            for table in tables:
                st.code(table)
    
    # Main content
    if not data_store.get_tables():
        st.info("Upload a CSV file to get started")
        
        # Show example
        with st.expander("Example: What kind of data works?"):
            st.markdown("""
            **Any CSV with columns like:**
            - Sales data: date, product, amount, region
            - Customer data: customer_id, signup_date, spend
            - Marketing data: campaign, clicks, conversions
            - A/B test data: variant (A/B), converted (0/1)
            
            **Download sample datasets from Kaggle:**
            - [Online Retail Dataset](https://www.kaggle.com/datasets/carrie1/ecommerce-data)
            - [E-Commerce Sales Dataset](https://www.kaggle.com/datasets/thedevastator/unlock-profits-with-e-commerce-sales-data)
            """)
        return
    
    # Get current data
    df = data_store.get_dataframe("uploaded_data")
    if df is None:
        st.warning("No data loaded")
        return
    
    # Tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Charts", "SQL Query", "A/B Testing", "Metrics"
    ])
    
    with tab1:
        render_overview(df, data_store)
    
    with tab2:
        render_charts(df, data_store)
    
    with tab3:
        render_sql_query(data_store)
    
    with tab4:
        render_ab_testing(df)
    
    with tab5:
        render_metrics(df)


def render_overview(df: pd.DataFrame, data_store: DataStore):
    """Render data overview tab."""
    st.subheader("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        numeric_cols = df.select_dtypes(include=['number']).columns
        st.metric("Numeric Columns", len(numeric_cols))
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column info
    st.subheader("Column Types")
    col_types = data_store.detect_column_types(df)
    
    type_df = pd.DataFrame([
        {"Column": col, "Detected Type": t, "Sample": str(df[col].iloc[0])[:50]}
        for col, t in col_types.items()
    ])
    st.dataframe(type_df, use_container_width=True)
    
    # Suggestions
    suggestions = data_store.suggest_analytics(df)
    if suggestions["recommended_charts"]:
        st.subheader("Recommended Charts")
        for chart in suggestions["recommended_charts"]:
            st.markdown(f"- **{chart['type'].title()}**: {chart['x']} vs {chart['y']} — {chart['description']}")


def render_charts(df: pd.DataFrame, data_store: DataStore):
    """Render interactive charts tab."""
    st.subheader("Create Charts")
    
    col_types = data_store.detect_column_types(df)
    numeric_cols = [c for c, t in col_types.items() if t == 'numeric']
    category_cols = [c for c, t in col_types.items() if t in ['category', 'text']]
    date_cols = [c for c, t in col_types.items() if t == 'date']
    
    chart_type = st.selectbox(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"]
    )
    
    col1, col2 = st.columns(2)
    
    if chart_type == "Bar Chart":
        with col1:
            x_col = st.selectbox("X-Axis (Category)", category_cols + date_cols)
        with col2:
            y_col = st.selectbox("Y-Axis (Value)", numeric_cols)
        
        if x_col and y_col:
            # Aggregate data
            agg_df = df.groupby(x_col)[y_col].sum().reset_index()
            fig = px.bar(agg_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Line Chart":
        with col1:
            x_col = st.selectbox("X-Axis (Date/Category)", date_cols + category_cols)
        with col2:
            y_col = st.selectbox("Y-Axis (Value)", numeric_cols)
        
        if x_col and y_col:
            agg_df = df.groupby(x_col)[y_col].sum().reset_index()
            fig = px.line(agg_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Scatter Plot":
        with col1:
            x_col = st.selectbox("X-Axis", numeric_cols)
        with col2:
            y_col = st.selectbox("Y-Axis", [c for c in numeric_cols if c != x_col] or numeric_cols)
        
        color_col = st.selectbox("Color by (optional)", ["None"] + category_cols)
        
        if x_col and y_col:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=None if color_col == "None" else color_col,
                title=f"{y_col} vs {x_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Pie Chart":
        cat_col = st.selectbox("Category", category_cols)
        val_col = st.selectbox("Value", numeric_cols)
        
        if cat_col and val_col:
            agg_df = df.groupby(cat_col)[val_col].sum().reset_index()
            fig = px.pie(agg_df, names=cat_col, values=val_col, title=f"{val_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        num_col = st.selectbox("Column", numeric_cols)
        bins = st.slider("Number of bins", 10, 100, 30)
        
        if num_col:
            fig = px.histogram(df, x=num_col, nbins=bins, title=f"Distribution of {num_col}")
            st.plotly_chart(fig, use_container_width=True)


def render_sql_query(data_store: DataStore):
    """Render SQL query tab."""
    st.subheader("SQL Query")
    st.caption("Query your data using SQL")
    
    # Show schema
    with st.expander("Table Schema"):
        for table in data_store.get_tables():
            st.markdown(f"**{table}**")
            schema = data_store.get_schema(table)
            st.dataframe(pd.DataFrame(schema), use_container_width=True)
    
    # Query input
    query = st.text_area(
        "Enter SQL query",
        value="SELECT * FROM uploaded_data LIMIT 10",
        height=100
    )
    
    if st.button("Run Query"):
        try:
            result = data_store.query(query)
            st.success(f"Query returned {len(result)} rows")
            st.dataframe(result, use_container_width=True)
            
            # Option to download
            csv = result.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                "query_results.csv",
                "text/csv"
            )
        except ValueError as e:
            st.error(str(e))


def render_ab_testing(df: pd.DataFrame):
    """Render A/B testing tab."""
    st.subheader("A/B Test Analyzer")
    
    st.markdown("""
    Analyze A/B test results to determine statistical significance.
    
    **Your data should have:**
    - A column with variant labels (A/B or control/treatment)
    - A column with conversion (1/0, True/False, or yes/no)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        variant_col = st.selectbox(
            "Variant Column (A/B groups)",
            df.columns.tolist()
        )
    
    with col2:
        # Try to find a conversion-like column
        potential_conv = [c for c in df.columns if 'convert' in c.lower() or 'success' in c.lower()]
        conversion_col = st.selectbox(
            "Conversion Column (0/1)",
            potential_conv + df.columns.tolist()
        )
    
    if variant_col and conversion_col:
        # Get unique variants
        variants = df[variant_col].unique().tolist()
        
        if len(variants) >= 2:
            control = st.selectbox("Control Group (A)", variants, index=0)
            treatment = st.selectbox("Treatment Group (B)", [v for v in variants if v != control], index=0)
            
            if st.button("Analyze A/B Test"):
                analyzer = ABTestAnalyzer(confidence_level=0.95)
                
                try:
                    result = analyzer.analyze_from_dataframe(
                        df,
                        variant_col=variant_col,
                        conversion_col=conversion_col,
                        control_label=control,
                        treatment_label=treatment
                    )
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Control Conversion", f"{result.control_conversion}%")
                    with col2:
                        st.metric("Treatment Conversion", f"{result.treatment_conversion}%")
                    with col3:
                        color = "green" if result.relative_uplift > 0 else "red"
                        st.metric("Relative Uplift", f"{result.relative_uplift}%")
                    
                    # Significance
                    if result.is_significant:
                        st.success(f"Statistically significant at {result.confidence_level}% confidence (p={result.p_value})")
                    else:
                        st.warning(f"Not statistically significant (p={result.p_value})")
                    
                    st.info(result.recommendation)
                    
                except Exception as e:
                    st.error(f"Analysis error: {e}")
        else:
            st.warning("Need at least 2 variants for A/B testing")


def render_metrics(df: pd.DataFrame):
    """Render business metrics tab."""
    st.subheader("Business Metrics Calculator")
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Simple aggregations
    st.markdown("### Quick Stats")
    
    if numeric_cols:
        metric_col = st.selectbox("Select metric column", numeric_cols)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sum", f"{df[metric_col].sum():,.2f}")
        with col2:
            st.metric("Average", f"{df[metric_col].mean():,.2f}")
        with col3:
            st.metric("Min", f"{df[metric_col].min():,.2f}")
        with col4:
            st.metric("Max", f"{df[metric_col].max():,.2f}")
    
    # ROI Calculator
    st.markdown("### ROI Calculator")
    col1, col2 = st.columns(2)
    with col1:
        revenue = st.number_input("Total Revenue", min_value=0.0, value=10000.0)
    with col2:
        cost = st.number_input("Total Cost", min_value=0.0, value=5000.0)
    
    if st.button("Calculate ROI"):
        roi_result = calculate_roi(revenue, cost)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ROI", f"{roi_result['roi_percent']}%")
        with col2:
            st.metric("Profit", f"${roi_result['profit']:,.2f}")
        with col3:
            st.metric("Revenue", f"${roi_result['revenue']:,.2f}")


if __name__ == "__main__":
    main()
