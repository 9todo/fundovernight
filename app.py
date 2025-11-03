
import streamlit as st
import pandas as pd
import numpy as np

pd.set_option("styler.render.max_elements", 3000000)

st.set_page_config(layout="wide")

st.title("Overnight Fund Discrimination Analysis")


# Load data from corresponding URL
@st.cache_data
def load_data(source):
    base_url = "https://storage.googleapis.com/overnighttrash/"
    file_map = {
        "TREPS": "TREPS%20Analysis.csv",
        "CROMS": "CROMS%20Analysis.csv"  # make sure this file exists at the URL
    }

    filename_in_bucket = file_map[source]
    url = base_url + filename_in_bucket
    
    try:
        df = pd.read_csv(url)
        # Clean the data
        df["Quantity Traded"] = df["Quantity Traded"].replace({',': ''}, regex=True).astype(float)
        df["YieldatwhichTraded"] = df["YieldatwhichTraded"].astype(float)
        df['Trade Date'] = pd.to_datetime(df['Trade Date'], format='mixed', dayfirst=True)
        return df
    except Exception as e:
        st.error(f"Could not load data from '{url}'. Please check the URL and ensure the file exists. Error: {e}")
        return None

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

with st.form("data_selection_form"):
    data_source = st.selectbox("Select Data Source", ["TREPS", "CROMS"])
    submitted = st.form_submit_button("Submit")

if submitted:
    with st.spinner("Loading and analyzing data... Please wait."):
        st.session_state.df = load_data(data_source)

if st.session_state.df is not None:
    df = st.session_state.df
    with st.spinner("Applying filters and updating view..."):
            # Exclude ETF funds
            df = df[~df['Scheme Name'].str.contains('ETF', case=False, na=False)]

            # Identify Overnight Funds
            df['Fund Type'] = 'Non-Overnight'
            df.loc[df['Scheme Name'].str.contains('Overnight', case=False, na=False), 'Fund Type'] = 'Overnight'

            # Rule 1: Filter for AMCs with Overnight Funds
            amcs_with_overnight = df[df['Fund Type'] == 'Overnight']['Fund House'].unique()
            df = df[df['Fund House'].isin(amcs_with_overnight)]

            # Calculate Weighted Yield
            df['Weighted Yield'] = df['Quantity Traded'] * df['YieldatwhichTraded']

            # Calculate daily yields for each fund type
            daily_yields = df.groupby(['Fund House', 'Trade Date', 'Fund Type']).agg(
                TotalQuantityTraded=('Quantity Traded', 'sum'),
                TotalWeightedYield=('Weighted Yield', 'sum')
            ).reset_index()
            daily_yields['Daily Avg Yield'] = daily_yields['TotalWeightedYield'] / daily_yields['TotalQuantityTraded']

            # Pivot the table to get Overnight and Non-Overnight yields side-by-side
            daily_yields_pivot = daily_yields.pivot_table(
                index=['Fund House', 'Trade Date'],
                columns='Fund Type',
                values='Daily Avg Yield'
            ).reset_index()
            
            # Rule 2: Ignore days with only Overnight or Non-Overnight trades
            daily_yields_pivot = daily_yields_pivot.dropna(subset=['Overnight', 'Non-Overnight'])

            # New Rule: Data corruption handling based on unique funds
            for index, row in daily_yields_pivot.iterrows():
                if not np.isclose(row['Overnight'], row['Non-Overnight'], atol=0.0025):
                    fund_house = row['Fund House']
                    trade_date = row['Trade Date']
                    
                    non_overnight_trades = df[
                        (df['Fund House'] == fund_house) &
                        (df['Trade Date'] == trade_date) &
                        (df['Fund Type'] == 'Non-Overnight')
                    ]
                    
                    if not non_overnight_trades.empty:
                        overnight_yield = row['Overnight']
                        
                        # Group by unique fund and get the average yield
                        non_overnight_funds = non_overnight_trades.groupby('Scheme Name').agg({'YieldatwhichTraded': 'mean'}).reset_index()
                        
                        # Count unique funds with a rate close to the overnight yield
                        same_rate_count = (np.isclose(non_overnight_funds['YieldatwhichTraded'], overnight_yield, atol=0.0025)).sum()
                        
                        if same_rate_count > (len(non_overnight_funds) / 2):
                            daily_yields_pivot.loc[index, 'Non-Overnight'] = overnight_yield

            # Calculate the spread
            daily_yields_pivot['Yield Spread'] = daily_yields_pivot['Overnight'] - daily_yields_pivot['Non-Overnight']

            # Calculate summary metrics for each AMC
            amc_summary = daily_yields_pivot.groupby('Fund House').agg(
                AvgOvernightYield=('Overnight', 'mean'),
                AvgNonOvernightYield=('Non-Overnight', 'mean'),
                AvgYieldSpread=('Yield Spread', 'mean'),
                NumDaysOvernightHigher=('Yield Spread', lambda x: (x > 1e-6).sum()),
                NumDaysOvernightLower=('Yield Spread', lambda x: (x < -1e-6).sum()),
                TotalDays=('Trade Date', 'count')
            ).reset_index()
            
            amc_summary['PctDaysOvernightHigher'] = (amc_summary['NumDaysOvernightHigher'] / amc_summary['TotalDays']) * 100
            amc_summary['PctDaysOvernightLower'] = (amc_summary['NumDaysOvernightLower'] / amc_summary['TotalDays']) * 100

            # Overnight Fund Analysis
            st.header("Overnight Fund Analysis")

            # Filters
            spread_threshold = st.slider("Filter by AvgYieldSpread threshold", min_value=0.0, max_value=0.1, value=0.005, step=0.001, format="%.4f")
            pct_days_range = st.slider("Filter by PctDaysOvernightHigher range", min_value=-100.0, max_value=100.0, value=(-100.0, 100.0), step=1.0, format="%.1f%%")

            filtered_amc_summary = amc_summary[
                (abs(amc_summary['AvgYieldSpread']) > spread_threshold) &
                (amc_summary['PctDaysOvernightHigher'] >= pct_days_range[0]) &
                (amc_summary['PctDaysOvernightHigher'] <= pct_days_range[1])
            ]

            # Always include DSP Mutual Fund
            dsp_benchmark = amc_summary[amc_summary['Fund House'] == 'DSP Mutual Fund']
            if not dsp_benchmark.empty:
                if 'DSP Mutual Fund' not in filtered_amc_summary['Fund House'].unique():
                    filtered_amc_summary = pd.concat([dsp_benchmark, filtered_amc_summary], ignore_index=True)

            filtered_amc_summary = filtered_amc_summary.reset_index(drop=True)
            
            st.dataframe(filtered_amc_summary[[
                'Fund House', 'AvgOvernightYield', 'AvgNonOvernightYield', 'AvgYieldSpread', 'PctDaysOvernightHigher', 'PctDaysOvernightLower'
            ]].style.format({
                "AvgOvernightYield": "{:.2f}",
                "AvgNonOvernightYield": "{:.2f}",
                "AvgYieldSpread": "{:.4f}",
                "PctDaysOvernightHigher": "{:.2f}%",
                "PctDaysOvernightLower": "{:.2f}%"
            }))

            st.header("Yield Spread (Overnight vs. Non-Overnight)")
            if not filtered_amc_summary.empty:
                st.bar_chart(filtered_amc_summary.set_index('Fund House')['AvgYieldSpread'])
            else:
                st.warning("No data to display for the selected filter settings.")


            # Fund Details
            st.header("Fund Details")

            # Filter by AMC
            selected_amc = st.selectbox("Select an AMC to view details", ['All'] + sorted(df['Fund House'].unique()))

            if selected_amc == 'All':
                filtered_df = df
            else:
                filtered_df = df[df['Fund House'] == selected_amc]

            # Summary Table
            st.subheader("Fund Summary")
            fund_summary = filtered_df.groupby(['Scheme Name', 'Fund Type']).agg(
                NumTrades=('Trade Date', 'count'),
                TotalQuantityTraded=('Quantity Traded', 'sum'),
                AvgYield=('YieldatwhichTraded', 'mean')
            ).reset_index().reset_index(drop=True)
            st.dataframe(fund_summary.style.format({
                "TotalQuantityTraded": "{:,.2f}",
                "AvgYield": "{:.2f}"
            }))

            # Detailed Transactions
            st.subheader("Individual Transactions")
            show_details = st.checkbox("Show Individual Transactions")

            if show_details:
                page_size = 20
                page_number = st.number_input("Page Number", min_value=1, value=1)
                
                start_index = (page_number - 1) * page_size
                end_index = start_index + page_size
                
                st.dataframe(filtered_df.iloc[start_index:end_index].reset_index(drop=True)[[
                    'Fund House', 'Scheme Name', 'Fund Type', 'Trade Date', 'Quantity Traded', 'YieldatwhichTraded'
                ]].style.format({
                    "Quantity Traded": "{:,.2f}",
                    "YieldatwhichTraded": "{:.2f}"
                }))
