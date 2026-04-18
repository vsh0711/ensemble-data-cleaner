import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from cleaner import EnsembleDataCleaner
from sample_data import generate_sample_data

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ensemble Data Cleaner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — minimal, clean ────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-num  { font-size: 2rem; font-weight: 600; color: #1a1a2e; }
    .metric-lbl  { font-size: 0.8rem; color: #6c757d; margin-top: 4px; }
    .anomaly-tag { color: #dc3545; font-weight: 600; }
    .clean-tag   { color: #198754; font-weight: 600; }
    .stDownloadButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Ensemble Data Cleaner")
    st.markdown("*AI-powered data quality detection*")
    st.divider()

    st.markdown("### Detection settings")
    contamination = st.slider(
        "Expected anomaly rate",
        min_value=0.01, max_value=0.20,
        value=0.05, step=0.01,
        help="What fraction of your data do you expect to be anomalous? "
             "5% is a safe default for most datasets."
    )
    zscore_thresh = st.slider(
        "Z-score threshold",
        min_value=2.0, max_value=5.0,
        value=3.0, step=0.5,
        help="Flag values this many standard deviations from the mean. "
             "3.0 is standard. Lower = stricter."
    )
    st.divider()

    st.markdown("### How it works")
    st.markdown("""
Three ML methods vote on each row:
- **Isolation Forest** — global outliers
- **Z-score** — univariate extremes  
- **Local Outlier Factor** — contextual anomalies

A row is flagged only when **2+ methods agree** — reducing false positives.
    """)
    st.divider()
    st.markdown("Built by **Vishalini Satheesh**")
    st.markdown("M.E. CSE (OR) · Anna University CEG")
    st.markdown("[GitHub](https://github.com/vishalinisatheesh) · "
                "[LinkedIn](https://linkedin.com/in/vishalinisatheesh)")


# ── Main area ───────────────────────────────────────────────────────────
st.title("Ensemble AI Data Cleaning Tool")
st.markdown(
    "Upload any CSV to detect anomalies using an ensemble of three ML methods. "
    "No manual rules — the models learn what 'normal' looks like and flag deviations."
)

# ── Data source selection ───────────────────────────────────────────────
tab_demo, tab_upload = st.tabs(["Use demo data", "Upload your CSV"])

df_input = None

with tab_demo:
    st.markdown("#### Preloaded banking demo dataset")
    st.markdown(
        "Synthetic BFSI-style data with injected anomalies — account balances, "
        "transaction amounts, credit scores, risk ratings across 500 customers. "
        "Some records have been deliberately corrupted to simulate real data quality issues."
    )
    n_rows = st.selectbox("Dataset size", [200, 500, 1000], index=1)
    anomaly_pct = st.selectbox("Injected anomaly rate", ["5%", "10%", "15%"], index=0)
    pct_map = {"5%": 0.05, "10%": 0.10, "15%": 0.15}

    if st.button("Generate demo data", type="primary"):
        df_input = generate_sample_data(
            n=n_rows,
            anomaly_fraction=pct_map[anomaly_pct],
            random_state=42
        )
        st.session_state['df_input'] = df_input
        st.success(f"Generated {n_rows} rows with ~{anomaly_pct} injected anomalies ✓")

    if 'df_input' in st.session_state and df_input is None:
        df_input = st.session_state['df_input']

with tab_upload:
    st.markdown("#### Upload your own CSV")
    st.markdown("Numeric columns are analysed automatically. Text columns are preserved but not analysed.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.session_state['df_input'] = df_input
        st.success(f"Uploaded: {df_input.shape[0]:,} rows × {df_input.shape[1]} columns ✓")

# ── Preview uploaded/demo data ──────────────────────────────────────────
if df_input is not None:
    with st.expander("Preview data", expanded=False):
        st.dataframe(df_input.head(20), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{df_input.shape[0]:,}")
        col2.metric("Columns", df_input.shape[1])
        col3.metric("Numeric columns",
                    len(df_input.select_dtypes(include=[np.number]).columns))

    # ── Run detection ────────────────────────────────────────────────────
    st.divider()
    run_col, _ = st.columns([1, 3])
    run_btn = run_col.button("Run anomaly detection", type="primary",
                              use_container_width=True)

    if run_btn or 'last_results' in st.session_state:

        if run_btn:
            with st.spinner("Running ensemble detection — Isolation Forest + Z-score + LOF..."):
                cleaner = EnsembleDataCleaner(
                    contamination=contamination,
                    zscore_threshold=zscore_thresh
                )
                try:
                    results = cleaner.fit_detect(df_input)
                    summary = cleaner.summary()
                    clean_df   = cleaner.get_clean()
                    anomaly_df = cleaner.get_anomalies()
                    st.session_state['last_results'] = {
                        'results': results,
                        'summary': summary,
                        'clean_df': clean_df,
                        'anomaly_df': anomaly_df,
                    }
                except ValueError as e:
                    st.error(f"Error: {e}")
                    st.stop()

        cached = st.session_state['last_results']
        results    = cached['results']
        summary    = cached['summary']
        clean_df   = cached['clean_df']
        anomaly_df = cached['anomaly_df']

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown("### Data quality report")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total records",  f"{summary['total']:,}")
        m2.metric("Clean records",  f"{summary['n_clean']:,}",
                  delta=f"{100 - summary['anomaly_pct']:.1f}% of data")
        m3.metric("Anomalies found", f"{summary['n_anomaly']:,}",
                  delta=f"-{summary['anomaly_pct']:.1f}%",
                  delta_color="inverse")
        m4.metric("Methods agree on", f"{summary['n_anomaly']:,} rows",
                  delta="2+ method consensus")

        # ── Method breakdown ──────────────────────────────────────────────
        st.markdown("#### What each method found")
        method_cols = st.columns(3)
        method_info = {
            'Isolation Forest':     "Global outliers via random partitioning",
            'Z-score':              "Univariate extremes (>3 std deviations)",
            'Local Outlier Factor': "Contextual anomalies in local neighbourhoods",
        }
        for i, (method, count) in enumerate(summary['method_counts'].items()):
            with method_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-num">{count:,}</div>
                  <div class="metric-lbl">{method}</div>
                  <div style="font-size:0.72rem;color:#6c757d;margin-top:6px">
                    {method_info[method]}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        # ── Visualisations ────────────────────────────────────────────────
        st.markdown("#### Visualisations")
        viz1, viz2 = st.columns(2)

        with viz1:
            # Anomaly score distribution
            fig = px.histogram(
                results, x='anomaly_score',
                color='anomaly',
                color_discrete_map={0: '#378ADD', 1: '#E24B4A'},
                labels={'anomaly_score': 'Anomaly score',
                        'anomaly': 'Is anomaly'},
                title='Anomaly score distribution',
                barmode='overlay',
                opacity=0.75,
                nbins=40
            )
            fig.update_layout(
                legend_title="Flagged",
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
           )
            fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
            st.plotly_chart(fig, use_container_width=True)

        with viz2:
            numeric_cols = [
            c for c in results.select_dtypes(include=[np.number]).columns
                if c not in ['iso_flag', 'zscore_flag', 'lof_flag',
                     'anomaly_score', 'anomaly', 'is_anomaly_injected']
            ]

            if len(numeric_cols) >= 2:
                # Let user pick which two columns to plot
                st.markdown("**Choose axes for scatter plot**")
                axis_col1, axis_col2 = st.columns(2)
                with axis_col1:
                    col_x = st.selectbox("X axis", numeric_cols, index=0)
                with axis_col2:
                    col_y = st.selectbox("Y axis", numeric_cols,
                                    index=min(1, len(numeric_cols)-1))

                plot_df = results[[col_x, col_y, 'anomaly', 'anomaly_score']].copy()
                plot_df['Status'] = plot_df['anomaly'].map({0: 'Clean', 1: 'Anomaly'})

                fig2 = px.scatter(
                    plot_df, x=col_x, y=col_y,
                    color='Status',
                    color_discrete_map={'Clean': '#378ADD', 'Anomaly': '#E24B4A'},
                    opacity=0.6,
                    title=f'Anomaly scatter — {col_x} vs {col_y}',
                    size='anomaly_score',
                    size_max=12,
                    hover_data=['anomaly_score']
            )
                fig2.update_layout(
                    legend_title="Status",
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20)
            )
                fig2.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
                fig2.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
                st.plotly_chart(fig2, use_container_width=True)

            elif len(numeric_cols) == 1:
            # Only one numeric column — show distribution instead
                fig2 = px.histogram(
                    results, x=numeric_cols[0],
                    color=results['anomaly'].map({0: 'Clean', 1: 'Anomaly'}),
                    color_discrete_map={'Clean': '#378ADD', 'Anomaly': '#E24B4A'},
                    title=f'Distribution — {numeric_cols[0]}',
                    barmode='overlay',
                    opacity=0.75
                )
                fig2.update_layout(height=350, plot_bgcolor='white')
                st.plotly_chart(fig2, use_container_width=True)

            else:
              st.info("Need at least one numeric column for scatter plot.")

        # Top drift columns
        if summary['top_drift_cols']:
            st.markdown("#### Columns with highest anomalous deviation")
            st.markdown(
                "These features show the greatest difference between anomalous "
                "and clean records — likely root causes of data quality issues."
            )
            drift_data = []
            for col in summary['top_drift_cols']:
                clean_mean   = clean_df[col].mean()   if col in clean_df.columns else 0
                anomaly_mean = anomaly_df[col].mean() if col in anomaly_df.columns else 0
                drift_data.append({
                    'Column': col,
                    'Clean mean': round(clean_mean, 2),
                    'Anomaly mean': round(anomaly_mean, 2),
                    'Deviation': round(abs(anomaly_mean - clean_mean), 2)
                })
            st.dataframe(pd.DataFrame(drift_data), use_container_width=True)

        # ── Anomalous rows table ──────────────────────────────────────────
        st.markdown("#### Flagged records")
        st.markdown(
            f"Showing {min(50, len(anomaly_df))} of {len(anomaly_df):,} anomalous records. "
            "Download the full clean dataset below."
        )
        st.dataframe(
            anomaly_df.head(50).style.background_gradient(
                subset=['anomaly_score'], cmap='Reds'
            ),
            use_container_width=True
        )

        # ── Download section ──────────────────────────────────────────────
        st.divider()
        st.markdown("### Download results")
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            csv_clean = clean_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download clean data (CSV)",
                data=csv_clean,
                file_name="clean_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dl2:
            csv_anomaly = anomaly_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download anomalies (CSV)",
                data=csv_anomaly,
                file_name="anomalies.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dl3:
            csv_full = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download full results with flags",
                data=csv_full,
                file_name="full_results_with_flags.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("Generate demo data or upload a CSV to get started.")