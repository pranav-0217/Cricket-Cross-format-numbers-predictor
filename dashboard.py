import streamlit as st
import pandas as pd
from pathlib import Path

from predictor import COUNTRY_NAMES, FORMATS, SPECS, Model, typical_errors

# Set page config
st.set_page_config(page_title="Cricket Analysis Dashboard", layout="wide", initial_sidebar_state="auto")

# CSS styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .title-main {
        color: #1f77b4;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title-main'>🏏 Cricket Performance Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("---")

# Data loading
@st.cache_data
def load_data():
    output_dir = Path("outputs")
    data = {}

    # Load CSVs
    ranking_files = {
        "all_players": "all_players_ranking.csv",
        "top100_batting": "top100_batting_rankings.csv",
        "top100_bowling": "top100_bowling_rankings.csv",
        "batting_1000runs": "top_batting_rankings_1000runs.csv",
        "bowling_50wkts": "top_bowling_rankings_50wkts.csv",
    }

    for key, filename in ranking_files.items():
        filepath = output_dir / filename
        if filepath.exists():
            data[key] = pd.read_csv(filepath)

    # Load images from log_checks
    log_checks_dir = output_dir / "log_checks"
    diagnostics_dir = output_dir / "diagnostics"

    data['log_checks_dir'] = log_checks_dir
    data['diagnostics_dir'] = diagnostics_dir

    return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio(
    "Select a section:",
    ["🎯 Try a prediction", "📊 Rankings", "📈 Performance Analysis", "📉 Distribution Analysis", "🔍 Player Search", "📋 Model Diagnostics"]
)

# ============= TRY A PREDICTION PAGE =============
@st.cache_resource
def get_model(kind):
    return Model(kind)


DEFAULTS = {
    "batting": {"Test": (35.0, 50.0), "ODI": (35.0, 85.0), "T20": (25.0, 130.0)},
    "bowling": {"Test": (3.0, 32.0), "ODI": (5.0, 32.0), "T20": (7.5, 25.0)},
}
STEPS = {"batting": (0.5, 1.0), "bowling": (0.1, 0.5)}

if page == "🎯 Try a prediction":
    st.header("Predict a player in another format")
    st.write(
        "Enter a player's numbers in one format and the model predicts the other two. "
        "It finds the 10 players whose numbers look most similar once the format and "
        "player attributes are taken out, and borrows their average ability. This is "
        "the k-NN transfer evaluated on the poster."
    )

    kind = st.radio("Discipline", ["Batting", "Bowling"], horizontal=True).lower()
    model = get_model(kind)
    name1, name2 = SPECS[kind]["stats"]
    known = st.selectbox("Format you know", FORMATS, index=1)

    c1, c2 = st.columns(2)
    (lo1, hi1), (lo2, hi2) = model.stat_ranges[known]
    step1, step2 = STEPS[kind]
    d1, d2 = DEFAULTS[kind][known]
    v1 = c1.number_input(f"{known} {name1.lower()}", float(lo1), float(hi1), d1, step1,
                         key=f"{kind}_{known}_1")
    v2 = c2.number_input(f"{known} {name2.lower()}", float(lo2), float(hi2), d2, step2,
                         key=f"{kind}_{known}_2")
    st.caption(f"Inputs are limited to the range of real {known} players in the data.")

    with st.expander("About the player (optional)"):
        roles = {"Batter": "batter", "All-rounder": "allrounder", "Bowler": "bowler"}
        role = st.selectbox("Role", list(roles), index=0 if kind == "batting" else 2)
        if kind == "batting":
            left = st.radio("Batting hand", ["Right", "Left"], horizontal=True) == "Left"
            spin = False
        else:
            left = st.radio("Bowling arm", ["Right", "Left"], horizontal=True) == "Left"
            spin = st.radio("Bowling type", ["Pace", "Spin"], horizontal=True) == "Spin"
        country = st.selectbox("Country", model.countries,
                               format_func=lambda c: COUNTRY_NAMES.get(c, c))
        first, last = (int(v) for v in model.ranges["debut_year"])
        debut = st.slider("Debut year", first, last, min(2015, last))
        age_debut = st.slider("Age at debut", 15, 38, 22)
        age_mid = st.slider("Age at career midpoint", 18, 42, 28)

    preds, neighbours = model.predict(known, v1, v2, role=roles[role], left=left, spin=spin,
                                      country=country, debut_year=debut,
                                      age_at_debut=age_debut, midpoint_age=age_mid)
    mae = typical_errors()
    err1, err2 = (mae[key] for key in SPECS[kind]["mae"])
    fmt1 = "{:.2f}" if kind == "bowling" else "{:.1f}"

    st.subheader("Predicted")
    for fmt, (p1, p2) in preds.items():
        st.markdown(f"**{fmt}**")
        a, b = st.columns(2)
        a.metric(name1, fmt1.format(p1))
        b.metric(name2, f"{p2:.1f}")
    st.caption(
        f"Typical miss for held-out players: about {fmt1.format(err1)} for {name1.lower()} "
        f"and {err2:.1f} for {name2.lower()} (mean absolute error, poster's held-out table)."
    )

    st.subheader(f"Most similar players in {known}")
    st.write(", ".join(neighbours["player_name"]))
    st.caption(
        "Similarity is measured after removing the format and the player attributes above. "
        "The same career-midpoint age is assumed in every format. Model: Bayesian "
        "multivariate mixed-effects model fitted to ESPN Cricinfo career statistics. "
        "Code: github.com/pranav-0217/Cricket-Cross-format-prediction"
    )

# ============= RANKINGS PAGE =============
elif page == "📊 Rankings":
    st.header("Player Rankings")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏏 Batting Rankings")
        ranking_type = st.selectbox(
            "Select Ranking Type (Batting):",
            ["All Players", "Top 100", "1000+ Runs"]
        )

        if ranking_type == "All Players" and "all_players" in data:
            df = data["all_players"].head(50)
            st.dataframe(df, width='stretch')
        elif ranking_type == "Top 100" and "top100_batting" in data:
            df = data["top100_batting"]
            st.dataframe(df, width='stretch')
        elif ranking_type == "1000+ Runs" and "batting_1000runs" in data:
            df = data["batting_1000runs"]
            st.dataframe(df, width='stretch')

    with col2:
        st.subheader("🎯 Bowling Rankings")
        bowl_type = st.selectbox(
            "Select Ranking Type (Bowling):",
            ["Top 100", "50+ Wickets"]
        )

        if bowl_type == "Top 100" and "top100_bowling" in data:
            st.dataframe(data["top100_bowling"], width='stretch')
        elif bowl_type == "50+ Wickets" and "bowling_50wkts" in data:
            st.dataframe(data["bowling_50wkts"], width='stretch')
        else:
            st.info("Bowling rankings data not available")

    st.caption(
        "Ranked by the format-only player random effect (psi), which adjusts for "
        "format but not for role or country. Higher is better in both panels."
    )

# ============= PERFORMANCE ANALYSIS PAGE =============
elif page == "📈 Performance Analysis":
    st.header("Performance Metrics by Format")

    output_dir = Path("outputs")

    col1, col2 = st.columns(2)

    with col1:
        # SR vs Avg
        sr_avg_path = output_dir / "sr_vs_avg_by_format.png"
        if sr_avg_path.exists():
            st.subheader("Strike Rate vs Average (Batting)")
            st.image(str(sr_avg_path), width='stretch')

        # Bowling Econ vs Avg
        econ_avg_path = output_dir / "bowling_econ_vs_avg_by_format.png"
        if econ_avg_path.exists():
            st.subheader("Economy vs Average (Bowling)")
            st.image(str(econ_avg_path), width='stretch')

    with col2:
        # SR vs Econ
        sr_econ_path = output_dir / "bowling_sr_vs_econ_by_format.png"
        if sr_econ_path.exists():
            st.subheader("Strike Rate vs Economy (Bowling)")
            st.image(str(sr_econ_path), width='stretch')

# ============= DISTRIBUTION ANALYSIS PAGE =============
elif page == "📉 Distribution Analysis":
    st.header("Log-Normality Distribution Checks")
    st.info("These visualizations show whether log-transforming the metrics improves normality for the statistical model")

    log_checks_dir = data.get('log_checks_dir')

    if log_checks_dir and log_checks_dir.exists():
        # The four modelled outcomes. The previous list named bowling_sr, which
        # is not an outcome in either model.
        metrics = ["batting_avg", "batting_sr", "bowling_econ", "bowling_avg"]

        for metric in metrics:
            st.subheader(f"{metric.replace('_', ' ').title()} - Log Normality Analysis")

            # Find related files
            metric_files = {}
            for file in log_checks_dir.glob(f"{metric}*"):
                if "hist" in file.name:
                    if "eps" in file.name:
                        eps_val = file.name.split("eps_")[1].split("_")[0]
                        key = f"hist_eps_{eps_val}"
                    else:
                        key = f"hist_{file.name.split('_')[-2]}"
                    metric_files[key] = file
                elif "qq" in file.name:
                    if "eps" in file.name:
                        eps_val = file.name.split("eps_")[1].split("_")[0]
                        key = f"qq_eps_{eps_val}"
                    else:
                        key = f"qq_{file.name.split('_')[-2]}"
                    metric_files[key] = file

            if metric_files:
                cols = st.columns(len(metric_files))
                for idx, (file_type, filepath) in enumerate(sorted(metric_files.items())):
                    with cols[idx]:
                        st.caption(file_type)
                        st.image(str(filepath), width='stretch')
            else:
                st.warning(f"No distribution checks found for {metric}")

            st.markdown("---")
    else:
        st.warning("Log checks directory not found")

# ============= PLAYER SEARCH PAGE =============
elif page == "🔍 Player Search":
    st.header("Player Performance Lookup")

    if "all_players" in data:
        df = data["all_players"]

        col1, col2 = st.columns(2)

        with col1:
            player_name = st.text_input("Search for a player:")

        with col2:
            metric = st.selectbox("Sort by:", df.columns[1:] if len(df.columns) > 1 else ["Name"])

        if player_name:
            # Search the name column explicitly. Column 0 is "rank" (an integer)
            # in the ranking exports, so indexing by position raised
            # AttributeError on the .str accessor.
            if "player_name" in df.columns:
                name_col = df["player_name"]
            else:
                text_cols = df.select_dtypes(include="object").columns
                name_col = df[text_cols[0]] if len(text_cols) else df.iloc[:, 0]
            filtered = df[name_col.astype(str).str.contains(player_name, case=False, na=False)]
            if not filtered.empty:
                st.dataframe(filtered, width='stretch')

                # Display stats
                if len(filtered) == 1:
                    player_data = filtered.iloc[0]
                    cols = st.columns(len(player_data))
                    for idx, (col, val) in enumerate(player_data.items()):
                        with cols[idx]:
                            st.metric(col, val)
            else:
                st.warning("Player not found")
        else:
            # Show top players
            st.write("Top 10 Players:")
            st.dataframe(df.head(10), width='stretch')
    else:
        st.info("Player data not available")

# ============= DIAGNOSTICS PAGE =============
elif page == "📋 Model Diagnostics":
    st.header("Model Diagnostics & Validation")
    st.info("Bayesian model convergence and posterior validation checks")

    diagnostics_dir = data.get('diagnostics_dir')

    if diagnostics_dir and diagnostics_dir.exists():
        models = ["batting", "bowling"]

        for model in models:
            st.subheader(f"{model.title()} Model Diagnostics")

            diagnostic_types = ["trace", "rhat", "posteriors", "ppc", "player_corr"]
            cols = st.columns(len(diagnostic_types))

            for idx, diag_type in enumerate(diagnostic_types):
                filepath = diagnostics_dir / f"{model}_{diag_type}.png"
                if filepath.exists():
                    with cols[idx]:
                        st.caption(diag_type.upper())
                        st.image(str(filepath), width='stretch')

            st.markdown("---")
    else:
        st.warning("Diagnostics directory not found")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    <small>Cricket Performance Analytics Dashboard | Mixed-Effects Bayesian Model</small>
    </div>
    """,
    unsafe_allow_html=True
)
