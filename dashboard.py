import streamlit as st

from predictor import COUNTRY_NAMES, FORMATS, SPECS, Model, typical_errors

st.set_page_config(page_title="Cricket cross-format predictor", page_icon="🏏", layout="centered")


@st.cache_resource
def get_model(kind):
    return Model(kind)


DEFAULTS = {
    "batting": {"Test": (35.0, 50.0), "ODI": (35.0, 85.0), "T20": (25.0, 130.0)},
    "bowling": {"Test": (3.0, 32.0), "ODI": (5.0, 32.0), "T20": (7.5, 25.0)},
}
STEPS = {"batting": (0.5, 1.0), "bowling": (0.1, 0.5)}

st.title("Cricket cross-format predictor")

kind = st.radio("Discipline", ["Batting", "Bowling"], horizontal=True).lower()
model = get_model(kind)
name1, name2 = SPECS[kind]["stats"]
chosen = st.multiselect("Formats you know", FORMATS, default=["ODI"], max_selections=2)
if not chosen:
    st.info("Pick at least one format.")
    st.stop()

known = {}
step1, step2 = STEPS[kind]
for fmt in (f for f in FORMATS if f in chosen):
    c1, c2 = st.columns(2)
    (lo1, hi1), (lo2, hi2) = model.stat_ranges[fmt]
    d1, d2 = DEFAULTS[kind][fmt]
    known[fmt] = (
        c1.number_input(f"{fmt} {name1.lower()}", float(lo1), float(hi1), d1, step1, key=f"{kind}_{fmt}_1"),
        c2.number_input(f"{fmt} {name2.lower()}", float(lo2), float(hi2), d2, step2, key=f"{kind}_{fmt}_2"),
    )

with st.expander("Player details (optional)"):
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

preds, neighbours = model.predict(known, role=roles[role], left=left, spin=spin,
                                  country=country, debut_year=debut,
                                  age_at_debut=age_debut, midpoint_age=age_mid)
err1, err2 = (typical_errors()[key] for key in SPECS[kind]["mae"])
fmt1 = "{:.2f}" if kind == "bowling" else "{:.1f}"

st.subheader("Predicted")
for fmt, (p1, p2) in preds.items():
    st.markdown(f"**{fmt}**")
    a, b = st.columns(2)
    a.metric(name1, fmt1.format(p1), help=f"Typical error: ±{fmt1.format(err1)}")
    b.metric(name2, f"{p2:.1f}", help=f"Typical error: ±{err2:.1f}")

st.subheader("Most similar players")
for fmt, top in neighbours.items():
    names = ", ".join(top["player_name"])
    st.write(f"**{fmt}:** {names}" if len(neighbours) > 1 else names)
