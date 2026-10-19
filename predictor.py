"""Predict a new player's numbers in another format with the paper's k-NN latent transfer."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "new" / "processed"
ME = ROOT / "outputs" / "mixed_effects"
FORMATS = ["Test", "ODI", "T20"]
K = 10

SPECS = {
    "batting": {"labels": ("log_avg", "log_sr"), "psi": ("psi_log_avg", "psi_log_sr"),
                "stats": ("Batting average", "Strike rate"), "mae": ("avg", "sr")},
    "bowling": {"labels": ("log_econ", "log_avg"), "psi": ("psi_log_econ", "psi_log_avg"),
                "stats": ("Economy rate", "Bowling average"), "mae": ("econ", "bowl_avg")},
}

COUNTRY_NAMES = {
    "AUS": "Australia", "AFG": "Afghanistan", "BAN": "Bangladesh", "CAN": "Canada",
    "ENG": "England", "HKG": "Hong Kong", "IND": "India", "IRE": "Ireland", "KENYA": "Kenya",
    "NAM": "Namibia", "NED": "Netherlands", "NEP": "Nepal", "NZ": "New Zealand", "OMA": "Oman",
    "PAK": "Pakistan", "PNG": "Papua New Guinea", "SA": "South Africa", "SCOT": "Scotland",
    "SL": "Sri Lanka", "UAE": "United Arab Emirates", "USA": "United States",
    "WI": "West Indies", "ZIM": "Zimbabwe", "OTHER_OR_MULTI": "Other",
}


def scale_of(rows, raw, scaled):
    """Mean and standard deviation that turned `raw` into `scaled` when the data were built."""
    sd, mean = np.polyfit(rows[scaled], rows[raw], 1)
    return mean, sd


class Model:
    def __init__(self, kind):
        spec = SPECS[kind]
        meta = json.loads((DATA / f"{kind}_joint_enriched_meta.json").read_text())
        self.kind, self.predictors = kind, meta["predictors"]
        self.knot = meta["age_curve"]["knot_age_years"]

        params = pd.read_csv(ME / f"joint_{kind}_enriched_params_mean.csv").iloc[0]
        self.alpha = np.array([params[f"alpha_{lab}"] for lab in spec["labels"]])
        self.beta = np.array([[params[f"beta_{p}_{lab}"] for p in self.predictors]
                              for lab in spec["labels"]])

        rows = pd.read_csv(DATA / f"{kind}_joint_enriched.csv")
        psi = pd.read_csv(ME / f"{kind}_enriched_player_rankings.csv")
        rows = rows.merge(psi[["player_id", *spec["psi"]]], on="player_id", how="left")
        fitted = self.alpha + rows[self.predictors].to_numpy(dtype=float) @ self.beta.T
        rows["res1"], rows["res2"] = rows["y1"] - fitted[:, 0], rows["y2"] - fitted[:, 1]
        self.rows = rows

        self.debut_scale = scale_of(rows, "debut_year", "debut_year_scaled")
        self.age_debut_scale = scale_of(rows, "age_at_debut", "age_at_debut_scaled")
        self.age_scale = scale_of(rows, "avg_age", "age_scaled")
        self.countries = ["AUS"] + sorted(
            (p.removeprefix("country_") for p in self.predictors
             if p.startswith("country_") and p != "country_OTHER_OR_MULTI"),
            key=lambda c: COUNTRY_NAMES.get(c, c)) + ["OTHER_OR_MULTI"]
        self.ranges = {col: (rows[col].min(), rows[col].max())
                       for col in ("debut_year", "age_at_debut", "avg_age")}
        stat_cols = ("avg", "sr") if kind == "batting" else ("econ", "avg")
        self.stat_ranges = {
            fmt: tuple((np.floor(r[c].min() * 10) / 10, np.ceil(r[c].max() * 10) / 10) for c in stat_cols)
            for fmt, r in rows.groupby("format")
        }

    def x_vector(self, fmt, role="batter", left=False, spin=False, country="AUS",
                 debut_year=None, age_at_debut=None, midpoint_age=None):
        """Predictor row for a player; missing numbers default to the data average."""
        x = dict.fromkeys(self.predictors, 0.0)
        x["format_odi"], x["format_t20"] = float(fmt == "ODI"), float(fmt == "T20")
        for value, name, (mean, sd) in ((debut_year, "debut_year_scaled", self.debut_scale),
                                        (age_at_debut, "age_at_debut_scaled", self.age_debut_scale)):
            x[name] = 0.0 if value is None else (value - mean) / sd
        x["role_allrounder"], x["role_bowler"] = float(role == "allrounder"), float(role == "bowler")
        if self.kind == "batting":
            x["is_lefthanded"] = float(left)
        else:
            x["bowling_hand_left"], x["bowling_type_spin"] = float(left), float(spin)
        if country != "AUS":
            x[f"country_{country}" if f"country_{country}" in x else "country_OTHER_OR_MULTI"] = 1.0
        mean, sd = self.age_scale
        age = mean if midpoint_age is None else midpoint_age
        x["age_scaled"], x["age_hinge"] = (age - mean) / sd, max(0.0, age - self.knot) / sd
        return np.array([x[p] for p in self.predictors])

    def transfer(self, known_fmt, log_stats, x_known, x_targets, exclude_id=None, k=K):
        """Borrow the average ability of the k players closest in format-adjusted residuals."""
        query = np.asarray(log_stats) - (self.alpha + self.beta @ x_known)
        cand = self.rows[self.rows["format"] == known_fmt]
        if exclude_id is not None:
            cand = cand[cand["player_id"] != exclude_id]
        dists = np.sqrt((cand["res1"].to_numpy() - query[0]) ** 2
                        + (cand["res2"].to_numpy() - query[1]) ** 2)
        top = cand.iloc[np.argsort(dists)[:k]]
        psi_hat = top[list(SPECS[self.kind]["psi"])].mean().to_numpy()
        preds = {fmt: np.exp(self.alpha + self.beta @ x + psi_hat) for fmt, x in x_targets.items()}
        return preds, top

    def predict(self, known_fmt, stat1, stat2, **player):
        x_known = self.x_vector(known_fmt, **player)
        targets = {f: self.x_vector(f, **player) for f in FORMATS if f != known_fmt}
        return self.transfer(known_fmt, np.log([stat1, stat2]), x_known, targets)


def typical_errors():
    """Mean absolute error of the k-NN transfer for held-out players, by statistic."""
    ev = pd.read_csv(ME / "lopo_kfold_evaluation.csv")
    ev = ev[ev["metric_set"].str.startswith("LOPO + k-NN transfer")]
    return {row.metric_set.split("|")[1].strip(): float(row.MAE) for row in ev.itertuples()}
