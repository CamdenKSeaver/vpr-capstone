from __future__ import annotations

import argparse
import math
from pathlib import Path

from scmrepo.git import Git

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .fullDataPipeline import run_for_season as build_preprocessed_season
    from .season_support import add_season_args, resolve_requested_seasons
except ImportError:
    from fullDataPipeline import run_for_season as build_preprocessed_season
    from season_support import add_season_args, resolve_requested_seasons

RANDOM_SEED = 42

MODEL_CONFIG = {
    "event_weights": {
        "serve": 0.1,
        "reception": 0.2,
        "set": 0.2,
        "attack": 0.3,
        "kill": 1.0,
        "ace": 1.0,
        "dig": 0.25,
        "block": 0.8,
        "attack_error": 1.0,
        "service_error": 1.0,
        "reception_error": 1.0,
        "set_error": 1.0,
        "block_error": 0.8,
        "dig_error": 0.8,
    },
    "team_phase_context": {
        "attack_efficiency_weight": 0.65,
        "kill_rate_weight": 0.25,
        "attack_error_weight": 0.1,
        "shrinkage_denominator": 500.0,
    },
    "context_multiplier": {
        "attack_factor": 0.08,
        "defense_factor": 0.08,
        "min_multiplier": 0.75,
        "max_multiplier": 1.25,
    },
    "server_toughness": {
        "serve_point_win_weight": 0.5,
        "serve_ace_weight": 0.35,
        "forced_reception_error_weight": 0.15,
        "serve_error_weight": 0.25,
        "shrinkage_denominator": 100.0,
    },
    "serve_receive_value": {
        "receive_point_win_weight": 0.35,
        "first_ball_kill_weight": 0.25,
        "first_ball_middle_attack_weight": 0.1,
        "first_ball_attack_weight": 0.15,
        "reception_error_weight": 0.15,
        "shrinkage_denominator": 100.0,
    },
    "aux_binary_model": {
        "max_iter": 400,
        "solver": "lbfgs",
    },
    "role_blend_weights": {
        "libero_ds": {"event": 0.55, "tabular": 0.15, "serve_receive": 0.3},
        "outside": {"event": 0.6, "tabular": 0.2, "serve_receive": 0.2},
        "opposite": {"event": 0.7, "tabular": 0.25, "serve_receive": 0.05},
        "setter": {"event": 0.7, "tabular": 0.25, "serve_receive": 0.05},
        "middle": {"event": 0.75, "tabular": 0.25, "serve_receive": 0.0},
        "unknown": {"event": 0.75, "tabular": 0.25, "serve_receive": 0.0},
    },
    "score_scaling": {
        "tanh_scale": 3.0,
        "cdf_scale": 100.0,
    },
    "role_inference_thresholds": {
        "high_attack_total_attacks": 150,
        "high_attack_kills": 80,
        "high_attack_event_kills": 80,
        "setter_assists": 500,
        "setter_event_sets": 1000,
        "libero_max_total_attacks": 50,
        "libero_max_assists": 200,
    },
    "tabular_value_model": {
        "min_train_rallies": 100,
    },
    "audit_thresholds": {
        "libero_attack_volume": 100,
        "libero_event_kills": 50,
        "high_rank_cutoff": 100,
        "low_volume_rallies": 500,
        "low_efficiency_attack_volume": 300,
        "low_efficiency_hit_pct": 0.2,
    },
    "data_split": {
        "train_fraction": 0.7,
        "valid_fraction": 0.85,
    },
    "role_blend_default": {
        "event": 0.75,
        "tabular": 0.25,
        "serve_receive": 0.0,
    },
    "team_strength_prior": {
        "weight": 0.3,
    },
    "wp_models": {
        "logistic": {"max_iter": 500, "solver": "lbfgs"},
        "mlp": {
            "hidden_layer_sizes": [64, 32],
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "early_stopping": True,
            "max_iter": 150,
            "n_iter_no_change": 10,
        },
    },
}

WP_NUMERIC_FEATURES = [
    "set_number",
    "home_sets",
    "away_sets",
    "home_points",
    "away_points",
    "home_score_diff",
    "total_points",
    "home_server",
    "away_server",
    "is_fifth_set",
    "is_late_set",
    "is_deuce_or_later",
]
WP_CATEGORICAL_FEATURES = ["server_side"]


def save_csv(df: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    """Write a dataframe to disk and print a short status line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    df.to_csv(path, index=False)
    print(f"saved: {path} ({len(df):,} rows)")
    return path


def normal_cdf(x: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype="float64")
    return 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))


def role_zscore(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    """Standardize a value within each role bucket before blending roles together."""
    df = df.copy()
    means = df.groupby("role_family", observed=True)[value_col].transform("mean")
    stds = (
        df.groupby("role_family", observed=True)[value_col]
        .transform("std")
        .replace(0, np.nan)
    )
    df[out_col] = ((df[value_col] - means) / stds).fillna(0.0)
    return df


def standard_z(s: pd.Series) -> pd.Series:
    """Return a plain z-score with a safe zero fallback for constant inputs."""
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
    return (s - s.mean()) / sd


def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide safely and fall back to zero when the denominator is missing or zero."""
    numerator = pd.to_numeric(num, errors="coerce")
    denominator = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return (numerator / denominator).fillna(0.0)


def infer_role_family(
    df: pd.DataFrame,
    source_role_col: str = "role_family",
    model_config: dict[str, object] | None = None,
) -> pd.Series:
    # Roster position labels are useful, but the stat profile is a safer fallback
    # when a player has a hybrid label or an inconsistent source row.
    cfg = model_config or MODEL_CONFIG
    role_thresholds = cfg["role_inference_thresholds"]
    base = (
        df.get(source_role_col, pd.Series("unknown", index=df.index))
        .fillna("unknown")
        .astype("string")
    )
    pos = df.get("pos", pd.Series("", index=df.index)).fillna("").astype("string").str.lower()

    numeric_defaults = {
        "box_total_attacks": 0,
        "box_kills": 0,
        "box_assists": 0,
        "event_count_kill": 0,
        "event_count_set": 0,
    }
    vals = {}
    for col, default in numeric_defaults.items():
        vals[col] = pd.to_numeric(
            df.get(col, pd.Series(default, index=df.index)),
            errors="coerce",
        ).fillna(0)

    high_attack_volume = (
        vals["box_total_attacks"].ge(int(role_thresholds["high_attack_total_attacks"]))
        | vals["box_kills"].ge(int(role_thresholds["high_attack_kills"]))
        | vals["event_count_kill"].ge(int(role_thresholds["high_attack_event_kills"]))
    )
    setterish = (
        pos.str.fullmatch(r"s", na=False)
        | vals["box_assists"].ge(int(role_thresholds["setter_assists"]))
        | vals["event_count_set"].ge(int(role_thresholds["setter_event_sets"]))
    )
    middleish = pos.str.contains(r"\b(?:mb|mh|m|middle)\b", regex=True, na=False)
    oppositeish = pos.str.contains(r"\b(?:rs|opp|opposite|right side)\b", regex=True, na=False)
    liberoish = pos.str.contains(r"\b(?:l|ds|libero)\b", regex=True, na=False)

    inferred = pd.Series(base.to_numpy(), index=df.index, dtype="string")
    inferred = inferred.mask(setterish, "setter")
    inferred = inferred.mask(high_attack_volume & middleish & ~setterish, "middle")
    inferred = inferred.mask(high_attack_volume & oppositeish & ~setterish, "opposite")
    inferred = inferred.mask(high_attack_volume & ~middleish & ~oppositeish & ~setterish, "outside")
    inferred = inferred.mask(
        vals["box_total_attacks"].lt(int(role_thresholds["libero_max_total_attacks"]))
        & vals["box_assists"].lt(int(role_thresholds["libero_max_assists"]))
        & liberoish
        & ~setterish,
        "libero_ds",
    )
    return inferred.fillna("unknown")


def add_basic_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable rate features used by audits and downstream tabular models."""
    df = df.copy()
    for col in [
        "box_kills",
        "box_errors",
        "box_total_attacks",
        "box_aces",
        "box_serr",
        "box_digs",
        "box_rerr",
        "box_assists",
        "event_rallies",
        "event_touches",
        "fb_receptions",
        "fb_reception_errors",
        "fb_kills",
        "fb_middle_attacks",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["calc_hit_pct"] = safe_rate(df["box_kills"] - df["box_errors"], df["box_total_attacks"])
    df["kill_rate_per_attempt"] = safe_rate(df["box_kills"], df["box_total_attacks"])
    df["attack_error_rate"] = safe_rate(df["box_errors"], df["box_total_attacks"])
    df["serve_error_rate"] = safe_rate(df["box_serr"], df["box_aces"] + df["box_serr"])
    df["box_reception_error_rate"] = (
        safe_rate(df["box_rerr"], df["box_retatt"])
        if "box_retatt" in df.columns
        else 0.0
    )
    df["touches_per_rally"] = safe_rate(df["event_touches"], df["event_rallies"])
    df["fb_reception_error_rate"] = safe_rate(df["fb_reception_errors"], df["fb_receptions"])
    df["fb_kill_rate"] = safe_rate(df["fb_kills"], df["fb_receptions"])
    df["fb_middle_attack_rate"] = safe_rate(df["fb_middle_attacks"], df["fb_receptions"])
    return df


def make_wp_state_features(rally_table: pd.DataFrame, state: str) -> pd.DataFrame:
    """Build scoreboard-state features for the win-probability model."""
    if state not in {"before", "after"}:
        raise ValueError("state must be 'before' or 'after'")

    if state == "before":
        set_number = rally_table["set"]
        home_sets = rally_table["home_sets_before"]
        away_sets = rally_table["away_sets_before"]
        home_points = rally_table["score_home_start"]
        away_points = rally_table["score_away_start"]
        server_side = rally_table["server_side"]
    else:
        set_number = rally_table["set_after"]
        home_sets = rally_table["home_sets_after"]
        away_sets = rally_table["away_sets_after"]
        home_points = rally_table["score_home_after_state"]
        away_points = rally_table["score_away_after_state"]
        server_side = rally_table["server_side_after"]

    features = pd.DataFrame(
        {
            "set_number": pd.to_numeric(set_number, errors="coerce").fillna(0),
            "home_sets": pd.to_numeric(home_sets, errors="coerce").fillna(0),
            "away_sets": pd.to_numeric(away_sets, errors="coerce").fillna(0),
            "home_points": pd.to_numeric(home_points, errors="coerce").fillna(0),
            "away_points": pd.to_numeric(away_points, errors="coerce").fillna(0),
            "server_side": server_side.fillna("unknown").astype("string"),
        },
        index=rally_table.index,
    )
    features["home_score_diff"] = features["home_points"] - features["away_points"]
    features["total_points"] = features["home_points"] + features["away_points"]
    features["home_server"] = features["server_side"].eq("home").astype("int8")
    features["away_server"] = features["server_side"].eq("away").astype("int8")
    features["is_fifth_set"] = features["set_number"].eq(5).astype("int8")
    features["is_late_set"] = (
        (features["set_number"].lt(5) & features["total_points"].ge(38))
        | (features["set_number"].eq(5) & features["total_points"].ge(20))
    ).astype("int8")
    features["is_deuce_or_later"] = (
        (features["home_points"].ge(24) | features["away_points"].ge(24))
        & (features["home_score_diff"].abs().le(1))
    ).astype("int8")
    return features[WP_NUMERIC_FEATURES + WP_CATEGORICAL_FEATURES]


def make_time_split(
    rally_table: pd.DataFrame,
    model_config: dict[str, object] | None = None,
) -> pd.Series:
    cfg = model_config or MODEL_CONFIG
    split_cfg = cfg["data_split"]
    dates = pd.to_datetime(rally_table["date"], errors="coerce")
    if dates.notna().sum() < len(rally_table) * 0.5:
        order_key = pd.to_numeric(rally_table["contestid"], errors="coerce")
    else:
        order_key = dates

    contest_order = (
        pd.DataFrame({"contestid": rally_table["contestid"], "order_key": order_key})
        .drop_duplicates("contestid")
        .sort_values(["order_key", "contestid"], kind="mergesort")
        .reset_index(drop=True)
    )
    n = len(contest_order)
    train_cut = int(n * float(split_cfg["train_fraction"]))
    valid_cut = int(n * float(split_cfg["valid_fraction"]))
    contest_order["split"] = "test"
    contest_order.loc[: max(train_cut - 1, -1), "split"] = "train"
    contest_order.loc[train_cut: max(valid_cut - 1, train_cut - 1), "split"] = "valid"
    return rally_table["contestid"].map(contest_order.set_index("contestid")["split"]).fillna("test")


def build_wp_model(
    model_type: str = "mlp",
    model_config: dict[str, object] | None = None,
) -> Pipeline:
    cfg = model_config or MODEL_CONFIG
    wp_model_cfg = cfg["wp_models"][model_type]
    numeric_pipe= Pipeline(
        steps =[
            ("imputer", SimpleImputer(strategy ="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent" )),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor= ColumnTransformer(
        transformers = [
            ("num", numeric_pipe, WP_NUMERIC_FEATURES),
            ("cat", categorical_pipe, WP_CATEGORICAL_FEATURES),
        ]
    )
    if model_type == "logistic":
        clf = LogisticRegression(
            max_iter=int(wp_model_cfg["max_iter"]),
            class_weight=None,
            solver=str(wp_model_cfg["solver"]),
        )
    elif model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(wp_model_cfg["hidden_layer_sizes"]),
            activation=str(wp_model_cfg["activation"]),
            alpha=float(wp_model_cfg["alpha"]),
            learning_rate_init=float(wp_model_cfg["learning_rate_init"]),
            early_stopping=bool(wp_model_cfg["early_stopping"]),
            max_iter=int(wp_model_cfg["max_iter"]),
            n_iter_no_change=int(wp_model_cfg["n_iter_no_change"]),
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown wp model type: {model_type}")
    return Pipeline(steps=[("features", preprocessor),("model", clf)])


def evaluate_classifier(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    metrics = {
        "rows": int(len(y_true)),
        "positive_rate":float(np.mean(y_true)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier":float(brier_score_loss(y_true, proba)),
    }

    if len(np.unique(y_true)) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, proba))
    else:
        metrics["auc"] = float("nan")
    return metrics


def train_win_probability_model(
    rally_table: pd.DataFrame,
    max_train_rallies: int | None,
    model_type: str = "mlp",
    model_config: dict[str, object] | None = None,
) -> tuple[Pipeline, pd.DataFrame, dict[str, dict[str, float]]]:
    # this is the actual rally value model
    valid = rally_table.loc[
        rally_table["point_delta_total"].eq(1)
        & rally_table["home_match_win"].notna()
        & rally_table["home_sets_total"].notna()
    ].copy()
    valid["split"] = make_time_split(valid, model_config=model_config)
    X = make_wp_state_features(valid, "before")
    y = valid["home_match_win"].astype("int8")

    train_mask = valid["split"].eq("train")
    train_idx = valid.index[train_mask]
    if max_train_rallies and max_train_rallies > 0 and len(train_idx) > max_train_rallies:
        rng = np.random.default_rng(RANDOM_SEED)
        sampled = rng.choice(train_idx.to_numpy(), size=max_train_rallies, replace=False)
        train_mask = valid.index.isin(sampled)

    model = build_wp_model(model_type=model_type, model_config=model_config)
    model.fit(X.loc[train_mask], y.loc[train_mask])

    metrics: dict[str, dict[str, float]] = {}
    for split in ["train", "valid", "test"]:
        mask = valid["split"].eq(split)
        if not mask.any():
            continue
        proba = model.predict_proba(X.loc[mask])[:, 1]
        metrics[split] = evaluate_classifier(y.loc[mask], proba)
        metrics[split]["model_type"] = model_type

    before_features = make_wp_state_features(rally_table, "before")
    after_features = make_wp_state_features(rally_table, "after")
    rally_values = rally_table[
        [
            "season",
            "date",
            "contestid",
            "set",
            "rally_id",
            "home_clean",
            "away_clean",
            "point_winner_side",
            "point_winner_team",
            "point_delta_total",
            "home_match_win",
        ]
    ].copy()
    rally_values["home_wp_before"] = model.predict_proba(before_features)[:, 1]
    rally_values["home_wp_after"] = model.predict_proba(after_features)[:, 1]
    rally_values["home_wpa"] = rally_values["home_wp_after"] - rally_values["home_wp_before"]
    return model, rally_values, metrics


def build_team_phase_context(
    event_table: pd.DataFrame,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    # team context only, not exact player matchups
    cfg = model_config or MODEL_CONFIG
    phase_cfg = cfg["team_phase_context"]
    if event_table["is_action_row"].dtype == bool:
        action_mask = event_table["is_action_row"]
    else:
        action_mask = event_table["is_action_row"].astype("string").str.lower().isin(["true", "1", "yes"])

    events = event_table.loc[
        action_mask
        & event_table["actor_side"].isin(["home", "away"])
        & event_table["actor_team"].notna()
    ].copy()
    events["team_clean"] = events["actor_team"]
    events["opponent_team"] = np.where(
        events["actor_side"].eq("home"),
        events["away_clean"],
        events["home_clean"],
    )

    attack_families = ["attack", "kill", "attack_error"]
    attack_events = events.loc[events["event_family"].isin(attack_families)].copy()
    attack_counts = (
        attack_events.pivot_table(
            index="team_clean",
            columns="event_family",
            values="contestid",
            aggfunc="size",
            fill_value=0,
            observed=True,
        )
        .reset_index()
    )
    for col in attack_families:
        if col not in attack_counts.columns:
            attack_counts[col] = 0
    attack_counts = attack_counts.rename(
        columns={
            "attack": "team_attack_events",
            "kill": "team_kills",
            "attack_error": "team_attack_errors",
        }
    )

    attacks_faced = (
        attack_events.groupby("opponent_team", observed=True)
        .size()
        .rename("opponent_attack_events_faced")
        .reset_index()
        .rename(columns={"opponent_team": "team_clean"})
    )
    block_counts = (
        events.loc[events["event_family"].eq("block")]
        .groupby("team_clean", observed=True)
        .size()
        .rename("team_blocks")
        .reset_index()
    )
    teams = pd.DataFrame(
        {
            "team_clean": sorted(
                set(events["team_clean"].dropna().unique()).union(set(events["opponent_team"].dropna().unique()))
            )
        }
    )
    context = (
        teams.merge(attack_counts, on="team_clean", how="left")
        .merge(attacks_faced, on="team_clean", how="left")
        .merge(block_counts, on="team_clean", how="left")
    )
    for col in [
        "team_attack_events",
        "team_kills",
        "team_attack_errors",
        "opponent_attack_events_faced",
        "team_blocks",
    ]:
        context[col] = pd.to_numeric(context[col], errors="coerce").fillna(0)
    context["team_attack_attempts"] = (
        context["team_attack_events"] + context["team_kills"] + context["team_attack_errors"]
    )
    context["team_attack_efficiency"] = safe_rate(
        context["team_kills"] - context["team_attack_errors"],
        context["team_attack_attempts"],
    )
    context["team_kill_rate"] = safe_rate(context["team_kills"], context["team_attack_attempts"])
    context["team_attack_error_rate"] = safe_rate(context["team_attack_errors"], context["team_attack_attempts"])
    context["team_block_rate_faced"] = safe_rate(context["team_blocks"], context["opponent_attack_events_faced"])
    shrinkage_denominator = float(phase_cfg["shrinkage_denominator"])
    attack_shrink = context["team_attack_attempts"] / (context["team_attack_attempts"] + shrinkage_denominator)
    block_shrink = context["opponent_attack_events_faced"] / (context["opponent_attack_events_faced"] + shrinkage_denominator)
    context["team_attack_strength_index"] = attack_shrink * (
        float(phase_cfg["attack_efficiency_weight"]) * standard_z(context["team_attack_efficiency"])
        + float(phase_cfg["kill_rate_weight"]) * standard_z(context["team_kill_rate"])
        - float(phase_cfg["attack_error_weight"]) * standard_z(context["team_attack_error_rate"])
    )
    context["team_block_strength_index"] = block_shrink * standard_z(context["team_block_rate_faced"])
    return context


def build_event_credit(
    event_table: pd.DataFrame,
    rally_values: pd.DataFrame,
    team_context: pd.DataFrame | None = None,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    cfg = model_config or MODEL_CONFIG
    event_weights = cfg["event_weights"]
    context_cfg = cfg["context_multiplier"]
    if event_table["is_action_row"].dtype == bool:
        action_mask= event_table["is_action_row"]
    else:
        action_mask = event_table["is_action_row"].astype("string").str.lower().isin(["true", "1", "yes"])

    events = event_table.loc[
        action_mask
        & event_table["event_family"].isin(event_weights)
        & event_table["player_clean"].fillna("").ne("")
        & event_table["actor_side"].isin(["home", "away"])
    ].copy()
    # raw PBP team can point to the point winner on errors
    events["team_clean"] = events["actor_team"]

    events["opponent_team"] = np.where(
        events["actor_side"].eq("home"),
        events.get("away_clean", pd.Series(pd.NA, index=events.index)),
        events.get("home_clean", pd.Series(pd.NA, index=events.index)),
    )

    events = events.merge(
        rally_values[["contestid", "set", "rally_id", "home_wpa", "home_wp_before", "home_wp_after"]],
        on=["contestid", "set", "rally_id"],
        how="left",
    )
    events["event_weight"] = events["event_family"].map(event_weights).fillna(0.0).astype("float64")
    denom= events.groupby(["contestid", "set", "rally_id"], observed=True)["event_weight"].transform("sum")
    events["event_weight_share"] = events["event_weight"] / denom.replace(0, np.nan)
    events["event_weight_share"] = events["event_weight_share"].fillna(0.0)
    events["team_wpa"] = np.where(events["actor_side"].eq("home"), events["home_wpa"], -events["home_wpa"])
    events["base_event_credit"] = events["team_wpa"] * events["event_weight_share"]
    events["opp_block_strength_index"] = 0.0
    events["opp_attack_strength_index"] = 0.0
    if team_context is not None and not team_context.empty:
        context_small = team_context[
            ["team_clean", "team_block_strength_index", "team_attack_strength_index"]
        ].copy()
        events = events.merge(
            context_small.rename(
                columns={
                    "team_clean": "opponent_team",
                    "team_block_strength_index": "opp_block_strength_index_ctx",
                    "team_attack_strength_index": "opp_attack_strength_index_ctx",
                }
            ),
            on="opponent_team",
            how="left",
        )
        events["opp_block_strength_index"] = events["opp_block_strength_index_ctx"].fillna(0.0)
        events["opp_attack_strength_index"] = events["opp_attack_strength_index_ctx"].fillna(0.0)
        events = events.drop(columns=["opp_block_strength_index_ctx", "opp_attack_strength_index_ctx"])

    events["context_multiplier"] = 1.0
    # just a small matchup bump
    attack_plus = events["event_family"].isin(["kill", "attack"])
    attack_minus = events["event_family"].eq("attack_error")
    defense_plus = events["event_family"].isin(["block", "dig"])
    defense_minus = events["event_family"].isin(["block_error", "dig_error"])
    events.loc[attack_plus, "context_multiplier"] = (
        1.0 + float(context_cfg["attack_factor"]) * events.loc[attack_plus, "opp_block_strength_index"]
    )
    events.loc[attack_minus, "context_multiplier"] = (
        1.0 - float(context_cfg["attack_factor"]) * events.loc[attack_minus, "opp_block_strength_index"]
    )
    events.loc[defense_plus, "context_multiplier"] = (
        1.0 + float(context_cfg["defense_factor"]) * events.loc[defense_plus, "opp_attack_strength_index"]
    )
    events.loc[defense_minus, "context_multiplier"] = (
        1.0 - float(context_cfg["defense_factor"]) * events.loc[defense_minus, "opp_attack_strength_index"]
    )
    events["context_multiplier"] = events["context_multiplier"].clip(
        float(context_cfg["min_multiplier"]),
        float(context_cfg["max_multiplier"]),
    )
    events["event_credit"] = events["base_event_credit"] * events["context_multiplier"]
    events["positive_credit"] = events["event_credit"].clip(lower=0)
    events["negative_credit"] = events["event_credit"].clip(upper=0)
    return events


def aggregate_event_credit(event_credit: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["season", "team_clean", "player_clean"]

    event_credit = event_credit.copy()
    event_credit["rally_key"] = (
        event_credit["contestid"].astype("string")
        + "::"
        + event_credit["set"].astype("string")
        + "::"
        + event_credit["rally_id"].astype("string")
    )
    base = (
        event_credit.groupby(group_cols, observed=True)
        .agg(
            player=("player", "first"),
            team=("actor_team", "first"),
            player_uid=("player_uid", "first"),
            role_family=("role_family", "first"),
            event_touches=("event_family", "size"),
            event_rallies=("rally_key", "nunique"),
            raw_event_credit=("event_credit", "sum"),
            raw_base_event_credit=("base_event_credit", "sum"),
            positive_event_credit=("positive_credit", "sum"),
            negative_event_credit=("negative_credit", "sum"),
            mean_event_credit=("event_credit", "mean"),
            avg_context_multiplier=("context_multiplier", "mean"),
            avg_opp_block_strength_index=("opp_block_strength_index", "mean"),
            avg_opp_attack_strength_index=("opp_attack_strength_index", "mean"),
        )
        .reset_index()
    )
    counts = (
        event_credit.pivot_table(
            index=group_cols,
            columns="event_family",
            values="event_credit",
            aggfunc="size",
            fill_value=0,
            observed=True,
        )
        .reset_index()
    )
    counts.columns = [
        f"event_count_{c}" if c not in group_cols else c
        for c in counts.columns
    ]
    return base.merge(counts, on=group_cols, how="left")


def build_server_toughness(
    rally_table: pd.DataFrame,
    player_season_features: pd.DataFrame,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    # serve toughness = pressure, minus missed serves
    cfg = model_config or MODEL_CONFIG
    server_cfg = cfg["server_toughness"]
    serve = rally_table.loc[
        rally_table["serve_team"].notna()
        & rally_table["server_player_clean"].fillna("").ne("")
    ].copy()
    serve["serve_point_win"] = serve["point_winner_team"].fillna("").astype("string").eq(
        serve["serve_team"].fillna("").astype("string")
    )
    serve["serve_ace"] = serve["terminal_event_family"].eq("ace")
    serve["serve_error"] = serve["terminal_event_family"].eq("service_error")
    serve["forced_reception_error"] = serve["terminal_event_family"].eq("reception_error")

    server = (
        serve.groupby(["season", "serve_team", "server_player_clean"], observed=True)
        .agg(
            serve_rallies=("rally_id", "size"),
            serve_points_won=("serve_point_win", "sum"),
            serve_aces=("serve_ace", "sum"),
            serve_errors=("serve_error", "sum"),
            forced_reception_errors=("forced_reception_error", "sum"),
        )
        .reset_index()
        .rename(columns={"serve_team": "team_clean", "server_player_clean": "player_clean"})
    )
    server["serve_point_win_rate"] = safe_rate(server["serve_points_won"], server["serve_rallies"])
    server["serve_ace_rate"] = safe_rate(server["serve_aces"], server["serve_rallies"])
    server["serve_error_rate"] = safe_rate(server["serve_errors"], server["serve_rallies"])
    server["forced_reception_error_rate"] = safe_rate(
        server["forced_reception_errors"], server["serve_rallies"]
    )
    server["server_toughness_raw"] = (
        float(server_cfg["serve_point_win_weight"]) * standard_z(server["serve_point_win_rate"])
        + float(server_cfg["serve_ace_weight"]) * standard_z(server["serve_ace_rate"])
        + float(server_cfg["forced_reception_error_weight"]) * standard_z(server["forced_reception_error_rate"])
        - float(server_cfg["serve_error_weight"]) * standard_z(server["serve_error_rate"])
    )
    shrink = server["serve_rallies"] / (
        server["serve_rallies"] + float(server_cfg["shrinkage_denominator"])
    )
    server["server_toughness_index"] = server["server_toughness_raw"] * shrink

    display = player_season_features[
        ["season", "team_clean", "player_clean", "player", "team", "conference", "role_family", "player_uid"]
    ].drop_duplicates()
    return server.merge(display, on=["season", "team_clean", "player_clean"], how="left")


def expected_binary_probability(
    frame: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    max_train_rows: int=300_000,
    model_config: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, float | str | int]]:
    cfg = model_config or MODEL_CONFIG
    aux_cfg = cfg["aux_binary_model"]
    y = frame[target_col].astype("int8")
    if y.nunique(dropna=True) < 2 or len(frame) < 50:
        mean=float(y.mean()) if len(y) else 0.0
        return np.full(len(frame), mean, dtype="float64"), {
            "type": "mean",
            "rows": int(len(frame)),
            "positive_rate": mean,
        }

    X = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_idx = X.index

    if len(X) > max_train_rows:
        rng = np.random.default_rng(RANDOM_SEED)
        train_idx = rng.choice(X.index.to_numpy(), size=max_train_rows, replace=False)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=int(aux_cfg["max_iter"]),
                    solver=str(aux_cfg["solver"]),
                ),
            ),
        ]
    )
    model.fit(X.loc[train_idx], y.loc[train_idx])
    proba = model.predict_proba(X)[:, 1]
    info: dict[str, float | str | int] = {
        "type": "logistic",
        "rows": int(len(frame)),
        "train_rows": int(len(train_idx)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y, proba)),
    }
    if y.nunique(dropna=True) == 2:
        info["auc"] = float(roc_auc_score(y, proba))
    return proba, info


def build_serve_receive_value(
    event_table: pd.DataFrame,
    rally_table: pd.DataFrame,
    rally_pass: pd.DataFrame,
    server_toughness: pd.DataFrame,
    player_season_features: pd.DataFrame,
    model_config: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # receive value needs server difficulty or it is kind of unfair
    cfg = model_config or MODEL_CONFIG
    sr_cfg = cfg["serve_receive_value"]
    if event_table["is_action_row"].dtype == bool:
        action_mask=event_table["is_action_row"]
    else:
        action_mask = event_table["is_action_row"].astype("string").str.lower().isin(["true", "1", "yes"])

    group_cols = ["contestid", "set", "rally_id"]
    receive_events = (
        event_table.loc[
            action_mask
            & event_table["event_family"].isin(["reception", "reception_error"])
            & event_table["player_clean"].fillna("").ne("")
            & event_table["actor_team"].notna()
            & event_table["actor_side"].isin(["home", "away"])
        ]
        .sort_values(group_cols + ["file_row"], kind="mergesort")
        .groupby(group_cols, sort=False, observed=True)
        .first()
        .reset_index()[
            group_cols + ["season", "file_row", "actor_team", "actor_side", "player", "player_clean", "event_family"]
        ]
        .rename(
            columns={
                "actor_team": "team_clean",
                "actor_side": "receiving_side",
                "event_family": "receive_event_family",
                "file_row": "receive_row",
            }
        )
    )
    rally_context_cols = [
        "season",
        *group_cols,
        "score_home_start",
        "score_away_start",
        "serve_team",
        "server_player_clean",
        "point_winner_team",
    ]
    rally_context = rally_table[[c for c in rally_context_cols if c in rally_table.columns]].copy()
    pass_cols = [
        "contestid",
        "set",
        "rally_id",
        "first_ball_attack",
        "first_ball_kill",
        "first_ball_middle_attack",
    ]
    pass_context = rally_pass[[c for c in pass_cols if c in rally_pass.columns]].copy()
    receive = (
        receive_events.merge(rally_context, on=["season", *group_cols], how="left")
        .merge(pass_context, on=group_cols, how="left")
    )

    server_small = server_toughness[
        [
            "season",
            "team_clean",
            "player_clean",
            "serve_rallies",
            "serve_point_win_rate",
            "serve_ace_rate",
            "serve_error_rate",
            "server_toughness_index",
        ]
    ].rename(
        columns={
            "team_clean": "serve_team",
            "player_clean": "server_player_clean",
            "serve_rallies": "server_serve_rallies",
        }
    )
    receive = receive.merge(server_small, on=["season", "serve_team", "server_player_clean"], how="left")
    for col in [
        "server_serve_rallies",
        "serve_point_win_rate",
        "serve_ace_rate",
        "serve_error_rate",
        "server_toughness_index",
    ]:
        receive[col] = pd.to_numeric(receive[col], errors="coerce").fillna(0.0)

    receive["reception_error"] = receive["receive_event_family"].eq("reception_error")
    receive["first_ball_attack"] = receive["first_ball_attack"].fillna(False).astype(bool) & ~receive[
        "reception_error"
    ]
    receive["first_ball_kill"] = receive["first_ball_kill"].fillna(False).astype(bool) & ~receive[
        "reception_error"
    ]
    receive["first_ball_middle_attack"] = receive["first_ball_middle_attack"].fillna(False).astype(bool) & ~receive[
        "reception_error"
    ]
    receive["receive_point_win"] = receive["point_winner_team"].fillna("").astype("string").eq(
        receive["team_clean"].fillna("").astype("string")
    )
    receive["receive_score_diff_start"] = np.where(
        receive["receiving_side"].eq("home"),
        pd.to_numeric(receive["score_home_start"], errors="coerce")
        - pd.to_numeric(receive["score_away_start"], errors="coerce"),
        pd.to_numeric(receive["score_away_start"], errors="coerce")
        - pd.to_numeric(receive["score_home_start"], errors="coerce"),
    )
    receive["total_points_start"] = (
        pd.to_numeric(receive["score_home_start"], errors="coerce").fillna(0)
        + pd.to_numeric(receive["score_away_start"], errors="coerce").fillna(0)
    )
    receive["is_late_set"] = (
        (pd.to_numeric(receive["set"], errors="coerce").lt(5) & receive["total_points_start"].ge(38))
        | (pd.to_numeric(receive["set"], errors="coerce").eq(5) & receive["total_points_start"].ge(20))
    ).astype("int8")

    feature_cols = [
        "server_toughness_index",
        "serve_point_win_rate",
        "serve_ace_rate",
        "serve_error_rate",
        "server_serve_rallies",
        "receive_score_diff_start",
        "total_points_start",
        "is_late_set",
    ]
    model_info: dict[str, object] = {}
    targets = {
        "reception_error": "expected_reception_error",
        "first_ball_attack": "expected_first_ball_attack",
        "first_ball_kill": "expected_first_ball_kill",
        "first_ball_middle_attack": "expected_first_ball_middle_attack",
        "receive_point_win": "expected_receive_point_win",
    }
    # compare actual pass outcomes to what was expected
    for target, expected_col in targets.items():
        receive[expected_col], model_info[target] = expected_binary_probability(
            receive,
            target,
            feature_cols,
            model_config=model_config,
        )

    receive["receive_attempt_value"] = (
        float(sr_cfg["receive_point_win_weight"])
        * (receive["receive_point_win"].astype(float) - receive["expected_receive_point_win"])
        + float(sr_cfg["first_ball_kill_weight"])
        * (receive["first_ball_kill"].astype(float) - receive["expected_first_ball_kill"])
        + float(sr_cfg["first_ball_middle_attack_weight"])
        * (
            receive["first_ball_middle_attack"].astype(float)
            - receive["expected_first_ball_middle_attack"]
        )
        + float(sr_cfg["first_ball_attack_weight"])
        * (receive["first_ball_attack"].astype(float) - receive["expected_first_ball_attack"])
        + float(sr_cfg["reception_error_weight"])
        * (receive["expected_reception_error"] - receive["reception_error"].astype(float))
    )

    sr = (
        receive.groupby(["season", "team_clean", "player_clean"], observed=True)
        .agg(
            player=("player", "first"),
            sr_attempts=("rally_id", "size"),
            sr_clean_receptions=("reception_error", lambda x: (~x).sum()),
            sr_reception_errors=("reception_error", "sum"),
            sr_first_ball_attacks=("first_ball_attack", "sum"),
            sr_first_ball_kills=("first_ball_kill", "sum"),
            sr_first_ball_middle_attacks=("first_ball_middle_attack", "sum"),
            sr_receive_points_won=("receive_point_win", "sum"),
            sr_expected_reception_errors=("expected_reception_error", "sum"),
            sr_expected_first_ball_attacks=("expected_first_ball_attack", "sum"),
            sr_expected_first_ball_kills=("expected_first_ball_kill", "sum"),
            sr_expected_first_ball_middle_attacks=("expected_first_ball_middle_attack", "sum"),
            sr_expected_receive_points_won=("expected_receive_point_win", "sum"),
            sr_avg_server_toughness=("server_toughness_index", "mean"),
            sr_avg_server_ace_rate=("serve_ace_rate", "mean"),
            sr_value_total=("receive_attempt_value", "sum"),
        )
        .reset_index()
    )
    sr["sr_reception_error_rate"] = safe_rate(sr["sr_reception_errors"], sr["sr_attempts"])
    sr["sr_first_ball_attack_rate"] = safe_rate(sr["sr_first_ball_attacks"], sr["sr_attempts"])
    sr["sr_first_ball_kill_rate"] = safe_rate(sr["sr_first_ball_kills"], sr["sr_attempts"])
    sr["sr_first_ball_middle_attack_rate"] = safe_rate(
        sr["sr_first_ball_middle_attacks"], sr["sr_attempts"]
    )
    sr["sr_receive_point_win_rate"] = safe_rate(sr["sr_receive_points_won"], sr["sr_attempts"])
    sr["sr_error_avoided_oe"] = sr["sr_expected_reception_errors"] - sr["sr_reception_errors"]
    sr["sr_first_ball_attack_oe"] = sr["sr_first_ball_attacks"] - sr["sr_expected_first_ball_attacks"]
    sr["sr_first_ball_kill_oe"] = sr["sr_first_ball_kills"] - sr["sr_expected_first_ball_kills"]
    sr["sr_first_ball_middle_attack_oe"] = (
        sr["sr_first_ball_middle_attacks"] - sr["sr_expected_first_ball_middle_attacks"]
    )
    sr["sr_receive_point_win_oe"] = sr["sr_receive_points_won"] - sr["sr_expected_receive_points_won"]
    sr["sr_value_per_attempt"] = safe_rate(sr["sr_value_total"], sr["sr_attempts"])
    shrink = sr["sr_attempts"] / (
        sr["sr_attempts"] + float(sr_cfg["shrinkage_denominator"])
    )
    sr["sr_shrunk_value_per_attempt"] = sr["sr_value_per_attempt"] * shrink
    sr["sr_shrunk_value"] = sr["sr_shrunk_value_per_attempt"] * sr["sr_attempts"]

    display = player_season_features[
        ["season", "team_clean", "player_clean", "team", "conference", "role_family", "player_uid"]
    ].drop_duplicates()
    sr = sr.merge(display, on=["season", "team_clean", "player_clean"], how="left")
    return sr, model_info


def add_shrunk_impact(player_values: pd.DataFrame, tau: float) -> pd.DataFrame:
    # pull small samples back toward the role average
    player_values = player_values.copy()
    player_values["event_rallies"] = pd.to_numeric(player_values["event_rallies"], errors="coerce").fillna(0)

    player_values["raw_event_credit"] = pd.to_numeric(
        player_values["raw_event_credit"], errors="coerce"
    ).fillna(0.0)
    player_values["raw_credit_per_rally"] = (
        player_values["raw_event_credit"] / player_values["event_rallies"].replace(0, np.nan)
    ).fillna(0.0)
    role_mean = player_values.groupby("role_family", observed=True)["raw_credit_per_rally"].transform("mean")
    n = player_values["event_rallies"]
    player_values["shrunk_credit_per_rally"] = (
        (n / (n + tau)) * player_values["raw_credit_per_rally"]
        + (tau / (n + tau)) * role_mean.fillna(0.0)
    )
    player_values["shrunk_event_value"] = player_values["shrunk_credit_per_rally"] * n
    return player_values


def prepare_tabular_features(player_values: pd.DataFrame) -> list[str]:
    excluded = {
        "season",
        "team",
        "teamid",
        "team_clean",
        "conference",
        "number",
        "number_clean",
        "player",
        "player_clean",
        "pos",
        "role_family",
        "roster_role_family",
        "player_uid",
        "raw_event_credit",
        "raw_base_event_credit",
        "positive_event_credit",
        "negative_event_credit",
        "mean_event_credit",
        "raw_credit_per_rally",
        "shrunk_credit_per_rally",
        "shrunk_event_value",
        "tabular_pred_value",
        "event_value_z",
        "tabular_value_z",
        "final_value_z",
        "role_score",
        "overall_score",
    }
    numeric_cols = [
        c
        for c in player_values.columns
        if c not in excluded
        and not c.startswith("sr_")
        and c != "role_inferred_from_profile"
        and pd.api.types.is_numeric_dtype(player_values[c])
    ]
    return numeric_cols


def add_tabular_value_model(
    player_values: pd.DataFrame,
    model_config: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # smoother backup model from the season stat table
    cfg = model_config or MODEL_CONFIG
    tabular_cfg = cfg["tabular_value_model"]
    player_values = player_values.copy()
    feature_cols= prepare_tabular_features(player_values)
    player_values["tabular_pred_value"] = 0.0
    model_info: dict[str, object] = {"feature_columns": feature_cols, "role_models": {}}

    for role, role_df in player_values.groupby("role_family", observed=True):
        role_idx = role_df.index
        train_mask = role_df["event_rallies"].fillna(0).ge(int(tabular_cfg["min_train_rallies"]))
        if train_mask.sum() < 10 or not feature_cols:
            player_values.loc[role_idx, "tabular_pred_value"] = role_df["shrunk_event_value"].mean()
            model_info["role_models"][role] = {"type": "role_mean", "train_rows": int(train_mask.sum())}
            continue

        X_train = role_df.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_train = role_df.loc[train_mask, "shrunk_event_value"]
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas =np.logspace(-3, 3, 13))),
            ]
        )
        model.fit(X_train, y_train)
        X_all = role_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        player_values.loc[role_idx, "tabular_pred_value"] = model.predict(X_all)
        ridge = model.named_steps["ridge"]
        model_info["role_models"][role] = {
            "type": "ridge_cv",
            "train_rows": int(train_mask.sum()),
            "alpha": float(ridge.alpha_),
        }

    return player_values, model_info


def build_rankings(
    player_season_features: pd.DataFrame,
    event_credit_summary: pd.DataFrame,
    serve_receive_value: pd.DataFrame | None,
    tau: float,
    min_rallies: int,
    model_config: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    # mix all the pieces into final ranks
    cfg = model_config or MODEL_CONFIG
    group_cols = ["season", "team_clean", "player_clean"]

    player_values = player_season_features.merge(
        event_credit_summary.drop(columns=["player", "team", "role_family", "player_uid"], errors="ignore"),
        on=group_cols,
        how="left",
    )
    if serve_receive_value is not None and not serve_receive_value.empty:
        sr_merge = serve_receive_value.drop(
            columns=["player", "team", "conference", "role_family", "player_uid"],
            errors="ignore",
        )
        player_values = player_values.merge(sr_merge, on=group_cols, how="left")

    credit_cols = [
        "event_touches",
        "event_rallies",
        "raw_event_credit",
        "raw_base_event_credit",
        "positive_event_credit",
        "negative_event_credit",
        "mean_event_credit",
        "avg_context_multiplier",
        "avg_opp_block_strength_index",
        "avg_opp_attack_strength_index",
    ]
    for col in credit_cols:
        if col not in player_values.columns:
            player_values[col] = 0
        player_values[col] = pd.to_numeric(player_values[col], errors="coerce").fillna(0)
    for col in [c for c in player_values.columns if c.startswith("event_count_")]:
        player_values[col] = pd.to_numeric(player_values[col], errors="coerce").fillna(0).astype("int32")

    for col in [c for c in player_values.columns if c.startswith("sr_")]:
        player_values[col] = pd.to_numeric(player_values[col], errors="coerce").fillna(0)

    player_values["roster_role_family"] = player_values["role_family"].fillna("unknown")
    player_values["role_family"] = infer_role_family(
        player_values,
        "roster_role_family",
        model_config=cfg,
    )
    player_values["role_inferred_from_profile"] = player_values["role_family"].ne(player_values["roster_role_family"])

    player_values = add_basic_rate_features(player_values)
    player_values = add_shrunk_impact(player_values, tau=tau)
    player_values, tabular_info = add_tabular_value_model(player_values, model_config=cfg)

    player_values = role_zscore(player_values, "shrunk_event_value", "event_value_z")
    player_values = role_zscore(player_values, "tabular_pred_value", "tabular_value_z")
    if "sr_shrunk_value" not in player_values.columns:
        player_values["sr_shrunk_value"] = 0.0
    player_values = role_zscore(player_values, "sr_shrunk_value", "serve_receive_value_z")

    role_weights = pd.DataFrame(
        [
            (
                role_family,
                float(weights["event"]),
                float(weights["tabular"]),
                float(weights["serve_receive"]),
            )
            for role_family, weights in cfg["role_blend_weights"].items()
        ],
        columns=[
            "role_family",
            "event_blend_weight",
            "tabular_blend_weight",
            "serve_receive_blend_weight",
        ],
    )
    player_values = player_values.merge(role_weights, on="role_family", how="left")
    role_blend_default = cfg["role_blend_default"]
    for col, default in [
        ("event_blend_weight", float(role_blend_default["event"])),
        ("tabular_blend_weight", float(role_blend_default["tabular"])),
        ("serve_receive_blend_weight", float(role_blend_default["serve_receive"])),
    ]:
        player_values[col] = player_values[col].fillna(default)
    weight_sum = (
        player_values["event_blend_weight"]
        + player_values["tabular_blend_weight"]
        + player_values["serve_receive_blend_weight"]
    ).replace(0, 1)
    player_values["event_blend_weight"] = player_values["event_blend_weight"] / weight_sum
    player_values["tabular_blend_weight"] = player_values["tabular_blend_weight"] / weight_sum
    player_values["serve_receive_blend_weight"] = player_values["serve_receive_blend_weight"] / weight_sum
    player_values["blended_value_z"] = (
        player_values["event_blend_weight"] * player_values["event_value_z"]
        + player_values["tabular_blend_weight"] * player_values["tabular_value_z"]
        + player_values["serve_receive_blend_weight"] * player_values["serve_receive_value_z"]
    )
    team_strength_cfg = cfg.get("team_strength_prior", {})
    team_strength_weight = float(team_strength_cfg.get("weight", 0.0))
    player_values["team_strength_value_z"] = standard_z(player_values["team_strength_index"])
    player_values["final_value_z"] = (
        player_values["blended_value_z"]
        + team_strength_weight * player_values["team_strength_value_z"]
    )
    tanh_scale = float(cfg["score_scaling"]["tanh_scale"])
    cdf_scale = float(cfg["score_scaling"]["cdf_scale"])
    player_values["score_z"] = tanh_scale * np.tanh(player_values["final_value_z"] / tanh_scale)
    player_values["role_score"] = cdf_scale * normal_cdf(player_values["score_z"])
    player_values["overall_score"] = player_values["role_score"]
    player_values["meets_volume_threshold"] = player_values["event_rallies"].ge(min_rallies)
    player_values["role_rank"] = (
        player_values.groupby("role_family", observed=True)["final_value_z"]
        .rank(method="dense", ascending=False)
        .astype("int32")
    )
    player_values["overall_rank"] = player_values["final_value_z"].rank(method="dense", ascending=False).astype("int32")

    rank_cols = [
        "season",
        "role_family",
        "roster_role_family",
        "role_inferred_from_profile",
        "role_rank",
        "overall_rank",
        "role_score",
        "overall_score",
        "final_value_z",
        "score_z",
        "event_value_z",
        "tabular_value_z",
        "serve_receive_value_z",
        "blended_value_z",
        "team_strength_value_z",
        "event_blend_weight",
        "tabular_blend_weight",
        "serve_receive_blend_weight",
        "shrunk_event_value",
        "shrunk_credit_per_rally",
        "raw_event_credit",
        "raw_base_event_credit",
        "raw_credit_per_rally",
        "event_rallies",
        "event_touches",
        "meets_volume_threshold",
        "player",
        "team",
        "conference",
        "pos",
        "player_clean",
        "team_clean",
        "player_uid",
        "positive_event_credit",
        "negative_event_credit",
        "tabular_pred_value",
    ]
    extra_cols = [
        c
        for c in player_values.columns
        if c.startswith("box_")
        or c.startswith("fb_")
        or c.startswith("sr_")
        or c.startswith("event_count_")
        or c
        in {
            "team_strength_index",
            "opp_strength_index",
            "matches_played_rows",
            "calc_hit_pct",
            "kill_rate_per_attempt",
            "attack_error_rate",
            "serve_error_rate",
            "box_reception_error_rate",
            "touches_per_rally",
            "avg_context_multiplier",
            "avg_opp_block_strength_index",
            "avg_opp_attack_strength_index",
        }
    ]
    rankings = player_values[rank_cols + [c for c in extra_cols if c not in rank_cols]].copy()
    rankings = rankings.sort_values(
        ["role_family", "role_rank", "overall_rank", "player_clean"], kind="mergesort"
    ).reset_index(drop=True)
    return rankings, tabular_info


def build_role_audit(
    rankings: pd.DataFrame,
    model_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build a small audit table for weird looking rankings."""
    cfg = model_config or MODEL_CONFIG
    audit_cfg = cfg["audit_thresholds"]
    audit = rankings.copy()
    for col in [
        "box_kills",
        "box_errors",
        "box_total_attacks",
        "box_assists",
        "box_digs",
        "box_rerr",
        "box_serr",
        "event_rallies",
        "event_touches",
        "event_count_attack",
        "event_count_kill",
        "event_count_reception",
        "event_count_set",
    ]:
        if col not in audit.columns:
            audit[col] = 0
        audit[col] = pd.to_numeric(audit[col], errors="coerce").fillna(0)

    audit = add_basic_rate_features(audit)
    rallies = audit["event_rallies"].replace(0, np.nan)
    touches = audit["event_touches"].replace(0, np.nan)
    audit["attacks_per_rally"] = (audit["box_total_attacks"] / rallies).fillna(0.0)
    audit["touches_per_rally"] = (audit["event_touches"] / rallies).fillna(0.0)
    audit["reception_share_of_touches"] = (audit["event_count_reception"] / touches).fillna(0.0)
    audit["set_share_of_touches"] = (audit["event_count_set"] / touches).fillna(0.0)
    audit["attack_share_of_touches"] = (
        (audit["event_count_attack"] + audit["event_count_kill"]) / touches
    ).fillna(0.0)

    if "roster_role_family" not in audit.columns:
        audit["roster_role_family"] = audit["role_family"]
    audit["inferred_role_family"] = infer_role_family(
        audit.assign(role_family=audit["roster_role_family"]),
        "role_family",
        model_config=cfg,
    )

    audit["role_mismatch_flag"] = audit["inferred_role_family"].ne(audit["roster_role_family"].fillna("unknown"))
    audit["hitter_in_libero_bucket_flag"] = audit["roster_role_family"].eq("libero_ds") & (
        audit["box_total_attacks"].ge(int(audit_cfg["libero_attack_volume"]))
        | audit["event_count_kill"].ge(int(audit_cfg["libero_event_kills"]))
    )
    audit["low_efficiency_high_rank_flag"] = (
        audit["box_total_attacks"].ge(int(audit_cfg["low_efficiency_attack_volume"]))
        & audit["calc_hit_pct"].lt(float(audit_cfg["low_efficiency_hit_pct"]))
        & audit["overall_rank"].le(int(audit_cfg["high_rank_cutoff"]))
    )
    audit["low_volume_high_rank_flag"] = audit["overall_rank"].le(
        int(audit_cfg["high_rank_cutoff"])
    ) & audit["event_rallies"].lt(int(audit_cfg["low_volume_rallies"]))

    reasons = []
    for row in audit.itertuples(index=False):
        row_reasons = []
        if row.hitter_in_libero_bucket_flag:
            row_reasons.append("libero_ds label with hitter-level attack volume")
        if row.role_mismatch_flag:
            row_reasons.append(
                f"inferred role {row.inferred_role_family} differs from roster role {row.roster_role_family}"
            )
        if row.low_efficiency_high_rank_flag:
            row_reasons.append("top-100 rank despite sub-.200 calculated hitting percentage")
        if row.low_volume_high_rank_flag:
            row_reasons.append(
                f"top-{int(audit_cfg['high_rank_cutoff'])} rank on fewer than "
                f"{int(audit_cfg['low_volume_rallies'])} event rallies"
            )
        reasons.append("; ".join(row_reasons))
    audit["audit_reason"] = reasons

    audit_cols = [
        "season",
        "overall_rank",
        "role_rank",
        "player",
        "team",
        "conference",
        "pos",
        "role_family",
        "roster_role_family",
        "inferred_role_family",
        "role_inferred_from_profile",
        "role_mismatch_flag",
        "hitter_in_libero_bucket_flag",
        "low_efficiency_high_rank_flag",
        "low_volume_high_rank_flag",
        "audit_reason",
        "overall_score",
        "role_score",
        "event_rallies",
        "event_touches",
        "touches_per_rally",
        "box_kills",
        "box_errors",
        "box_total_attacks",
        "calc_hit_pct",
        "kill_rate_per_attempt",
        "attack_error_rate",
        "attacks_per_rally",
        "box_assists",
        "box_digs",
        "box_rerr",
        "box_serr",
        "event_count_kill",
        "event_count_attack",
        "event_count_attack_error",
        "event_count_reception",
        "event_count_reception_error",
        "event_count_set",
        "event_count_service_error",
        "sr_attempts",
        "sr_avg_server_toughness",
        "sr_reception_error_rate",
        "sr_first_ball_kill_rate",
        "sr_first_ball_middle_attack_rate",
        "sr_receive_point_win_rate",
        "sr_error_avoided_oe",
        "sr_first_ball_kill_oe",
        "sr_first_ball_middle_attack_oe",
        "sr_receive_point_win_oe",
        "sr_shrunk_value",
        "serve_receive_value_z",
        "avg_context_multiplier",
        "avg_opp_block_strength_index",
        "avg_opp_attack_strength_index",
        "raw_event_credit",
        "raw_base_event_credit",
        "shrunk_event_value",
        "positive_event_credit",
        "negative_event_credit",
    ]
    audit_cols = [c for c in audit_cols if c in audit.columns]
    return audit[audit_cols].sort_values(["overall_rank", "role_rank"], kind="mergesort").reset_index(drop=True)


def combine_model_outputs(out_dir: Path, seasons: list[str]) -> None:
    if len(seasons) < 2:
        return

    csv_stems = [
        "player_rankings",
        "top25_rankings_by_role",
        "ranking_role_audit",
        "serve_receive_player_value",
        "server_toughness",
        "team_phase_context",
    ]
    for stem in csv_stems:
        frames = []
        for season in seasons:
            path = out_dir / f"{stem}_{season}.csv"
            if path.exists():
                frames.append(pd.read_csv(path, low_memory=False))
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        save_csv(combined, out_dir, f"{stem}_all_seasons")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train volleyball value models and write ranking outputs."
    )
    add_season_args(parser)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument(
        "--max-train-rallies",
        type=int,
        default=500_000,
        help="Maximum train rallies for the win-probability fit. Use 0 for all.",
    )
    parser.add_argument("--shrinkage-tau", type=float, default=250.0)
    parser.add_argument("--min-rallies", type=int, default=200)
    parser.add_argument(
        "--wp-model",
        choices=["logistic", "mlp"],
        default="mlp",
        help="Model family for rally win probability. Defaults to mlp.",
    )
    parser.add_argument(
        "--build-missing-preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build missing preprocessed season files automatically before ranking.",
    )
    parser.add_argument(
        "--combine-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write combined all-season ranking outputs when multiple seasons are requested.",
    )
    return parser.parse_args()


def run_for_season(
    season: str,
    *,
    repo_root: Path | None = None,
    max_train_rallies: int | None = 500_000,
    shrinkage_tau: float = 250.0,
    min_rallies: int = 200,
    wp_model_type: str = "mlp",
    build_missing_preprocessed: bool = True,
    download_missing: bool = True,
    model_config: dict[str, object] | None = None,
) -> None:
    """Run the ranking stage for a single season using existing preprocessed files."""
    cfg = model_config or MODEL_CONFIG
    package_root = repo_root or Path(Git(root_dir=".").root_dir)
    data_dir = package_root / "Data" / "preprocessed"
    out_dir = package_root / "Data" / "model_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_root: {package_root}")
    print(f"season   : {season}")
    print(f"data_dir : {data_dir}")
    print(f"out_dir  : {out_dir}")

    rally_path = data_dir / f"rally_table_{season}.csv"
    event_path = data_dir / f"event_table_{season}.csv"
    rally_pass_path = data_dir / f"rally_pass_{season}.csv"
    player_features_path = data_dir / f"player_season_features_{season}.csv"

    missing_preprocessed = [
        path
        for path in [rally_path, event_path, rally_pass_path, player_features_path]
        if not path.exists()
    ]
    if missing_preprocessed and build_missing_preprocessed:
        print("Missing preprocessed files detected. Building the data pipeline first.")
        build_preprocessed_season(
            season,
            repo_root=package_root,
            save_event_table=True,
            download_missing=download_missing,
        )
        missing_preprocessed = [
            path
            for path in [rally_path, event_path, rally_pass_path, player_features_path]
            if not path.exists()
        ]
    if missing_preprocessed:
        missing_text = "\n".join(str(path) for path in missing_preprocessed)
        raise FileNotFoundError(f"Missing required preprocessed file(s):\n{missing_text}")

    rally_table = pd.read_csv(rally_path, low_memory=False)
    player_season_features = pd.read_csv(player_features_path, low_memory=False)
    _, rally_values, _ = train_win_probability_model(
        rally_table,
        max_train_rallies=max_train_rallies,
        model_type=wp_model_type,
        model_config=cfg,
    )
    save_csv(rally_values, out_dir, f"rally_win_probability_{season}")

    event_usecols = [
        "season",
        "file_row",
        "contestid",
        "set",
        "rally_id",
        "home_clean",
        "away_clean",
        "actor_team",
        "actor_side",
        "event_family",
        "is_action_row",
        "player",
        "player_clean",
        "player_uid",
        "role_family",
    ]
    event_table = pd.read_csv(event_path, usecols=event_usecols, low_memory=False)

    team_phase_context = build_team_phase_context(event_table, model_config=cfg)
    save_csv(team_phase_context, out_dir, f"team_phase_context_{season}")

    event_credit = build_event_credit(
        event_table,
        rally_values,
        team_phase_context,
        model_config=cfg,
    )
    event_credit_summary = aggregate_event_credit(event_credit)
    save_csv(event_credit_summary, out_dir, f"player_event_credit_{season}")
    del event_credit

    server_toughness = build_server_toughness(
        rally_table,
        player_season_features,
        model_config=cfg,
    )
    save_csv(server_toughness, out_dir, f"server_toughness_{season}")

    rally_pass = pd.read_csv(rally_pass_path, low_memory=False)
    serve_receive_value, _ = build_serve_receive_value(
        event_table,
        rally_table,
        rally_pass,
        server_toughness,
        player_season_features,
        model_config=cfg,
    )
    save_csv(serve_receive_value, out_dir, f"serve_receive_player_value_{season}")
    del event_table, rally_pass

    rankings, _ = build_rankings(
        player_season_features,
        event_credit_summary,
        serve_receive_value,
        tau=shrinkage_tau,
        min_rallies=min_rallies,
        model_config=cfg,
    )
    save_csv(rankings, out_dir, f"player_rankings_{season}")
    role_audit = build_role_audit(rankings, model_config=cfg)
    save_csv(role_audit, out_dir, f"ranking_role_audit_{season}")
    top_rankings = (
        rankings.loc[rankings["meets_volume_threshold"]]
        .sort_values(["role_family", "role_rank"], kind="mergesort")
        .groupby("role_family", observed=True)
        .head(25)
        .reset_index(drop=True)
    )
    save_csv(top_rankings, out_dir, f"top25_rankings_by_role_{season}")

    print("\nTop ranked players by role:")
    print(
        top_rankings[
            ["role_family", "role_rank", "role_score", "player", "team", "event_rallies"]
        ]
        .head(50)
        .to_string(index=False)
    )


def main() -> None:
    args = parse_args()

    package_root = Path(Git(root_dir=".").root_dir)
    source_data_dir = package_root / "Data" / "original"
    seasons = resolve_requested_seasons(
        season=args.season,
        seasons=args.seasons,
        season_range=args.season_range,
        all_seasons=args.all_seasons,
        data_dir=source_data_dir,
        download_missing=args.download_missing,
    )
    max_train_rallies = None if args.max_train_rallies == 0 else args.max_train_rallies

    for season in seasons:
        run_for_season(
            season,
            repo_root=package_root,
            max_train_rallies=max_train_rallies,
            shrinkage_tau=args.shrinkage_tau,
            min_rallies=args.min_rallies,
            wp_model_type=args.wp_model,
            build_missing_preprocessed=args.build_missing_preprocessed,
            download_missing=args.download_missing,
            model_config=MODEL_CONFIG,
        )

    if args.combine_outputs:
        combine_model_outputs(package_root / "Data" / "model_outputs", seasons)


if __name__ == "__main__":
    main()
