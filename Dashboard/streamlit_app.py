from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "Data" / "model_outputs"

CORE_RANKING_COLUMNS = [
    "overall_rank",
    "role_rank",
    "player",
    "team",
    "conference",
    "role_family",
    "overall_score",
    "event_rallies",
    "calc_hit_pct",
    "sr_shrunk_value",
    "meets_volume_threshold",
]

AUDIT_COLUMNS = [
    "overall_rank",
    "role_rank",
    "player",
    "team",
    "role_family",
    "roster_role_family",
    "inferred_role_family",
    "overall_score",
    "event_rallies",
    "audit_reason",
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-1: #f6f2e9;
            --bg-2: #e7ece5;
            --panel: rgba(255, 252, 247, 0.82);
            --panel-strong: #fff9f0;
            --text: #17261e;
            --muted: #53645a;
            --line: rgba(23, 38, 30, 0.10);
            --accent: #0f766e;
            --accent-2: #b45309;
            --shadow: 0 20px 50px rgba(23, 38, 30, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 30%),
                radial-gradient(circle at top right, rgba(180, 83, 9, 0.16), transparent 25%),
                linear-gradient(180deg, var(--bg-1), var(--bg-2));
            color: var(--text);
        }

        .main .block-container {
            max-width: 1200px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        h1, h2, h3, h4 {
            font-family: "Space Grotesk", sans-serif;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        p, li, label, [data-testid="stMetricLabel"], [data-testid="stMarkdownContainer"] {
            font-family: "Space Grotesk", sans-serif;
        }

        code, pre, .stCodeBlock {
            font-family: "IBM Plex Mono", monospace;
        }

        [data-testid="stSidebar"] {
            background: rgba(248, 244, 236, 0.92);
            border-right: 1px solid var(--line);
        }

        [data-testid="stDataFrame"], [data-testid="stTable"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 18px;
        }

        .hero {
            padding: 1.6rem 1.6rem 1.4rem 1.6rem;
            border-radius: 24px;
            background:
                linear-gradient(130deg, rgba(255, 249, 240, 0.92), rgba(238, 248, 244, 0.88));
            border: 1px solid rgba(23, 38, 30, 0.08);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.75rem;
            color: var(--accent);
            font-weight: 700;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1;
            font-weight: 700;
            margin: 0.45rem 0 0.4rem 0;
        }

        .hero-copy {
            max-width: 780px;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
            margin: 0;
        }

        .metric-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: var(--shadow);
            min-height: 126px;
        }

        .metric-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--muted);
            margin-bottom: 0.5rem;
        }

        .metric-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
            color: var(--text);
        }

        .metric-sub {
            color: var(--muted);
            margin-top: 0.5rem;
            font-size: 0.92rem;
        }

        .section-note {
            color: var(--muted);
            margin-top: -0.2rem;
            margin-bottom: 0.8rem;
        }

        .chip {
            display: inline-block;
            padding: 0.28rem 0.6rem;
            margin-right: 0.35rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.10);
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 600;
        }

        .player-card {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: var(--shadow);
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, subtext: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtext}</div>
    </div>
    """


def normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.fillna(False)
        .astype("string")
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def season_sort_key(season: str) -> tuple[int, str]:
    try:
        return (int(season.split("-")[0]), season)
    except (ValueError, IndexError):
        return (-1, season)


def available_seasons() -> list[str]:
    seasons = []
    for path in OUTPUT_DIR.glob("player_rankings_*.csv"):
        season = path.stem.replace("player_rankings_", "", 1)
        seasons.append(season)
    return sorted(set(seasons), key=season_sort_key, reverse=True)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=False)
def load_season_bundle(season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rankings = load_csv(OUTPUT_DIR / f"player_rankings_{season}.csv")
    top_by_role = load_csv(OUTPUT_DIR / f"top25_rankings_by_role_{season}.csv")
    audit = load_csv(OUTPUT_DIR / f"ranking_role_audit_{season}.csv")
    metrics = load_json(OUTPUT_DIR / f"model_metrics_{season}.json")

    if not rankings.empty and "meets_volume_threshold" in rankings.columns:
        rankings["meets_volume_threshold"] = normalize_bool(rankings["meets_volume_threshold"])
    if not audit.empty and "role_mismatch_flag" in audit.columns:
        audit["role_mismatch_flag"] = normalize_bool(audit["role_mismatch_flag"])

    for frame in [rankings, top_by_role, audit]:
        for col in ["overall_rank", "role_rank", "overall_score", "event_rallies", "calc_hit_pct", "sr_shrunk_value"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    return rankings, top_by_role, audit, metrics


def apply_rankings_filters(
    rankings: pd.DataFrame,
    roles: list[str],
    conferences: list[str],
    team: str,
    min_rallies: int,
    only_threshold: bool,
    search: str,
) -> pd.DataFrame:
    df = rankings.copy()
    if roles:
        df = df.loc[df["role_family"].isin(roles)]
    if conferences:
        df = df.loc[df["conference"].isin(conferences)]
    if team != "All teams":
        df = df.loc[df["team"].eq(team)]
    if "event_rallies" in df.columns:
        df = df.loc[pd.to_numeric(df["event_rallies"], errors="coerce").fillna(0).ge(min_rallies)]
    if only_threshold and "meets_volume_threshold" in df.columns:
        df = df.loc[df["meets_volume_threshold"]]

    search = search.strip().lower()
    if search:
        player_match = df["player"].fillna("").str.lower().str.contains(search, regex=False)
        team_match = df["team"].fillna("").str.lower().str.contains(search, regex=False)
        conference_match = df["conference"].fillna("").str.lower().str.contains(search, regex=False)
        df = df.loc[player_match | team_match | conference_match]

    sort_cols = [c for c in ["overall_rank", "role_rank", "overall_score"] if c in df.columns]
    ascending = [True, True, False][: len(sort_cols)]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, kind="mergesort")
    return df.reset_index(drop=True)


def apply_audit_filters(
    audit: pd.DataFrame,
    roles: list[str],
    conferences: list[str],
    team: str,
    min_rallies: int,
    search: str,
) -> pd.DataFrame:
    df = audit.copy()
    if roles and "role_family" in df.columns:
        df = df.loc[df["role_family"].isin(roles)]
    if conferences and "conference" in df.columns:
        df = df.loc[df["conference"].isin(conferences)]
    if team != "All teams" and "team" in df.columns:
        df = df.loc[df["team"].eq(team)]
    if "event_rallies" in df.columns:
        df = df.loc[pd.to_numeric(df["event_rallies"], errors="coerce").fillna(0).ge(min_rallies)]

    search = search.strip().lower()
    if search:
        player_match = df["player"].fillna("").str.lower().str.contains(search, regex=False)
        team_match = df["team"].fillna("").str.lower().str.contains(search, regex=False)
        reason_match = df["audit_reason"].fillna("").str.lower().str.contains(search, regex=False)
        df = df.loc[player_match | team_match | reason_match]
    return df.sort_values(["overall_rank", "role_rank"], kind="mergesort").reset_index(drop=True)


def compact_number(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if value.is_integer():
        return f"{int(value)}"
    return f"{value:.1f}"


def format_pct(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{100 * float(value):.{digits}f}%"


def render_overview(
    rankings: pd.DataFrame,
    top_by_role: pd.DataFrame,
    metrics: dict,
    selected_season: str,
) -> None:
    top_player = rankings.sort_values("overall_rank", kind="mergesort").head(1)
    top_name = top_player["player"].iloc[0] if not top_player.empty else "-"
    top_team = top_player["team"].iloc[0] if not top_player.empty else "-"
    players_ranked = metrics.get("ranking", {}).get("players_ranked", len(rankings))
    threshold_count = metrics.get("ranking", {}).get(
        "players_meeting_volume_threshold",
        int(rankings.get("meets_volume_threshold", pd.Series(dtype=bool)).sum()),
    )
    test_auc = metrics.get("win_probability", {}).get("test", {}).get("auc")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card("Season", selected_season, f"{compact_number(players_ranked)} total ranked players"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            metric_card("Volume Qualified", compact_number(threshold_count), "Players above the minimum-rally threshold"),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            metric_card("Top Overall", top_name, top_team),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            metric_card("WP Test AUC", "-" if test_auc is None else f"{float(test_auc):.3f}", "Holdout performance of the rally model"),
            unsafe_allow_html=True,
        )

    st.markdown("### Snapshot")
    st.markdown(
        '<div class="section-note">Quick view of the current season before any deep filtering.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        preview_cols = [
            c
            for c in ["overall_rank", "player", "team", "role_family", "overall_score", "event_rallies"]
            if c in rankings.columns
        ]
        st.dataframe(
            rankings.sort_values("overall_rank", kind="mergesort")[preview_cols].head(15),
            use_container_width=True,
            hide_index=True,
            column_config={
                "overall_score": st.column_config.NumberColumn("Overall Score", format="%.2f"),
                "event_rallies": st.column_config.NumberColumn("Event Rallies", format="%d"),
            },
        )

    with right:
        role_counts = rankings.groupby("role_family", observed=True).size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        palette = ["#0f766e", "#b45309", "#2b6cb0", "#7c3aed", "#2f855a", "#c05621"]
        ax.bar(role_counts.index, role_counts.values, color=palette[: len(role_counts)])
        ax.set_title("Players Ranked by Role", fontsize=13, fontweight="bold")
        ax.set_ylabel("Players")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### Top by Role")
    st.markdown(
        '<div class="section-note">These lists honor the model volume threshold and mirror the publishable role-based rankings.</div>',
        unsafe_allow_html=True,
    )
    role_preview = top_by_role.copy()
    if role_preview.empty:
        role_preview = rankings.loc[rankings.get("meets_volume_threshold", False)].copy()
    preview_cols = [
        c
        for c in ["role_family", "role_rank", "player", "team", "overall_score", "event_rallies"]
        if c in role_preview.columns
    ]
    st.dataframe(
        role_preview[preview_cols].head(25),
        use_container_width=True,
        hide_index=True,
        column_config={
            "overall_score": st.column_config.NumberColumn("Overall Score", format="%.2f"),
            "event_rallies": st.column_config.NumberColumn("Event Rallies", format="%d"),
        },
    )


def render_rankings_table(filtered: pd.DataFrame) -> None:
    st.markdown("### Filtered Rankings")
    st.markdown(
        '<div class="section-note">Use the sidebar to change season, role, team, conference, and rally threshold.</div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    display_cols = [c for c in CORE_RANKING_COLUMNS if c in filtered.columns]
    csv_data = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered rankings",
        data=csv_data,
        file_name="filtered_rankings.csv",
        mime="text/csv",
    )
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "overall_score": st.column_config.NumberColumn("Overall Score", format="%.2f"),
            "event_rallies": st.column_config.NumberColumn("Event Rallies", format="%d"),
            "calc_hit_pct": st.column_config.NumberColumn("Calc Hit %", format="%.3f"),
            "sr_shrunk_value": st.column_config.NumberColumn("SR Value", format="%.2f"),
        },
    )


def render_player_explorer(filtered: pd.DataFrame) -> None:
    st.markdown("### Player Explorer")
    st.markdown(
        '<div class="section-note">Pick a player from the filtered result set to inspect score components and season stats.</div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.info("Change the filters first so there is at least one player to inspect.")
        return

    options = filtered.sort_values("overall_rank", kind="mergesort").copy()
    options["player_option"] = (
        "#"
        + options["overall_rank"].fillna(0).astype("Int64").astype("string")
        + " | "
        + options["player"].fillna("Unknown")
        + " | "
        + options["team"].fillna("Unknown")
    )
    selected_label = st.selectbox("Player", options["player_option"].tolist(), index=0)
    player_row = options.loc[options["player_option"].eq(selected_label)].iloc[0]

    chips = [
        player_row.get("team", "-"),
        player_row.get("conference", "-"),
        player_row.get("role_family", "-"),
    ]
    chip_html = "".join(f'<span class="chip">{chip}</span>' for chip in chips if chip and chip != "-")
    st.markdown(
        f"""
        <div class="player-card">
            <div class="hero-kicker">Player Profile</div>
            <div class="hero-title" style="font-size: 2rem;">{player_row.get("player", "-")}</div>
            <div style="margin: 0.4rem 0 0.7rem 0;">{chip_html}</div>
            <div class="hero-copy">Overall rank #{int(player_row.get("overall_rank", 0))} and role rank #{int(player_row.get("role_rank", 0))} with an overall score of {float(player_row.get("overall_score", 0.0)):.2f}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Overall Score", f"{float(player_row.get('overall_score', 0.0)):.2f}")
    with c2:
        st.metric("Event Rallies", compact_number(player_row.get("event_rallies")))
    with c3:
        st.metric("Hit %", "-" if pd.isna(player_row.get("calc_hit_pct")) else f"{float(player_row.get('calc_hit_pct', 0.0)):.3f}")
    with c4:
        st.metric("Serve/Receive Value", "-" if pd.isna(player_row.get("sr_shrunk_value")) else f"{float(player_row.get('sr_shrunk_value', 0.0)):.2f}")

    left, right = st.columns(2, gap="large")
    with left:
        comp_data = pd.DataFrame(
            [
                ("Event value z", player_row.get("event_value_z")),
                ("Tabular value z", player_row.get("tabular_value_z")),
                ("Serve/receive value z", player_row.get("serve_receive_value_z")),
                ("Final value z", player_row.get("final_value_z")),
                ("Event blend weight", player_row.get("event_blend_weight")),
                ("Tabular blend weight", player_row.get("tabular_blend_weight")),
                ("Serve/receive blend weight", player_row.get("serve_receive_blend_weight")),
            ],
            columns=["Component", "Value"],
        )
        st.markdown("#### Model Components")
        st.dataframe(
            comp_data,
            use_container_width=True,
            hide_index=True,
            column_config={"Value": st.column_config.NumberColumn("Value", format="%.3f")},
        )

    with right:
        stat_rows = [
            ("Kills", player_row.get("box_kills")),
            ("Assists", player_row.get("box_assists")),
            ("Digs", player_row.get("box_digs")),
            ("Attacks", player_row.get("box_total_attacks")),
            ("Aces", player_row.get("box_aces")),
            ("Reception attempts", player_row.get("box_retatt")),
            ("Touches per rally", player_row.get("touches_per_rally")),
            ("Calculated hit %", player_row.get("calc_hit_pct")),
        ]
        stat_data = pd.DataFrame(stat_rows, columns=["Stat", "Value"])
        st.markdown("#### Season Stat Line")
        st.dataframe(
            stat_data,
            use_container_width=True,
            hide_index=True,
            column_config={"Value": st.column_config.NumberColumn("Value", format="%.3f")},
        )


def render_audit(audit_filtered: pd.DataFrame) -> None:
    st.markdown("### Ranking Audit")
    st.markdown(
        '<div class="section-note">Flags help spot role mismatches, low-volume outliers, and rankings that deserve a manual check.</div>',
        unsafe_allow_html=True,
    )

    if audit_filtered.empty:
        st.info("No audit rows match the current filters.")
        return

    flagged_only = audit_filtered.loc[audit_filtered["audit_reason"].fillna("").ne("")]
    st.markdown(
        metric_card(
            "Flagged Players",
            compact_number(len(flagged_only)),
            "Rows with at least one audit reason under the current filter set",
        ),
        unsafe_allow_html=True,
    )

    display_cols = [c for c in AUDIT_COLUMNS if c in audit_filtered.columns]
    st.dataframe(
        audit_filtered[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "overall_score": st.column_config.NumberColumn("Overall Score", format="%.2f"),
            "event_rallies": st.column_config.NumberColumn("Event Rallies", format="%d"),
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="VPR Rankings Dashboard",
        page_icon="VB",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    seasons = available_seasons()
    if not seasons:
        st.error(
            "No ranking files were found in Data/model_outputs. Run the ranking pipeline first."
        )
        st.code(
            "python src/modules/log_regression/fullDataPipeline.py --season 2025-2026\n"
            "python src/modules/log_regression/baselineModel.py --season 2025-2026"
        )
        st.stop()

    with st.sidebar:
        st.markdown("## VPR Filters")
        selected_season = st.selectbox("Season", seasons, index=0)

    rankings, top_by_role, audit, metrics = load_season_bundle(selected_season)
    if rankings.empty:
        st.error(f"Could not load rankings for {selected_season}.")
        st.stop()

    roles = sorted(rankings["role_family"].dropna().astype(str).unique().tolist())
    conferences = sorted(rankings["conference"].dropna().astype(str).unique().tolist())
    teams = sorted(rankings["team"].dropna().astype(str).unique().tolist())
    max_rallies = int(pd.to_numeric(rankings["event_rallies"], errors="coerce").fillna(0).max())
    default_rallies = 200 if max_rallies >= 200 else 0

    with st.sidebar:
        selected_roles = st.multiselect("Role", roles, default=roles)
        selected_conferences = st.multiselect("Conference", conferences, default=[])
        selected_team = st.selectbox("Team", ["All teams", *teams], index=0)
        min_rallies = st.slider("Minimum event rallies", 0, max_rallies, default_rallies, step=25)
        only_threshold = st.checkbox("Only show volume-qualified players", value=True)
        search = st.text_input("Search player, team, or conference", value="")

    filtered_rankings = apply_rankings_filters(
        rankings=rankings,
        roles=selected_roles,
        conferences=selected_conferences,
        team=selected_team,
        min_rallies=min_rallies,
        only_threshold=only_threshold,
        search=search,
    )
    filtered_audit = apply_audit_filters(
        audit=audit,
        roles=selected_roles,
        conferences=selected_conferences,
        team=selected_team,
        min_rallies=min_rallies,
        search=search,
    )

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Volleyball Performance Rankings</div>
            <div class="hero-title">Rankings Dashboard</div>
            <p class="hero-copy">
                Explore the model outputs by season, role, conference, team, and rally volume.
                The season selector auto-detects whatever ranking files are available in
                <code>Data/model_outputs</code>, so the dashboard expands naturally as you build more years.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_tab, rankings_tab, player_tab, audit_tab = st.tabs(
        ["Overview", "Rankings", "Player Explorer", "Audit"]
    )

    with overview_tab:
        render_overview(rankings, top_by_role, metrics, selected_season)
    with rankings_tab:
        render_rankings_table(filtered_rankings)
    with player_tab:
        render_player_explorer(filtered_rankings)
    with audit_tab:
        render_audit(filtered_audit)


if __name__ == "__main__":
    main()
