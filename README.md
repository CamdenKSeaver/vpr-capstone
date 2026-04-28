If using mamba/conda:
* Create environment with `mamba env create -f environment.yml`
* To add new dependencies, update the `environment.yml`, then update the environment with `mamba env update -f environment.yml`

If using pip/venv:
* Install the dependencies with `pip install -r requirements.txt`

Run the reproducible data and ranking pipeline from the repository root:
* Build the preprocessed season tables only with `python src/modules/log_regression/fullDataPipeline.py --season 2025-2026`
* Train the win-probability and ranking models only with `python src/modules/log_regression/baselineModel.py --season 2025-2026`
* Run the full stack for one season with `python src/modules/log_regression/run_full_stack.py --season 2025-2026`
* Run the full stack for every supported season with `python src/modules/log_regression/run_full_stack.py --season-range 2022-2023 2025-2026`
* The default win-probability model is now MLP for both `baselineModel.py` and `run_full_stack.py`
* Run the logistic win-probability model explicitly with `python src/modules/log_regression/run_full_stack.py --season 2025-2026 --wp-model logistic`
* Or run the wrapper entrypoint with `python src/main.py`
* `python src/main.py` defaults to the full log-regression stack; use `python src/main.py log_regression_pipeline` or `python src/main.py ranking` to select another module
* Missing source files in `Data/original` are downloaded automatically for supported seasons `2022-2023` through `2025-2026`
* Preprocessing writes these season CSVs to `Data/preprocessed`:
  * `player_master_<season>.csv`
  * `team_strength_<season>.csv`
  * `contest_master_<season>.csv`
  * `rally_table_<season>.csv`
  * `rally_pass_<season>.csv`
  * `first_ball_pass_player_contest_<season>.csv`
  * `first_ball_pass_player_season_<season>.csv`
  * `player_match_enriched_<season>.csv`
  * `player_season_features_<season>.csv`
  * `event_table_<season>.csv`
* Ranking/modeling writes these season files to `Data/model_outputs`:
  * `rally_win_probability_<season>.csv`
  * `player_event_credit_<season>.csv`
  * `team_phase_context_<season>.csv`
  * `server_toughness_<season>.csv`
  * `serve_receive_player_value_<season>.csv`
  * `player_rankings_<season>.csv`
  * `ranking_role_audit_<season>.csv`
  * `top25_rankings_by_role_<season>.csv`
* When more than one season is requested, combined outputs are also written to `Data/model_outputs`:
  * `player_rankings_all_seasons.csv`
  * `top25_rankings_by_role_all_seasons.csv`
  * `ranking_role_audit_all_seasons.csv`
  * `serve_receive_player_value_all_seasons.csv`
  * `server_toughness_all_seasons.csv`
  * `team_phase_context_all_seasons.csv`
* Explore rankings interactively with `src/visualize_outputs.ipynb`
* Launch the interactive dashboard with `streamlit run Dashboard/streamlit_app.py`

Most source data is small enough to store in the repository, but the play-by-play file is large and should stay out of git.

Set up Tableau Desktop locally via https://www.tableau.com/products/desktop-free/download. From the application, you should be able to open our Dashboard/dashboard.twb file, committing updates to the repository when significant changes are made.
* There's potentially a licensed version available at https://www.tableau.com/academic/students, which may work.
