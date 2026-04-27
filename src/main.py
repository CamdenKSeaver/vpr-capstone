"""
Add code here to run your module, using config JSON files to specify parameters. This is preferred over CLI interfaces for each module.
"""

import json
from modules.log_regression.baselineModel import main as log_regression_main
from modules.log_regression.fullDataPipeline import main as log_regression_pipeline_main
from modules.ranker.calc_stats import main as calc_stats_main
from scmrepo.git import Git
from pathlib import Path

PACKAGE_ROOT = Path(Git(root_dir=".").root_dir)
config = json.load(open(PACKAGE_ROOT / "config.json"))
# Edit your config to "log_regression" or "log_regression_pipeline" or "ranking" to run the modules
#   Do NOT push changes to the config.json unless they are meaningful (not just whatever module you ran last)
if config["module"] == "log_regression":
    print("Running log regression module with data pipeline")
    log_regression_pipeline_main()
    log_regression_main()
elif config["module"] == "ranking":
    print("Running ranking module")
    calc_stats_main()
else:
    print(f"Unknown module {config['module']}")
