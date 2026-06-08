"""
QuantaAlpha custom workspace.

Overrides rdagent QlibFBWorkspace: project-level factor_template overrides default YAML;
base files (read_exp_res.py, etc.) still from rdagent; init empty git repo in workspace to suppress qlib recorder git output.
On macOS/venv (no conda), overrides execute() to use LocalEnv with venv/bin in PATH so qrun is found.
"""

import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from rdagent.scenarios.qlib.experiment.workspace import QlibFBWorkspace as _RdagentQlibFBWorkspace
from rdagent.utils.env import LocalConf, LocalEnv
from rdagent.log import rdagent_logger as logger

_CUSTOM_TEMPLATE_DIR = Path(__file__).resolve().parent / "factor_template"


class QlibFBWorkspace(_RdagentQlibFBWorkspace):
    """
    Override rdagent QlibFBWorkspace: inject project factor_template/ YAML over defaults;
    init empty git repo in workspace to avoid qlib recorder git help output.
    """

    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(template_folder_path, *args, **kwargs)
        if _CUSTOM_TEMPLATE_DIR.exists():
            self.inject_code_from_folder(_CUSTOM_TEMPLATE_DIR)
            logger.info(f"Overrode rdagent default config with project template: {_CUSTOM_TEMPLATE_DIR}")

    def before_execute(self) -> None:
        """Init empty git repo in workspace to suppress qlib recorder git warnings."""
        super().before_execute()
        git_dir = self.workspace_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=str(self.workspace_path),
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass

    def execute(self, qlib_config_name: str = "conf_baseline.yaml", run_env: dict = {}, *args, **kwargs):
        """
        Override to use LocalEnv (venv) when conda is unavailable (macOS local dev).

        Falls back to the rdagent default (conda/docker) when CONDA_DEFAULT_ENV is set.
        When running in a venv without conda, prepend VIRTUAL_ENV/bin to PATH so that
        `qrun` (installed in the venv) is found by the subprocess.
        """
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        if conda_env:
            # conda is active – use the standard rdagent implementation
            return super().execute(qlib_config_name=qlib_config_name, run_env=run_env, *args, **kwargs)

        # --- venv / no-conda path ---
        venv_root = os.environ.get("VIRTUAL_ENV", "")
        bin_path = f"{venv_root}/bin" if venv_root else ""
        # default_entry is required by EnvConf but we always pass explicit entry; use placeholder
        local_env = LocalEnv(conf=LocalConf(bin_path=bin_path, enable_cache=False, default_entry="qrun conf_baseline.yaml"))
        local_env.prepare()

        # Run qlib backtest
        execute_qlib_log = local_env.check_output(
            local_path=str(self.workspace_path),
            entry=f"qrun {qlib_config_name}",
            env=run_env or {},
        )
        logger.log_object(execute_qlib_log, tag="Qlib_execute_log")

        # Read back the factor result
        execute_log = local_env.check_output(
            local_path=str(self.workspace_path),
            entry="python read_exp_res.py",
            env=run_env or {},
        )

        quantitative_backtesting_chart_path = self.workspace_path / "ret.pkl"
        if quantitative_backtesting_chart_path.exists():
            ret_df = pd.read_pickle(quantitative_backtesting_chart_path)
            logger.log_object(ret_df, tag="Quantitative Backtesting Chart")
        else:
            logger.error("No result file found.")
            return None, execute_qlib_log

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log
