import os
from typing import cast

import pandas as pd


def generate_summary(path: str) -> pd.DataFrame:
  """Generate summary statistics from metrics.csv files in subject directories.

  Args:
      path (str): Path containing version folders with metrics.csv files

  Returns:
      pd.DataFrame: Summary dataframe with mean and std rows
  """
  version_list = []
  for version_name in os.listdir(path):
    if "subject" in version_name and not version_name.startswith("_"):
      version_list.append(version_name)

  version_list = sorted(version_list)

  summary_frames: list[pd.DataFrame] = []
  for version_name in version_list:
    metrics_csv = os.path.join(path, version_name, "metrics.csv")
    metrics_pd = pd.read_csv(metrics_csv)
    test_res_columns = [col for col in metrics_pd.columns if "test" in col]
    # Keep rows where ANY test column is populated; one fold's NaN gate-loss
    # shouldn't drop the whole row and break the row-count alignment below.
    summary_frames.append(
      cast(pd.DataFrame, metrics_pd[test_res_columns].dropna(how="all"))
    )

  if not summary_frames:
    raise ValueError(f"No subject metrics found in {path}")

  summary_pd = pd.concat(summary_frames)
  summary_pd.index = version_list
  summary_pd.loc["mean"] = summary_pd.mean()
  summary_pd.loc["std"] = summary_pd.std()

  return summary_pd


def generate_summary_accross_versions(
  path: str, versions: list[str]
) -> pd.DataFrame:
  summary_frames: list[pd.DataFrame] = []
  for version_name in versions:
    metrics_csv = os.path.join(path, version_name, "metrics.csv")
    metrics_pd = pd.read_csv(metrics_csv)
    test_res_columns = [col for col in metrics_pd.columns if "test" in col]
    summary_frames.append(
      cast(pd.DataFrame, metrics_pd[test_res_columns].dropna(how="all"))
    )

  if not summary_frames:
    raise ValueError("At least one version is required")

  summary_pd = pd.concat(summary_frames)
  summary_pd.index = versions
  summary_pd.loc["mean"] = summary_pd.mean()
  summary_pd.loc["std"] = summary_pd.std()

  return summary_pd


def generate_kfold_summary(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
  df = generate_summary(path)
  columns = pd.Index([str(column) for column in df.columns] + ["session"])
  by_session_summary = pd.DataFrame(columns=columns)

  for row_idx in df.index:
    if row_idx not in ["mean", "std"]:
      row = df.loc[row_idx].copy()
      name_components = str(row_idx).split("_")
      row["session"] = f"{name_components[0]}_{name_components[1]}"
      by_session_summary.loc[len(by_session_summary)] = row

  by_session_summary = cast(
    pd.DataFrame, by_session_summary.groupby("session").mean()
  )
  by_session_summary.loc["mean"] = by_session_summary.mean()
  by_session_summary.loc["std"] = by_session_summary.std()

  return by_session_summary, df
