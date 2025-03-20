from pathlib import Path
from typing import Tuple

from pandas import DataFrame, Series
from pynwb import NWBHDF5IO


def filter_df_by_dict(df: DataFrame, filter_dict: dict) -> dict:
    """Filter a DataFrame by a dict of cols names/values. Return row as dict."""
    mask = df[list(filter_dict)].eq(Series(filter_dict)).all(axis=1)
    return df[mask].to_dict(orient="records").pop()


def load_data(
    path_to_nwb: str, df_obj_ids: dict
) -> Tuple[DataFrame, DataFrame]:
    """Load data and events from an NWB file.

    Assumes events_per_epoch is one of the objects to load.

    Parameters
    ----------
    path_to_nwb : str
        Path to the NWB file.
    df_obj_ids : Dict[str, UUID]
        Dictionary with the names of the objects to load as keys and their
        corresponding object ids as values.

    Returns
    -------
    indexed_dataframes : DataFrame
        Dictionary of DataFrames with the data from the objects.
    events_per_epoch_df : DataFrame
        DataFrame with the events per epoch.

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if not Path(path_to_nwb).exists():
        raise FileNotFoundError(f"File not found: {path_to_nwb}")

    with NWBHDF5IO(path_to_nwb, "r") as io:
        nwb = io.read()
        dataframes = {
            name: nwb.objects[obj_id].to_dataframe()
            for name, obj_id in df_obj_ids.items()
        }

    events_per_epoch_df = dataframes.pop("events_per_epoch_df")

    indexed_dataframes = {k: set_index(v) for k, v in dataframes.items()}

    return indexed_dataframes, events_per_epoch_df


def set_index(df: DataFrame) -> DataFrame:
    """Set the index of the dataframe to 'time' or 'index' if they exist."""
    cols = df.columns
    ind_col = "time" if "time" in cols else "index" if "index" in cols else None
    return df.set_index(ind_col) if ind_col else df
