import numpy as np
import pandas as pd
from dataclasses import dataclass, InitVar

from metaclean3.channels import get_clean_time_chan, get_clean_fp_channels
from metaclean3.binning import get_time_bin_unit, get_time_binned

@dataclass(init=True)
class FCSfile:
    """Prepares an FCS file for MetaClean3.0.

    Attributes:
        data (pandas.DataFrame): FCS file data matrix sorted by time.
        time_step (float | None, optional):
            See :func:`binning.get_time_binned`. Defaults to None.
        time_chan (str): Time channel name. Defaults to 'time'.
        bin_chan (str, optional): Name of the bin column to be added.
        fluo_chans (list | np.ndarray, optional):
            Fluorescent channels that can be used for 0.
            Defaults to `np.array([])`.
        phys_chans (list | np.ndarray, optional):
            Physical morphological channels that can be used for 0.
            Defaults to `np.array([])`.
    """
    data: pd.DataFrame
    time_step: float | None = None # can only be None if time_bin is specified; meta.get('$TIMESTEP')
    time_chan: str = 'time'
    bin_chan: str = 'bin'
    fluo_chans: list | np.ndarray = np.array([])
    phys_chans: list | np.ndarray = np.array([])

    time_bin: InitVar[str | None] = None
    channel_unique_no: InitVar[int] = 25 # minimum number of unique values per column
    min_bin_size: InitVar[int] = 2000
    max_bin_size: InitVar[int] = 10000
    min_events_per_bin: InitVar[int] = 50

    def __post_init__(
        self,
        time_bin,
        channel_unique_no,
        min_bin_size,
        max_bin_size,
        min_events_per_bin
    ):
        """
        Args:
            time_bin (str, optional): See :func:`binning.get_time_binned`.
                Defaults to '1S'.
            channel_unique_no (int):
                See :func:`channels.get_clean_fp_channels`. Defaults to 25.
            min_bin_size (int, optional):
                See :func:`binning.get_time_binned`. Defaults to 2000.
            max_bin_size (int, optional):
                See :func:`binning.get_time_binned`. Defaults to 10000.
            min_events_per_bin (int, optional):
                See :func:`binning.get_time_binned`. Defaults to 50.
        """
        if len(self.data) == 0:
            raise ValueError('Data has no events.')

        # process channels: time; reorder data if needed
        self.data['index_original'] = self.data.index
        self.time_chan, self.data = get_clean_time_chan(
            data=self.data,
            time_chan=self.time_chan,
            min_bin_size=min_bin_size
        )
        # process channels: fluo, phys
        self.fluo_chans, self.phys_chans = get_clean_fp_channels(
            data=self.data,
            fluo_chans=self.fluo_chans,
            phys_chans=self.phys_chans,
            channel_unique_no=channel_unique_no
        )
        if self.fluo_chans is None or len(self.fluo_chans) == 0:
            raise ValueError('No fluorescent channels found.')

        # bin data by time
        time_bin = get_time_bin_unit(
            time_values=self.data[self.time_chan],
            time_bin=time_bin
        )
        self.data[self.bin_chan] = get_time_binned(
            time_values=self.data[self.time_chan],
            time_bin=time_bin,
            time_step=self.time_step,
            min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_events_per_bin=min_events_per_bin
        )
