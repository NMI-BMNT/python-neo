"""
Class for reading data from Multi Channel Systems (MCS) HDF5 files.

Based on the HDF5 MCS Raw Data Definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neo.core import NeoReadWriteError
from .baserawio import (
    BaseRawIO,
    _signal_channel_dtype,
    _signal_stream_dtype,
    _signal_buffer_dtype,
    _spike_channel_dtype,
    _event_channel_dtype,
)


@dataclass
class _McsH5Stream:
    stream_id: str
    name: str
    sampling_rate: float
    channel_data: object
    time_axis: int
    channel_names: list[str]
    units: list[str]
    gains: list[float]
    offsets: list[float]

    @property
    def num_channels(self) -> int:
        return len(self.channel_names)

    @property
    def num_samples(self) -> int:
        if self.time_axis == 0:
            return self.channel_data.shape[0]
        return self.channel_data.shape[1]


class McsH5RawIO(BaseRawIO):
    """
    Class for reading MCS HDF5 raw data files.

    Parameters
    ----------
    filename: str, default: ''
        The *.h5 file to be read.
    """

    extensions = ["h5", "hdf5"]
    rawmode = "one-file"

    def __init__(self, filename=""):
        BaseRawIO.__init__(self)
        self.filename = filename
        self._h5file = None
        self._segments: list[list[_McsH5Stream]] = []

    def _source_name(self):
        return self.filename

    def _parse_header(self):
        try:
            import h5py
        except ImportError as exc:
            raise NeoReadWriteError("McsH5RawIO requires the h5py package.") from exc

        self._h5file = h5py.File(self.filename, "r")

        data_group = self._h5file.get("Data")
        if data_group is None:
            raise NeoReadWriteError("Could not find the 'Data' group in the MCS HDF5 file.")

        recording_keys = sorted(
            [key for key in data_group.keys() if key.startswith("Recording_")],
            key=lambda name: int(name.split("_")[1]),
        )
        if not recording_keys:
            raise NeoReadWriteError("No Recording_* groups found in the MCS HDF5 file.")

        for recording_key in recording_keys:
            recording_group = data_group[recording_key]
            analog_group = recording_group.get("AnalogStream")
            if analog_group is None:
                raise NeoReadWriteError(f"No AnalogStream group found in {recording_key}.")

            stream_keys = sorted(
                [key for key in analog_group.keys() if key.startswith("Stream_")],
                key=lambda name: int(name.split("_")[1]),
            )
            if not stream_keys:
                raise NeoReadWriteError(f"No Stream_* groups found in {recording_key}/AnalogStream.")

            segment_streams: list[_McsH5Stream] = []
            for stream_key in stream_keys:
                stream_group = analog_group[stream_key]
                stream = self._parse_stream(stream_group, recording_group, stream_key)
                segment_streams.append(stream)

            self._segments.append(segment_streams)

        first_segment = self._segments[0]

        signal_buffers = np.array([], dtype=_signal_buffer_dtype)
        signal_streams = np.array(
            [(stream.name, stream.stream_id, "") for stream in first_segment],
            dtype=_signal_stream_dtype,
        )

        sig_channels = []
        for stream in first_segment:
            for channel_index in range(stream.num_channels):
                sig_channels.append(
                    (
                        stream.channel_names[channel_index],
                        str(channel_index),
                        stream.sampling_rate,
                        str(stream.channel_data.dtype),
                        stream.units[channel_index],
                        stream.gains[channel_index],
                        stream.offsets[channel_index],
                        stream.stream_id,
                        "",
                    )
                )
        sig_channels = np.array(sig_channels, dtype=_signal_channel_dtype)

        event_channels = np.array([], dtype=_event_channel_dtype)
        spike_channels = np.array([], dtype=_spike_channel_dtype)

        self.header = {
            "nb_block": 1,
            "nb_segment": [len(self._segments)],
            "signal_buffers": signal_buffers,
            "signal_streams": signal_streams,
            "signal_channels": sig_channels,
            "spike_channels": spike_channels,
            "event_channels": event_channels,
        }

        self._generate_minimal_annotations()

    def _segment_t_start(self, block_index, seg_index):
        return 0.0

    def _segment_t_stop(self, block_index, seg_index):
        stream = self._segments[seg_index][0]
        return stream.num_samples / stream.sampling_rate

    def _get_signal_size(self, block_index, seg_index, stream_index):
        return self._segments[seg_index][stream_index].num_samples

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return 0.0

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop, stream_index, channel_indexes):
        stream = self._segments[seg_index][stream_index]
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = stream.num_samples
        if channel_indexes is None:
            channel_indexes = slice(None)

        if stream.time_axis == 0:
            data = stream.channel_data[i_start:i_stop, channel_indexes]
        else:
            data = stream.channel_data[channel_indexes, i_start:i_stop].T
        return np.asarray(data)

    def _parse_stream(self, stream_group, recording_group, stream_key):
        channel_data = stream_group.get("ChannelData")
        if channel_data is None:
            raise NeoReadWriteError(f"Missing ChannelData dataset in {stream_key}.")
        if channel_data.ndim != 2:
            raise NeoReadWriteError(f"ChannelData in {stream_key} is expected to be 2D.")

        info_channel = stream_group.get("InfoChannel")
        channel_names = None
        units = None
        gains = None
        offsets = None

        if info_channel is not None:
            channel_names = self._read_info_field(info_channel, ["Label", "ChannelLabel", "Name"])
            units = self._read_info_field(info_channel, ["Unit", "Units"])
            gains = self._read_info_field(info_channel, ["ConversionFactor", "ConversionFactorV", "Gain"])
            offsets = self._read_info_field(info_channel, ["ConversionOffset", "Offset"])
            ad_zero = self._read_info_field(info_channel, ["ADZero"])
        else:
            ad_zero = None

        if info_channel is not None:
            num_channels = info_channel.shape[0]
            if channel_data.shape[0] == num_channels and channel_data.shape[1] != num_channels:
                time_axis = 1
            elif channel_data.shape[1] == num_channels and channel_data.shape[0] != num_channels:
                time_axis = 0
            else:
                time_axis = 1
        else:
            if channel_data.shape[0] <= channel_data.shape[1]:
                num_channels = channel_data.shape[0]
                time_axis = 1
            else:
                num_channels = channel_data.shape[1]
                time_axis = 0

        channel_names = self._coerce_info_array(channel_names, num_channels, "channel names")
        units = self._coerce_info_array(units, num_channels, "units")
        gains = self._coerce_info_array(gains, num_channels, "gains")
        offsets = self._coerce_info_array(offsets, num_channels, "offsets")
        ad_zero = self._coerce_info_array(ad_zero, num_channels, "ADZero")

        if channel_names is not None:
            channel_names = [self._decode_text(value) for value in channel_names]
        if units is not None:
            units = [self._decode_text(value) for value in units]

        if channel_names is None:
            channel_names = [f"ch{index}" for index in range(num_channels)]
        if units is None:
            units = ["uV"] * num_channels
        if gains is None:
            gains = np.ones(num_channels, dtype=float)
        if offsets is None:
            offsets = np.zeros(num_channels, dtype=float)

        if ad_zero is not None and offsets is not None and np.allclose(offsets, 0):
            offsets = -np.asarray(ad_zero, dtype=float) * np.asarray(gains, dtype=float)

        sampling_rate = self._read_sampling_rate(stream_group, recording_group)

        return _McsH5Stream(
            stream_id=stream_key,
            name=stream_key,
            sampling_rate=sampling_rate,
            channel_data=channel_data,
            time_axis=time_axis,
            channel_names=list(channel_names),
            units=list(units),
            gains=list(np.asarray(gains, dtype=float)),
            offsets=list(np.asarray(offsets, dtype=float)),
        )

    def _read_sampling_rate(self, stream_group, recording_group):
        for group in (stream_group, recording_group):
            if group is None:
                continue
            sampling_rate = group.attrs.get("SamplingRate")
            if sampling_rate is not None:
                return float(sampling_rate)
        raise NeoReadWriteError("Could not determine sampling rate for MCS stream.")

    @staticmethod
    def _read_info_field(info_channel, field_names):
        for name in field_names:
            if info_channel.dtype.fields and name in info_channel.dtype.fields:
                return info_channel[name]
        return None

    @staticmethod
    def _coerce_info_array(values, num_channels, label):
        if values is None:
            return None
        array = np.atleast_1d(values)
        if array.size == 1 and num_channels > 1:
            return np.broadcast_to(array, (num_channels,))
        if array.size != num_channels:
            raise NeoReadWriteError(
                f"Unexpected {label} length {array.size}; expected {num_channels} entries."
            )
        return array

    @staticmethod
    def _decode_text(value):
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.bytes_):
            return value.astype(str)
        return str(value)
