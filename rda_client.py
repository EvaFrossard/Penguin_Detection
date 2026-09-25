"""Minimal client for BrainVision Recorder's RDA (Remote Data Access) protocol, 32-bit float version.

Recorder must have RDA enabled (Configuration > Preferences > Remote Data Access). The 32-bit stream is served on
TCP port 51244. Every message is a 24-byte header (16-byte GUID, uint32 size, uint32 type) followed by a body:
    type 1 = START: channel count, sampling interval (us), per-channel resolution (uV/unit), channel names
    type 4 = DATA : block number, points, marker count, points x channels float32 (multiplexed), then markers
    type 3 = STOP : acquisition stopped
Marker positions are relative to the first sample of the block they arrive in.
"""
import socket
import struct
from dataclasses import dataclass, field

import numpy as np

RDA_PORT_32BIT = 51244
RDA_GUID = bytes([0x8E, 0x45, 0x58, 0x43, 0x96, 0xC9, 0x86, 0x4C,
                  0xAF, 0x4A, 0x98, 0xBB, 0xF6, 0xC9, 0x14, 0x50])

MSG_START = 1
MSG_STOP = 3
MSG_DATA32 = 4


@dataclass
class StartInfo:
    channel_names: list
    fs: float              # sampling rate (Hz)
    resolutions: np.ndarray  # multiply raw values by these to get uV (same as load_channel)


@dataclass
class Marker:
    position: int          # sample index within the block
    points: int
    channel: int           # -1 = all channels
    type: str              # e.g. "Stimulus"
    description: str       # e.g. "S  1"

    @property
    def code(self):
        """Stimulus code as an int ("S  1" -> 1), or None for other markers. Same parsing as load_events."""
        if self.type != "Stimulus" or not self.description.startswith("S"):
            return None
        try:
            return int(self.description[1:].strip())
        except ValueError:
            return None


@dataclass
class DataBlock:
    block: int             # block counter from Recorder (consecutive unless blocks were dropped)
    data: np.ndarray       # shape (points, n_channels), float64, already scaled to uV
    markers: list = field(default_factory=list)


def _split_strings(raw: bytes):
    return [s.decode("utf-8", errors="replace") for s in raw.split(b"\x00")]


class RDAClient:
    """Usage:
        with RDAClient("localhost") as client:
            for msg in client.messages():
                if isinstance(msg, StartInfo): ...
                elif isinstance(msg, DataBlock): ...
                elif msg is None: ...   # STOP
    """

    def __init__(self, host="localhost", port=RDA_PORT_32BIT, timeout_s=None):
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self.sock = None
        self.info = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("RDA connection closed by Recorder")
            buf += chunk
        return bytes(buf)

    def messages(self):
        """Yield StartInfo, DataBlock, or None (STOP) as they arrive. Other message types are skipped."""
        while True:
            header = self._recv_exact(24)
            guid, size, mtype = header[:16], *struct.unpack("<LL", header[16:])
            if guid != RDA_GUID:
                raise ConnectionError("Unexpected data on RDA socket (bad GUID). Is this the 32-bit port 51244?")
            body = self._recv_exact(size - 24)

            if mtype == MSG_START:
                self.info = self._parse_start(body)
                yield self.info
            elif mtype == MSG_DATA32:
                if self.info is None:
                    continue  # data before any START (connected mid-stream): wait for the next START
                yield self._parse_data(body)
            elif mtype == MSG_STOP:
                yield None

    @staticmethod
    def _parse_start(body: bytes) -> StartInfo:
        n_channels, sampling_interval_us = struct.unpack("<Ld", body[:12])
        resolutions = np.frombuffer(body, dtype="<f8", count=n_channels, offset=12).astype(np.float64)
        names = _split_strings(body[12 + 8 * n_channels:])[:n_channels]
        return StartInfo(channel_names=names, fs=1e6 / sampling_interval_us, resolutions=resolutions)

    def _parse_data(self, body: bytes) -> DataBlock:
        n_ch = len(self.info.channel_names)
        block, points, n_markers = struct.unpack("<LLL", body[:12])
        raw = np.frombuffer(body, dtype="<f4", count=points * n_ch, offset=12)
        data = raw.reshape(points, n_ch).astype(np.float64) * self.info.resolutions

        markers = []
        idx = 12 + 4 * points * n_ch
        for _ in range(n_markers):
            m_size, = struct.unpack("<L", body[idx:idx + 4])
            position, m_points, channel = struct.unpack("<LLl", body[idx + 4:idx + 16])
            strings = _split_strings(body[idx + 16:idx + m_size])
            markers.append(Marker(position, m_points, channel,
                                  strings[0] if strings else "", strings[1] if len(strings) > 1 else ""))
            idx += m_size
        return DataBlock(block=block, data=data, markers=markers)
