from __future__ import annotations

from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class RTPPacket:
    sequence: int
    timestamp: int
    payload: bytes
    sample_rate: int = 16_000
    payload_type: int = 11
    marker: bool = False

    def encode(self) -> bytes:
        flags=(2<<6) | (0x80 if self.marker else 0); return struct.pack(">BBHII",flags,self.payload_type,self.sequence,self.timestamp,0)+self.payload

    @classmethod
    def decode(cls, data: bytes, sample_rate: int = 16_000) -> "RTPPacket":
        if len(data)<12: raise ValueError("RTP-like packet is shorter than the 12-byte header")
        flags,payload_type,sequence,timestamp,_=struct.unpack(">BBHII",data[:12])
        if flags>>6 != 2: raise ValueError("unsupported RTP version")
        return cls(sequence,timestamp,data[12:],sample_rate,payload_type,bool(flags&0x80))
