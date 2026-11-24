"""Read and write the LSTM network inside a Tesseract .traineddata file (used by average.py).

From Rivok Labs' weight-space tools; in this repository under its Apache-2.0 licence.

The layout follows Tesseract 5.5.0:
    ccutil/tessdatamanager    container: int32 count, int64 offsets (-1 = absent), components
    lstm/lstmrecognizer.cpp   'lstm' component: network, [unicharset], tail fields, [recoder]
    lstm/network.cpp          layer header
    lstm/plumbing.cpp         Series / Parallel / Reversed: child count, children, [learning rates]
    lstm/reconfig.cpp         Reconfig and Maxpool: x_scale, y_scale
    lstm/convolve.cpp         half_x, half_y
    lstm/input.cpp            StaticShape: batch, height, width, depth, loss_type
    lstm/fullyconnected.cpp   one WeightMatrix
    lstm/lstm.cpp             na, gate WeightMatrices CI GI GF1 GO [GFS if 2-D], [softmax layer]
    lstm/weightmatrix.cpp     mode byte; int8 matrix + scales, or float64 matrix [+ training state]
    ccstruct/matrix.h         2-D array: uint32 dim1, uint32 dim2, empty value, dim1*dim2 values

Everything is little-endian. Float weights are stored as float64 whatever TFloat is at run time.
A weight matrix has shape [outputs, inputs + 1]; the last column is the bias.

Only the network is parsed into objects. The bytes after it (the recognizer's tail fields) are kept
verbatim, so writing an unmodified model gives back the original bytes exactly.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TYPE_NAMES = [
    "Invalid", "Input", "Convolve", "Maxpool", "Parallel", "Replicated", "ParBidiLSTM",
    "DepParUDLSTM", "Par2dLSTM", "Series", "Reconfig", "RTLReversed", "TTBReversed", "XYTranspose",
    "LSTM", "SummLSTM", "Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu", "Linear", "Softmax",
    "SoftmaxNoCTC", "LSTMSoftmax", "LSTMBinarySoftmax", "TensorFlow",
]
PLUMBING = {"Parallel", "Replicated", "ParBidiLSTM", "DepParUDLSTM", "Par2dLSTM", "Series",
            "RTLReversed", "TTBReversed", "XYTranspose"}
RECONFIG = {"Reconfig", "Maxpool"}
FULLY_CONNECTED = {"Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu", "Linear", "Softmax",
                   "SoftmaxNoCTC"}
LSTM_TYPES = {"LSTM", "SummLSTM", "LSTMSoftmax", "LSTMBinarySoftmax"}
GATES = ["CI", "GI", "GF1", "GO", "GFS"]  # cell input, input gate, forget gate, output gate, 2-D forget

NF_LAYER_SPECIFIC_LR = 64
K_INT8, K_ADAM, K_DOUBLE = 1, 4, 128
TS_ENABLED = 1

COMPONENTS = [
    "config", "unicharset", "unicharambigs", "inttemp", "pffmtable", "normproto", "punc-dawg",
    "word-dawg", "number-dawg", "freq-dawg", "fixed-length-dawgs", "cube-unicharset",
    "cube-word-dawg", "shapetable", "bigram-dawg", "unambig-dawg", "params-model", "lstm",
    "lstm-punc-dawg", "lstm-word-dawg", "lstm-number-dawg", "lstm-unicharset", "lstm-recoder",
    "version",
]
LSTM_INDEX = COMPONENTS.index("lstm")


# ---------------------------------------------------------------- byte streams

class _Reader:
    def __init__(self, data: bytes):
        self.b, self.p = data, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.p + n > len(self.b):
            raise ValueError(f"read of {n} bytes past end at offset {self.p} of {len(self.b)}")
        out = self.b[self.p:self.p + n]
        self.p += n
        return out

    def num(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> bytes:
        return self.take(self.num("I"))

    def array(self, dtype: str, count: int) -> np.ndarray:
        dt = np.dtype(dtype)
        return np.frombuffer(self.take(count * dt.itemsize), dtype=dt).copy()


class _Writer:
    def __init__(self):
        self.parts: list[bytes] = []

    def raw(self, b: bytes):
        self.parts.append(bytes(b))

    def num(self, fmt: str, v):
        self.parts.append(struct.pack("<" + fmt, v))

    def string(self, b: bytes):
        self.num("I", len(b))
        self.raw(b)

    def array(self, a: np.ndarray, dtype: str):
        self.raw(np.ascontiguousarray(a, dtype=np.dtype(dtype)).tobytes())

    def bytes(self) -> bytes:
        return b"".join(self.parts)


# ---------------------------------------------------------------- 2-D arrays and weight matrices

def _read_2d(r: _Reader, dtype: str):
    d1, d2 = r.num("I"), r.num("I")
    empty = r.num("d" if dtype == "<f8" else "b")
    return r.array(dtype, d1 * d2).reshape(d1, d2), empty


def _write_2d(w: _Writer, a: np.ndarray, empty, dtype: str):
    w.num("I", a.shape[0])
    w.num("I", a.shape[1])
    w.num("d" if dtype == "<f8" else "b", empty)
    w.array(a, dtype)


@dataclass
class WeightMatrix:
    """One Tesseract WeightMatrix. `w` is [outputs, inputs + 1]; the last column is the bias."""
    mode: int
    w: np.ndarray                 # float64, or int8 when is_int
    w_empty: float | int = 0
    scales: np.ndarray | None = None      # int8 models only; per-output scale as stored on disk
    updates: tuple | None = None          # training state, kept for round trip
    dw_sq_sum: tuple | None = None

    @property
    def is_int(self) -> bool:
        return bool(self.mode & K_INT8)

    @property
    def num_outputs(self) -> int:
        return self.w.shape[0]

    def dequantized(self) -> np.ndarray:
        """Float view of the weights. For int8 models the stored per-row scale is max|row| / 127
        (bias included), so a weight is int8 * scale. Scales may be padded to a SIMD multiple of
        the output count (weightmatrix.cpp DeSerialize resizes them before writing back)."""
        if not self.is_int:
            return self.w
        return self.w.astype(np.float64) * self.scales[:self.num_outputs, None]


def _read_wm(r: _Reader, training: bool) -> WeightMatrix:
    mode = r.num("B")
    if not mode & K_DOUBLE:
        raise NotImplementedError("pre-4.0 weight matrix format (no kDoubleFlag)")
    if mode & K_INT8:
        w, empty = _read_2d(r, "i1")
        scales = r.array("<f8", r.num("I"))
        return WeightMatrix(mode, w, empty, scales)
    w, empty = _read_2d(r, "<f8")
    wm = WeightMatrix(mode, w, empty)
    if training:
        wm.updates = _read_2d(r, "<f8")
        if mode & K_ADAM:
            wm.dw_sq_sum = _read_2d(r, "<f8")
    return wm


def _write_wm(out: _Writer, wm: WeightMatrix, training: bool):
    out.num("B", wm.mode)
    if wm.is_int:
        _write_2d(out, wm.w, wm.w_empty, "i1")
        out.num("I", len(wm.scales))
        out.array(wm.scales, "<f8")
        return
    _write_2d(out, wm.w, wm.w_empty, "<f8")
    if training:
        _write_2d(out, *wm.updates, "<f8")
        if wm.mode & K_ADAM:
            _write_2d(out, *wm.dw_sq_sum, "<f8")


# ---------------------------------------------------------------- layers

def _ceil_log2(n: int) -> int:
    k = 0
    while (1 << k) < n:
        k += 1
    return k


@dataclass
class Layer:
    type: str
    name: str
    ni: int
    no: int
    num_weights: int
    training: int          # raw byte; the layer serializes training state only when == 1
    needs_backprop: int
    flags: int
    type_byte: int = 0     # 0 = type written as a name string (current format); else legacy index
    children: list = field(default_factory=list)
    learning_rates: np.ndarray | None = None
    params: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)   # FC: {"w": ...}; LSTM: {"CI": ..., "GI": ..., ...}
    softmax: "Layer | None" = None

    @property
    def is_training(self) -> bool:
        return self.training == TS_ENABLED

    def walk(self, path: str = ""):
        """Yield (path, layer) for this layer and everything inside it."""
        here = f"{path}/{self.name}" if path else self.name
        yield here, self
        for c in self.children:
            yield from c.walk(here)
        if self.softmax is not None:
            yield from self.softmax.walk(here)


def _read_layer(r: _Reader) -> Layer:
    tb = r.num("B")
    tname = r.string().decode("ascii") if tb == 0 else TYPE_NAMES[tb]
    if tname not in TYPE_NAMES:
        raise ValueError(f"unknown layer type {tname!r}")
    L = Layer(type=tname, type_byte=tb, training=r.num("b"), needs_backprop=r.num("b"),
              flags=r.num("i"), ni=r.num("i"), no=r.num("i"), num_weights=r.num("i"),
              name=r.string().decode("utf-8", "surrogateescape"))
    if tname in PLUMBING:
        L.children = [_read_layer(r) for _ in range(r.num("I"))]
        if L.flags & NF_LAYER_SPECIFIC_LR:
            L.learning_rates = r.array("<f4", r.num("I"))
    elif tname in RECONFIG:
        L.params = {"x_scale": r.num("i"), "y_scale": r.num("i")}
    elif tname == "Convolve":
        L.params = {"half_x": r.num("i"), "half_y": r.num("i")}
    elif tname == "Input":
        L.params = {k: r.num("i") for k in ("batch", "height", "width", "depth", "loss_type")}
    elif tname in FULLY_CONNECTED:
        L.weights["w"] = _read_wm(r, L.is_training)
    elif tname in LSTM_TYPES:
        na = r.num("i")
        nf = L.no if tname == "LSTMSoftmax" else _ceil_log2(L.no) if tname == "LSTMBinarySoftmax" else 0
        is_2d = False
        for g in GATES:
            if g == "GFS" and not is_2d:
                continue
            L.weights[g] = _read_wm(r, L.is_training)
            if g == "CI":
                ns = L.weights["CI"].num_outputs
                is_2d = na - nf == L.ni + 2 * ns
        L.params = {"na": na, "ns": L.weights["CI"].num_outputs, "is_2d": is_2d}
        if tname in ("LSTMSoftmax", "LSTMBinarySoftmax"):
            L.softmax = _read_layer(r)
    else:
        raise NotImplementedError(f"layer type {tname!r} is not supported")
    return L


def _write_layer(out: _Writer, L: Layer):
    out.num("B", L.type_byte)
    if L.type_byte == 0:
        out.string(L.type.encode("ascii"))
    out.num("b", L.training)
    out.num("b", L.needs_backprop)
    out.num("i", L.flags)
    out.num("i", L.ni)
    out.num("i", L.no)
    out.num("i", L.num_weights)
    out.string(L.name.encode("utf-8", "surrogateescape"))
    if L.type in PLUMBING:
        out.num("I", len(L.children))
        for c in L.children:
            _write_layer(out, c)
        if L.flags & NF_LAYER_SPECIFIC_LR:
            out.num("I", len(L.learning_rates))
            out.array(L.learning_rates, "<f4")
    elif L.type in RECONFIG:
        out.num("i", L.params["x_scale"])
        out.num("i", L.params["y_scale"])
    elif L.type == "Convolve":
        out.num("i", L.params["half_x"])
        out.num("i", L.params["half_y"])
    elif L.type == "Input":
        for k in ("batch", "height", "width", "depth", "loss_type"):
            out.num("i", L.params[k])
    elif L.type in FULLY_CONNECTED:
        _write_wm(out, L.weights["w"], L.is_training)
    elif L.type in LSTM_TYPES:
        out.num("i", L.params["na"])
        for g in GATES:
            if g in L.weights:
                _write_wm(out, L.weights[g], L.is_training)
        if L.softmax is not None:
            _write_layer(out, L.softmax)


# ---------------------------------------------------------------- the 'lstm' component

@dataclass
class LstmModel:
    net: Layer
    tail: bytes   # recognizer fields after the network, verbatim

    @classmethod
    def from_bytes(cls, data: bytes) -> "LstmModel":
        r = _Reader(data)
        net = _read_layer(r)
        return cls(net, data[r.p:])

    def to_bytes(self) -> bytes:
        w = _Writer()
        _write_layer(w, self.net)
        w.raw(self.tail)
        return w.bytes()

    def tensors(self) -> dict[str, np.ndarray]:
        """Name -> weight array. Arrays are the live objects: edit them in place, then save."""
        out = {}
        for path, L in self.net.walk():
            for g, wm in L.weights.items():
                key = f"{path}:{g}"
                if key in out:
                    raise ValueError(f"duplicate tensor path {key}; layer names are not unique")
                out[key] = wm.w
        return out

    def weight_count(self) -> int:
        return sum(a.size for a in self.tensors().values())

    def tail_fields(self) -> dict:
        """Decode the tail when it holds no charsets (true when the traineddata has lstm-unicharset
        and lstm-recoder). Field types follow lstmrecognizer.h; returns {} if the length is off."""
        r = _Reader(self.tail)
        try:
            spec = r.string().decode("utf-8", "replace")
            f = {"network_spec": spec, "training_flags": r.num("i"), "training_iteration": r.num("i"),
                 "sample_iteration": r.num("i"), "null_char": r.num("i"), "adam_beta": r.num("f"),
                 "learning_rate": r.num("f"), "momentum": r.num("f")}
        except ValueError:
            return {}
        return f if r.p == len(self.tail) else {}

    def summary(self) -> str:
        lines = []
        for path, L in self.net.walk():
            depth = path.count("/")
            shapes = ", ".join(f"{g}{tuple(wm.w.shape)}{' int8' if wm.is_int else ''}"
                               for g, wm in L.weights.items())
            extra = " ".join(f"{k}={v}" for k, v in L.params.items())
            lines.append(f"{'  ' * depth}{L.type:<12} {L.name:<28} ni={L.ni:<4} no={L.no:<4} "
                         f"{extra} {shapes}".rstrip())
        return "\n".join(lines)


# ---------------------------------------------------------------- the .traineddata container

@dataclass
class Traineddata:
    components: dict[int, bytes]
    num_entries: int = len(COMPONENTS)

    @classmethod
    def read(cls, path: str | Path) -> "Traineddata":
        data = Path(path).read_bytes()
        r = _Reader(data)
        n = r.num("i")
        offsets = [r.num("q") for _ in range(n)]
        comps = {}
        for i, off in enumerate(offsets):
            if off < 0:
                continue
            later = [o for o in offsets[i + 1:] if o >= 0]
            end = later[0] if later else len(data)
            comps[i] = data[off:end]
        return cls(comps, n)

    def to_bytes(self) -> bytes:
        w = _Writer()
        w.num("i", self.num_entries)
        off = 4 + 8 * self.num_entries
        for i in range(self.num_entries):
            if i in self.components:
                w.num("q", off)
                off += len(self.components[i])
            else:
                w.num("q", -1)
        for i in range(self.num_entries):
            if i in self.components:
                w.raw(self.components[i])
        return w.bytes()

    def write(self, path: str | Path):
        Path(path).write_bytes(self.to_bytes())

    @property
    def lstm(self) -> LstmModel:
        return LstmModel.from_bytes(self.components[LSTM_INDEX])

    def with_lstm(self, model: LstmModel) -> "Traineddata":
        comps = dict(self.components)
        comps[LSTM_INDEX] = model.to_bytes()
        return Traineddata(comps, self.num_entries)

    def directory(self) -> str:
        return "\n".join(f"{i:2} {COMPONENTS[i] if i < len(COMPONENTS) else '?':<20} {len(b):>9} bytes"
                         for i, b in sorted(self.components.items()))

    def unichars(self) -> list[str]:
        """The lstm-unicharset entries by unichar id (id 0 is NULL, then Joined, |Broken|0|1, ...)."""
        text = self.components[COMPONENTS.index("lstm-unicharset")].decode("utf-8")
        lines = text.splitlines()
        return [ln.split(" ", 1)[0] for ln in lines[1:1 + int(lines[0])]]

    def recoder(self) -> list[tuple[int, ...]]:
        """Code sequence per unichar id (UnicharCompress: uint32 count, then per entry
        int8 self_normalized, int32 length, int32 code[length])."""
        r = _Reader(self.components[COMPONENTS.index("lstm-recoder")])
        out = []
        for _ in range(r.num("I")):
            r.num("b")
            n = r.num("i")
            out.append(tuple(r.num("i") for _ in range(n)))
        return out

    def output_labels(self) -> dict[int, str]:
        """Output-layer row -> character, for unichars encoded as a single code. Unichar 0 is
        written "NULL" in the unicharset file but is the space character (UNICHAR_SPACE)."""
        return {codes[0]: (" " if i == 0 else ch)
                for i, (ch, codes) in enumerate(zip(self.unichars(), self.recoder())) if len(codes) == 1}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("info", "roundtrip"):
        print("usage: python3 -I tesslstm.py info|roundtrip FILE.traineddata", file=sys.stderr)
        return 2
    path = argv[2]
    raw = Path(path).read_bytes()
    td = Traineddata.read(path)
    m = td.lstm
    if argv[1] == "info":
        print(td.directory())
        print()
        print(m.summary())
        print()
        print(f"weights: {m.weight_count():,}  (root header num_weights={m.net.num_weights:,})")
        for k, v in m.tail_fields().items():
            print(f"{k}: {v}")
        return 0
    lstm_ok = m.to_bytes() == td.components[LSTM_INDEX]
    file_ok = td.with_lstm(m).to_bytes() == raw
    print(f"lstm component round trip: {'identical' if lstm_ok else 'DIFFERENT'}")
    print(f"whole file round trip:     {'identical' if file_ok else 'DIFFERENT'}")
    return 0 if lstm_ok and file_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
