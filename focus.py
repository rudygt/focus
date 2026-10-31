# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe>=0.10.14,<0.10.22", "scikit-learn>=1.4"]
# ///
"""Keep keyboard focus on the monitor you are looking at, without face detection.

Each frame is reduced right after capture to a person silhouette, a 32x24 grayscale
thumbnail with the background masked out, and a HEAD_PX square crop around the head cut
from that masked image; the camera image never leaves Camera.observe(). A logistic
regression trained on your own observations decides left or right. focus.ps1 activates
the top window on that monitor.

    uv run focus.py --calibrate              # look left, right, left, right; Enter when your head is there
    uv run focus.py --retrain --feature rows # refit from calibration.npz with another feature set
    uv run focus.py --show --log live.npz    # run; the preview shows only silhouette and thumbnail
"""
import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "model.joblib"
CALIB_DATA = HERE / "calibration.npz"  # the observations the model was trained on, for retraining and diagnosis
PS1_WIN = subprocess.check_output(["wslpath", "-w", str(HERE / "focus.ps1")], text=True).strip()
POWERSHELL = shutil.which("powershell.exe") or r"/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

CONFIDENCE = 0.75  # below this probability the gaze is treated as undecided and focus stays put
SMOOTHING = 0.5  # weight of the previous decision's probability; one odd frame cannot flip focus
DECISION_INTERVAL = 0.2  # seconds; the camera delivers 5 fps, so this decides on every frame
LOG_EVERY = 10.0  # seconds between log file rewrites
CALIB_BLOCKS = ("left", "right", "left", "right")
CALIB_BLOCK_SECONDS = 8
DEFAULT_FEATURE = "head"
HEAD_PX = 48  # side of the head crop; more pixels means more gaze signal and less privacy. 48 is near native and recognisable.


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class Observation:
    """Everything kept from one frame. None of it shows a face, text, or the room."""

    mask: np.ndarray  # 64x48 bool person silhouette
    mask_hi: np.ndarray  # 160x120 bool person silhouette
    thumb: np.ndarray  # 32x24 uint8 grayscale, background zeroed by the silhouette
    head: np.ndarray  # HEAD_PX x HEAD_PX uint8 grayscale crop centred on the head, background zeroed

    FIELDS = ("mask", "mask_hi", "thumb", "head")


class Camera:
    def __init__(self, index: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"cannot open camera {index} (is the webcam attached to WSL? usbipd attach --wsl --busid 5-8)")
        # Over usbip, raw YUYV crawls at ~4 fps and MJPG above 5 fps arrives with corrupt frames.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def observe(self) -> Observation:
        """Read one frame and reduce it. The frame is dropped here."""
        ok, frame = self.cap.read()
        if not ok:
            sys.exit("camera read failed (device detached?)")
        person = self.seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask > 0.5
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) * person
        return Observation(
            mask=cv2.resize(person.astype(np.float32), (64, 48), interpolation=cv2.INTER_AREA) > 0.5,
            mask_hi=cv2.resize(person.astype(np.float32), (160, 120), interpolation=cv2.INTER_AREA) > 0.5,
            thumb=cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA),
            head=head_crop(person, gray),
        )


def head_box(person: np.ndarray) -> tuple[int, int, int] | None:
    """(centre x, centre y, side) of a square over the upper head in a full-res silhouette, or None without a person.

    Everything is measured on the crown, the rows just below the hairline, so a hand at the chin or a
    shoulder cannot move or widen the box. The square runs from the hairline to about the upper lip.
    """
    rows = np.flatnonzero(person.any(axis=1))
    if len(rows) < 20:
        return None
    top = rows[0]
    crown = person[top + 4 : top + 24]
    widths = crown.sum(axis=1)
    if not (widths > 0).any():
        return None
    head_w = float(np.median(widths[widths > 0]))
    if head_w < 8:
        return None
    xs = np.nonzero(crown)[1]
    return int(xs.mean()), int(top + 0.5 * head_w), int(head_w)


def head_crop(person: np.ndarray, gray: np.ndarray) -> np.ndarray:
    box = head_box(person)
    if box is None:
        return np.zeros((HEAD_PX, HEAD_PX), np.uint8)
    cx, cy, side = box
    half = side // 2
    padded = np.pad(gray, half)  # so a head at the frame edge still yields a full square
    crop = padded[cy : cy + 2 * half, cx : cx + 2 * half]
    return cv2.resize(crop, (HEAD_PX, HEAD_PX), interpolation=cv2.INTER_AREA)


def posture(mask: np.ndarray) -> tuple[int, float] | None:
    """Head top row and body centre x of a silhouette, or None when there is no person."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 20:
        return None
    top = ys.min()
    body = xs[ys >= top + (ys.max() - top) * 0.4]
    return int(top), float(body.mean()) if len(body) else float(xs.mean())


def feat_rows(o: Observation) -> np.ndarray:
    """Width and centre of each silhouette row, by absolute row."""
    out = np.zeros(2 * o.mask.shape[0])
    for i, row in enumerate(o.mask):
        xs = np.nonzero(row)[0]
        if len(xs):
            out[2 * i], out[2 * i + 1] = len(xs), xs.mean()
    return out


def feat_aligned(o: Observation) -> np.ndarray:
    """Width and centre of each row of the high-res silhouette, rows counted from the head top and x from the body centre."""
    rows = 48
    out = np.zeros(2 * rows)
    p = posture(o.mask_hi)
    if p is None:
        return out
    top, bx = p
    for i in range(rows):
        if top + i >= o.mask_hi.shape[0]:
            break
        xs = np.nonzero(o.mask_hi[top + i])[0]
        if len(xs):
            out[2 * i], out[2 * i + 1] = len(xs), xs.mean() - bx
    return out


def feat_thumb(o: Observation) -> np.ndarray:
    return o.thumb.reshape(-1).astype(float)


def feat_head(o: Observation) -> np.ndarray:
    """Head crop with its own brightness normalised away, so lighting changes do not move the decision."""
    x = o.head.reshape(-1).astype(float)
    return (x - x.mean()) / (x.std() + 1e-6)


FEATURES = {"rows": feat_rows, "aligned": feat_aligned, "thumb": feat_thumb, "head": feat_head}


class Classifier:
    def __init__(self, feature: str, pipeline):
        self.feature, self.pipeline = feature, pipeline

    @classmethod
    def fit(cls, feature: str, observations: list[Observation], labels: np.ndarray) -> "Classifier":
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.1))
        pipeline.fit(np.array([FEATURES[feature](o) for o in observations]), labels)
        return cls(feature, pipeline)

    @classmethod
    def load(cls) -> "Classifier":
        if not MODEL_FILE.exists():
            sys.exit(f"no {MODEL_FILE.name}; run with --calibrate first")
        saved = joblib.load(MODEL_FILE)
        if saved.get("head_px") != HEAD_PX:
            sys.exit(f"{MODEL_FILE.name} was trained with a {saved.get('head_px')}px head crop, this build uses {HEAD_PX}px; run --calibrate again")
        return cls(saved["feature"], saved["pipeline"])

    def save(self) -> None:
        joblib.dump({"feature": self.feature, "pipeline": self.pipeline, "head_px": HEAD_PX}, MODEL_FILE)

    def p_right(self, o: Observation) -> float:
        return float(self.pipeline.predict_proba(FEATURES[self.feature](o)[None, :])[0, 1])

    @staticmethod
    def side(p: float) -> Side | None:
        if p >= CONFIDENCE:
            return Side.RIGHT
        if p <= 1 - CONFIDENCE:
            return Side.LEFT
        return None


class FocusServer:
    """One PowerShell process kept alive for the session; each switch is one line in, one line out."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        cmd = [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", PS1_WIN, "-Serve"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if self._readline() != "ready":
            sys.exit("focus.ps1 did not start; run it with -Side list to see the error")

    def _readline(self) -> str:
        return self.proc.stdout.readline().decode("utf-8", errors="replace").strip()

    def switch(self, side: Side) -> str:
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        self.proc.stdin.write(f"{side.value}\n".encode())
        self.proc.stdin.flush()
        reply = self._readline()
        if not reply:
            self.proc = None
            return "focus.ps1 exited; restarting on next switch"
        return reply


def save_observations(path: Path, observations: list[Observation], **extra) -> None:
    arrays = {f: np.array([getattr(o, f) for o in observations]) for f in Observation.FIELDS}
    np.savez_compressed(path, **arrays, **extra)


def load_observations(path: Path) -> tuple[list[Observation], dict]:
    d = np.load(path)
    missing = [f for f in Observation.FIELDS if f not in d.files]
    if missing:
        sys.exit(f"{path.name} predates the {', '.join(missing)} field(s); run --calibrate again")
    if d["head"].shape[1:] != (HEAD_PX, HEAD_PX):
        sys.exit(f"{path.name} holds {d['head'].shape[1]}px head crops, this build uses {HEAD_PX}px; run --calibrate again")
    n = len(d["mask"])
    observations = [Observation(**{f: d[f][i] for f in Observation.FIELDS}) for i in range(n)]
    return observations, {k: d[k] for k in d.files if k not in Observation.FIELDS}


def train(feature: str, observations: list[Observation], labels: np.ndarray, blocks: np.ndarray) -> None:
    """Score every feature set on the second left/right pair after training on the first, then fit the chosen one on all."""
    first = blocks < 2
    print(f"held-out accuracy on the second pair of blocks ({(~first).sum()} observations):")
    for name in FEATURES:
        held = Classifier.fit(name, [o for o, f in zip(observations, first) if f], labels[first])
        acc = np.mean([(held.p_right(o) > 0.5) == y for o, y, f in zip(observations, labels, first) if not f])
        print(f"  {name:8s} {acc:4.0%}{'   <- chosen' if name == feature else ''}")
    Classifier.fit(feature, observations, labels).save()
    print(f"saved {MODEL_FILE.name} ({feature}) trained on {len(observations)} observations")


def calibrate(cam: Camera, feature: str) -> None:
    print(f"calibration: {len(CALIB_BLOCKS)} blocks, order {' '.join(CALIB_BLOCKS)}, {CALIB_BLOCK_SECONDS}s of sampling each", flush=True)
    observations, labels, blocks = [], [], []
    for i, block in enumerate(CALIB_BLOCKS):
        input(f"  look at your {block.upper()} monitor, press Enter, then keep looking there while you vary posture: lean, sit up, rest your chin on a hand... ")
        for _ in range(3):
            cam.observe()  # drop frames buffered while waiting for Enter
        end = time.monotonic() + CALIB_BLOCK_SECONDS
        n = 0
        while time.monotonic() < end:
            observations.append(cam.observe())
            labels.append(int(block == "right"))
            blocks.append(i)
            n += 1
        print(f"    {n} observations", flush=True)
    labels, blocks = np.array(labels), np.array(blocks)
    save_observations(CALIB_DATA, observations, label=labels, block=blocks)
    train(feature, observations, labels, blocks)


def retrain(feature: str) -> None:
    if not CALIB_DATA.exists():
        sys.exit(f"no {CALIB_DATA.name}; run with --calibrate first")
    observations, extra = load_observations(CALIB_DATA)
    train(feature, observations, extra["label"], extra["block"])


def preview(o: Observation, label: str) -> np.ndarray:
    scale = 6
    outline = cv2.Canny(cv2.resize(o.mask.astype(np.uint8) * 255, (64 * scale, 48 * scale), interpolation=cv2.INTER_NEAREST), 50, 150)
    thumb = cv2.resize(o.thumb, (32 * scale * 3 // 2, 24 * scale * 3 // 2), interpolation=cv2.INTER_NEAREST)
    head = cv2.resize(o.head, (48 * scale, 48 * scale), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((48 * scale, 64 * scale + thumb.shape[1] + head.shape[1], 3), np.uint8)
    canvas[:, : 64 * scale] = cv2.cvtColor(outline, cv2.COLOR_GRAY2BGR)
    canvas[: thumb.shape[0], 64 * scale : 64 * scale + thumb.shape[1]] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
    canvas[:, 64 * scale + thumb.shape[1] :] = cv2.cvtColor(head, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return canvas


def run(cam: Camera, clf: Classifier, interval: float, confirm: int, show: bool, log: Path | None) -> None:
    server = FocusServer()
    server.start()
    current: Side | None = None
    streak: tuple[Side | None, int] = (None, 0)
    last = 0.0
    p = 0.5
    logged: list[Observation] = []
    raws: list[float] = []
    log_written = time.monotonic()
    try:
        while True:
            o = cam.observe()  # read continuously so the driver buffer never goes stale
            now = time.monotonic()
            if now - last < interval:
                continue
            last = now

            if not o.head.any():  # nobody in front of the camera: hold focus, decay towards undecided
                raw = 0.5
            else:
                raw = clf.p_right(o)
            p = SMOOTHING * p + (1 - SMOOTHING) * raw
            want = clf.side(p)
            if log is not None:
                logged.append(o)
                raws.append(raw)
                if now - log_written > LOG_EVERY:
                    save_observations(log, logged, raw=np.array(raws))
                    log_written = now
            streak = (want, streak[1] + 1) if want == streak[0] else (want, 1)

            note = ""
            if want is not None and want != current and streak[1] >= confirm:
                note = server.switch(want)
                current = want
            label = f"p(right)={p:.2f} gaze={want.value if want else '-'}"
            print(f"{label:32} {note}", flush=True)

            if show:
                cv2.imshow("focus (silhouette, masked thumbnail, head crop)", preview(o, label))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return
    finally:
        if log is not None and logged:
            save_observations(log, logged, raw=np.array(raws))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="V4L2 device index (default 0)")
    ap.add_argument("--interval", type=float, default=DECISION_INTERVAL, help=f"seconds between decisions (default {DECISION_INTERVAL})")
    ap.add_argument("--confirm", type=int, default=1, help="consecutive agreeing decisions before switching (default 1)")
    ap.add_argument("--calibrate", action="store_true", help="record observations for each monitor, train, and exit")
    ap.add_argument("--retrain", action="store_true", help="refit the model from calibration.npz and exit")
    ap.add_argument("--feature", choices=FEATURES, default=DEFAULT_FEATURE, help=f"feature set to train on (default {DEFAULT_FEATURE})")
    ap.add_argument("--show", action="store_true", help="preview window with silhouette outline and thumbnail (q quits)")
    ap.add_argument("--log", type=Path, help="save each decision's observation and probability to this .npz for diagnosis")
    args = ap.parse_args()

    if args.retrain:
        retrain(args.feature)
        return
    cam = Camera(args.camera)
    if args.calibrate:
        calibrate(cam, args.feature)
        return
    try:
        run(cam, Classifier.load(), args.interval, args.confirm, args.show, args.log)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
