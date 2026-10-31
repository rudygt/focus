# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["mediapipe>=0.10.14,<0.10.22", "scikit-learn>=1.4"]
# ///
"""Which privacy-preserving representation still separates left from right?

    uv run experiment.py record OUT.npz [SECONDS]   # alternate looking left/right while it runs
    uv run experiment.py analyze OUT.npz

Per frame it stores three things and the face-mesh yaw ratio as ground truth:
  silhouette  64x48 binary person mask (selfie segmentation), no interior detail
  thumb       24x18 grayscale blob of the whole frame
  outline     64x48 edge map of the silhouette only
"""
import sys, time
import cv2, numpy as np
import mediapipe as mp

def record(path, seconds):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(3, 320); cap.set(4, 240); cap.set(5, 5)
    mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
    ratios, sil, thumb, ts = [], [], [], []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        ok, f = cap.read()
        if not ok: continue
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        r = mesh.process(rgb)
        if not r.multi_face_landmarks: continue
        lm = r.multi_face_landmarks[0].landmark; xs = (lm[234].x, lm[454].x)
        ratios.append((lm[1].x - min(xs)) / (max(xs) - min(xs)))
        mask = seg.process(rgb).segmentation_mask > 0.5
        sil.append(cv2.resize(mask.astype(np.uint8), (64, 48), interpolation=cv2.INTER_AREA) > 0)
        thumb.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (24, 18), interpolation=cv2.INTER_AREA))
        ts.append(time.monotonic())
    np.savez_compressed(path, ratio=np.array(ratios), sil=np.array(sil), thumb=np.array(thumb), t=np.array(ts))
    print(f"recorded {len(ratios)} frames with a face over {seconds}s -> {path}")

def analyze(path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = np.load(path); ratio = d["ratio"]
    lo, hi = np.percentile(ratio, [10, 90]); mid = (lo + hi) / 2; band = (hi - lo) * 0.2
    keep = np.abs(ratio - mid) > band
    y = (ratio[keep] > mid).astype(int)
    print(f"{len(ratio)} frames, ratio 10th/90th pct = {lo:.2f}/{hi:.2f}, {keep.sum()} outside dead band, class balance {y.mean():.2f}")
    if hi - lo < 0.15: print("WARNING: head barely turned during recording; labels are unreliable"); 
    sil = d["sil"][keep]
    outline = np.array([cv2.Canny(m.astype(np.uint8) * 255, 50, 150) > 0 for m in sil])
    reps = {
        "silhouette 64x48": sil.reshape(len(sil), -1).astype(float),
        "outline 64x48": outline.reshape(len(outline), -1).astype(float),
        "thumbnail 24x18": d["thumb"][keep].reshape(keep.sum(), -1).astype(float),
    }
    # Silhouette-only handcrafted cue: head centroid vs shoulder centroid.
    def head_vs_shoulders(m):
        ys, xs = np.nonzero(m)
        if len(ys) < 20: return [0, 0]
        top, bot = ys.min(), ys.max(); cut = top + (bot - top) * 0.4
        head, body = xs[ys < cut], xs[ys >= cut]
        return [head.mean() - body.mean() if len(head) and len(body) else 0, head.std() if len(head) else 0]
    reps["head-vs-shoulders (2 numbers)"] = np.array([head_vs_shoulders(m) for m in sil])
    print("timeline (1 char/sec):", "".join("L" if v > mid + band else "R" if v < mid - band else "." for v in ratio[::5]))
    # Alternate 5-second blocks between train and test so neighbouring frames cannot leak, while both halves see both classes.
    test = (np.arange(len(y)) // 25) % 2 == 1
    if len(set(y[~test])) < 2 or len(set(y[test])) < 2:
        print("one class only in train or test; recording needs more head turning"); return
    for name, X in reps.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
        clf.fit(X[~test], y[~test])
        acc = (clf.predict(X[test]) == y[test]).mean()
        print(f"{name:32s} held-out accuracy {acc:.0%}")
    print(f"{'face mesh ratio (reference)':32s} held-out accuracy 100% by construction")

def dump(path):
    d = np.load(path); i = len(d["ratio"]) // 2
    cv2.imwrite(path.replace(".npz", "_sil.png"), cv2.resize(d["sil"][i].astype(np.uint8) * 255, (256, 192), interpolation=cv2.INTER_NEAREST))
    cv2.imwrite(path.replace(".npz", "_thumb.png"), cv2.resize(d["thumb"][i], (240, 180), interpolation=cv2.INTER_NEAREST))

if __name__ == "__main__":
    cmd, path = sys.argv[1], sys.argv[2]
    if cmd == "record": record(path, float(sys.argv[3]) if len(sys.argv) > 3 else 45)
    elif cmd == "analyze": analyze(path)
    elif cmd == "dump": dump(path)
