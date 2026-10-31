# focus

Keeps keyboard focus on the monitor you are looking at, without face detection.

A webcam frame is reduced the moment it is read to a person silhouette, a 32x24
grayscale thumbnail with the background masked out, and a square crop of the upper head
cut from that masked image. A logistic regression trained on your own observations picks
left or right, and a PowerShell script activates the top window on that monitor.

Built for a two-monitor Windows desktop with the vision side running in WSL2. The Windows
side is PowerShell only; nothing is installed on Windows beyond usbipd-win.

## Requirements

- Windows 10/11 with two monitors arranged left and right in display settings.
- WSL2 with a kernel that ships `uvcvideo` (Microsoft's 6.x kernels do; check with
  `zcat /proc/config.gz | grep USB_VIDEO_CLASS`).
- [usbipd-win](https://github.com/dorssel/usbipd-win) to pass the webcam to WSL.
- [uv](https://docs.astral.sh/uv/) inside WSL. It installs Python and the dependencies
  from the script header on first run.

## Files

- `focus.py` runs in WSL. Camera capture, reduction, classifier, decision loop.
- `focus.ps1` runs on Windows. Lists monitors and windows, activates one. `focus.py`
  keeps one instance alive in `-Serve` mode and sends it a line per switch, which takes
  about 0.1 s instead of 0.4 s for a fresh PowerShell each time.
- `model.joblib` and `calibration.npz` are written by `--calibrate` and ignored by git.
  The npz holds the observations the model was trained on so `--retrain --feature <name>`
  can refit without the camera. `--log file.npz` records what the running tool saw.
- `experiment.py` is the evidence behind the design: it records silhouettes, thumbnails
  and face-mesh yaw together and scores how well each anonymous representation predicts
  left or right.

## One-time setup

Find the webcam's bus id with `usbipd list`, then from an elevated Windows prompt:

    usbipd bind --force --busid <busid>

`--force` swaps in the usbipd stub driver so Windows apps release the device. Software
that keeps the camera open at all times, Logi Tune for example, vetoes the attach, so
close it first. While the camera is attached to WSL, Windows apps cannot use it.
`usbipd unbind --busid <busid>` gives it back.

## Every session

    usbipd attach --wsl --busid <busid>  # from Windows, after each reboot or WSL restart
    uv run focus.py --calibrate          # left, right, left, right; Enter when your head is there
    uv run focus.py --show               # q quits the preview; Ctrl-C stops the loop
    uv run focus.py                      # background mode, no preview

Calibration asks you to look at one monitor, press Enter, and keep looking there for a
few seconds while varying posture: lean, sit up, rest your chin on a hand. The model only
knows positions it has seen. It prints, for each feature set, the accuracy on the second
left/right pair after training on the first. Recalibrate when the camera moves, the
lighting changes a lot, or you sit differently.

Feature sets: `rows` and `aligned` use the silhouette only, `thumb` the masked
thumbnail, `head` the head crop. The default is `head`. Its square is placed and sized
from the crown of the silhouette alone, the rows just below the hairline, and runs down
to about the upper lip, so sitting higher, lower or closer does not move the input and a
hand at the chin or a shoulder cannot enter it. Its brightness is normalised per frame.
Pick another feature with `--feature` if it scores better.

Flags: `--interval` seconds between decisions (0.2, every camera frame), `--confirm`
agreeing decisions before switching (1), `--camera` V4L2 index (0). The probability is
smoothed across decisions, so a switch needs two agreeing frames, about half a second,
plus 0.1 s for the window activation. Nobody in frame holds focus where it is.

Check what the Windows side sees without switching anything:

    powershell.exe -NoProfile -ExecutionPolicy Bypass -File focus.ps1 -Side list

## Privacy design

The camera image exists only inside `Camera.observe()`, at 320x240, for the few
milliseconds MediaPipe selfie segmentation needs to produce a person mask. Everything
downstream, including the preview window, the classifier, the model file and the logs,
sees three things: a binary person silhouette (64x48 and 160x120), a 32x24 grayscale
thumbnail in which every background pixel is zero, and a `HEAD_PX` square of the upper
head cut from that masked image. None of it carries text or the room.

The head crop is the privacy knob: it holds the most gaze signal and the most pixels of
you. At 12x12 it is a blob, at 16x16 a coarse patch of hair and skin tone, and at 48x48,
the current default, close to the camera's own resolution of the head, so the patch shows
eyes and nose and is recognisable. Change `HEAD_PX` in `focus.py` after looking at what
the preview shows, and recalibrate after any change; the model records the crop size it
was trained with and refuses to run on another.

Why not silhouette alone: with the head turned naturally rather than exaggerated, the
silhouette moved under one pixel at 64x48 between monitors, too little to be reliable.
The outline is kept for the preview and as a feature option.

What this does not protect against: a process that can read this process's memory, or
the camera itself.

## Measured

Over USB/IP the camera delivers about 4 fps in raw YUYV and 15 fps in MJPG, but MJPG above
5 fps arrives with corrupt frames, so the tool requests MJPG at 5 fps. Running costs about
5% of one core and 340 MB resident, flat over time.
