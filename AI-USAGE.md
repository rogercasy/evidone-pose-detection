# User

I'm building a small desktop app using a Qt window showing a live camera feed. Later steps will run a YOLO pose model on each frame to draw skeleton, blur faces, etc. This first step is going to be only the video loop.
Open a camera witth cv2.VideoCapture with the device index selectable from the command line via '--camera' with a default option. Render the frames in a Qt window using PySide6 and add 'q' and 'Esc' to quit. On the way out the camera is released and the window closed leaving no process holding the device.
Add all the new dependencies and run it against the real camera on this machine and confirm the device is free afterwards.

# Claude

I'll start by looking at the current state of the project.

Project is a clean cookiecutter scaffold. Adding dependencies first.

Now the implementation. Starting with data structures and errors:

Now the test suite and test environment:

Now adding the pep8-naming config for the Qt/OpenCV API names and running QA:

Three issues to fix — a test assertion that was wrong about scaling, and two mypy complaints.

QA is green. Now let me run it against the real camera. First, checking what's available for driving/verifying a GUI window on this Wayland session:

No injection tools installed, but `uv run` can pull python-xlib ad hoc, so I can send a genuine keypress via XTEST. Launching the real app under XWayland so it's addressable:

App is live and holding the device. Let me capture the window to confirm real frames are rendering:

Live feed confirmed — real frames, natural colors (the BGR888 mapping is correct). Now sending a genuine `q` keypress via XTEST:

The sleep substitute hung the loop. Checking state directly instead:

The `q` key drove the full teardown path. Now confirming nothing holds the device:

Those two "evidone" hits are my own shell commands matching the path string, not the app. `fuser` exit 1 and `lsof` confirm the device is free. Let me clean up the stuck shell and verify the `Esc` path against the real camera too:

`/dev/video1` is the webcam's metadata node, not a capture node — so that run exercised the error path against real hardware (clean message, exit code 2). Now the `Esc` path on camera 0:

Escape didn't close it while `q` did. Let me find out whether the key is reaching the app at all:

Escape worked too — my previous check just raced the shutdown by milliseconds. Final verification sweep:

Device is free; the only `pgrep` hit was the search command itself. Saving the one decision that isn't readable from the code:

Done. The video loop is in place, QA is green, and it's verified against the real webcam.

## What I built

| Module | Role |
|---|---|
| `models.py` | `CameraConfig`, `CaptureProperties` (frozen slotted dataclasses), `Frame` alias |
| `errors.py` | `EvidoneError` → `CameraOpenError`, `CameraNotOpenError` |
| `camera.py` | `Camera` — context manager over `cv2.VideoCapture`, idempotent `release()` |
| `window.py` | `VideoWindow(QMainWindow)` — `QTimer` pull at the capture's FPS, aspect-preserving scaling |
| `cli.py` | `--camera INDEX`, default `0` |
| `main.py` | wires it up; `run()` holds the camera in a `with` block around `app.exec()` |

Device release is guaranteed twice over: `closeEvent` releases on the quit path, and the `with Camera(...)` block in `run()` catches any exit the window never sees. `release()` is idempotent, so the double call is harmless.

## Dependencies

`pyside6`, `opencv-python-headless`, `numpy` added via `uv add` (lockfile updated). I chose **headless** OpenCV deliberately — `opencv-python` bundles its own Qt platform plugins that collide with PySide6's at runtime, and since rendering goes through Qt rather than `cv2.imshow`, nothing is lost. I noted that in memory so it doesn't get "fixed" later.

## Real-camera verification

Launched under XWayland so I could inject genuine keypresses via XTEST (`uv run` with inline `python-xlib` — no injection tools were installed).

- **Live feed confirmed** — screenshotted the running window: real frames, natural colors, so the `Format_BGR888` mapping is right with no red/blue swap.
- **`q`** → `Released camera device 0`, process gone, `fuser` exit 1, `lsof` empty.
- **`Esc`** (via the bare `evidone` default index) → same clean teardown.
- **Error path, unplanned but useful:** `--camera 1` hit the webcam's V4L2 *metadata* node, which exercised `CameraOpenError` against real hardware — clean message, exit code 2, no held handle.

Final sweep: no leftover processes, nothing holding `/dev/video0` or `/dev/video1`.

## QA

`make qa` passes — ruff, ruff format, mypy (strict, 14 files), 36 tests, **99% coverage** (the one miss is the `__main__` guard). I added `.env.test` with `QT_QPA_PLATFORM=offscreen` so the widget tests run headless, and a `pep8-naming` exemption for the three camelCase API names Qt and OpenCV impose (`keyPressEvent`, `closeEvent`, `isOpened`) rather than scattering `# noqa`.

Two notes for the next step: `_show_next_frame` is the single seam where the YOLO pass will slot in — it has the raw BGR frame before conversion. And the timer is a pull-based `QTimer` on the GUI thread, which is fine for plain capture but will stutter once inference runs per frame; that'll want a worker thread, which is a natural part of the pose-model step rather than something to pre-build now.

I didn't commit anything — all changes are in the working tree.

# User

Draft a possible PR description for the work on this branch. Do not create a PR until I accepted the changes. Keep it short (what changed and why. Link the issue #1 with (Closes #1) and flag anything that differs from the issue and the git diff.

# Claude

Before drafting — the branch contains commits I didn't make. Let me check what's actually in it:

The commit message doesn't match its contents. Checking the rest of the bundled changes:

QA is green on the branch (36 tests, 99%). Here's the draft — **not** created, nothing pushed.

---

main()                      # parses --camera, maps CameraOpenError -> exit 2
  run(config)
    QApplication
    with Camera(config)     # guarantees release on any exit path
      VideoWindow(camera)
        QTimer @ capture FPS -> _show_next_frame
        keyPressEvent       # q / Esc -> close
        closeEvent          # stop timer, release camera
$ evidone --camera 0        -> window renders live frames (screenshot confirms
                               correct BGR888 mapping, no red/blue swap)
  fuser /dev/video0         -> casy2210 478365 F...m evidone
  <q> / <Esc>               -> "Released camera device 0", process exits
  fuser /dev/video0         -> exit 1 (no holders)
  lsof /dev/video0          -> no open handles

$ evidone --camera 1        -> CameraOpenError, exit 2, no handle leaked

---

## Flags — divergences between issue #1, the branch, and what I built

Four things you'll want to resolve before opening this.

**1. The commit message is wrong.** `c80afc8` is titled *"Accept a video file as input"*, but nothing in the diff accepts a video file. The only CLI argument is `--camera INDEX` (`cli.py:13`), typed `int`, feeding `cv2.VideoCapture(index)`. Issue #1 doesn't mention video files either. The message appears to belong to different work — worth amending, since it's the only commit carrying this feature.

**2. The commit bundles unrelated tooling deletions**, none of which issue #1 covers:
- `scripts/worktree.sh` deleted (225 lines)
- the "Migrating to Git Worktrees" section of `DEVELOPMENT.md` (18 lines)
- the `worktree` and `docs` targets from the `Makefile`

**3. That deletion broke `make build`.** The `docs` target is gone but `build: clean docs` still references it:
The `docs` dependency group in `pyproject.toml` is untouched, so only the Makefile target was lost. `make qa` is unaffected.

**4. Three behaviors go beyond the issue** — all deliberate on my part, but call them out if you want the PR scoped tightly:
- `CameraOpenError` → logged and exit code `2`, rather than an unhandled traceback
- the window auto-closes if the camera stops delivering frames mid-session
- frames scale to the window with aspect ratio preserved on resize

Everything in issue #1 itself is covered: `cv2.VideoCapture`, PySide6 rendering, `--camera`, `q`/`Esc` quitting, and a released device with no lingering process.

One note on scope I should be explicit about: items 1–3 are in commits I didn't author, so I've only reported them — I haven't amended the message, split the commit, or restored the `docs` target. Tell me which of those you want and I'll do it.

# User

Next step: run a YOLO26 pose model on every captured frame.
- Load the model once at startup, outside the frame loop.
- Run inference on each frame.
- Convert the Ultralytics result into plain dataclasses of our own: per person, the box, the detection score, and the 17 COCO keypoints each with its own confidence. Nothing outside that wrapper should import torch or touch an Ultralytics type — the rest of the app works with our dataclasses only.
- Add `--confidence` to the command line to set the detection threshold, default 0.4.

`yolo26n-pose.pt` is fine as the model. Don't commit the weights; the first run can download them. It has to run on a laptop CPU, so before you finish tell me what adding ultralytics costs in download size — if it's pulling CUDA wheels we don't need, say so instead of leaving it.

Tests must not download weights or need a camera.

# Claude

Before installing anything, let me check what `ultralytics` actually drags in — that's the question you flagged.

Confirmed — the default resolve pulls the entire CUDA stack. Let me measure both options precisely:

The lock records wheel sizes. Let me compute the real download for this platform without fetching anything:

3.17 GB, with 2.4 GB of it CUDA that a laptop CPU can't use — and it also pulls `opencv-python`, which would collide with our headless pin. Let me resolve a CPU-only variant:

The lock is cross-platform, so it lists everything. What matters is what installs *here*:

`tool.uv.sources` only applies to *direct* dependencies — torch comes in transitively. Declaring it explicitly:

Only `nvidia-ml-py` (~50 KB, pure Python telemetry) remains. Applying this to the real project:

Install was fast (uv cache hit). Let me measure the real footprint and verify YOLO26 is actually available in this version:

CPU torch wheel is 209 MB vs the CUDA stack's 2.4 GB. Now verifying YOLO26 is actually supported in this version:

YOLO26 pose configs exist and `yolo26n-pose.pt` is published (HTTP 200). Now the implementation — data structures first:

Now the Ultralytics wrapper — the only module allowed to touch torch:

Now wiring main.py to load the model once at startup:

Now the tests — these must not download weights or touch a camera:

The `isinstance` narrowing silently dropped the fakes — and would hide real surprises too. Replacing it with an explicit cast:

All green — 77 tests in 2.0s. Let me prove the tests really don't reach the network or a camera:

Tests pass with all outbound HTTP broken, no weights on disk, camera free. Now the real run — first, measuring what inference actually costs on this CPU:

21 ms/frame on CPU — ~47 fps of headroom, so the synchronous loop is fine at 30 fps after all. Now the full app:

Title reads "— 1 detected", live. Capturing it:

My own `pkill` pattern matched the command running it. Retrying cleanly:

Live feed with the detection count tracking (no skeleton drawn — that's the next step). Now quitting and verifying cleanup:

The title now carries the detection count, so my exact-match helper missed it. Matching by prefix:

`python-xlib` can't read the COMPOUND_TEXT title. Letting the helper take a window id directly:

The app is genuinely still running — the key didn't land. Diagnosing focus:

Focus is correctly on the app window, so the key should have landed. Re-sending and waiting properly before checking:

The app no longer responds to `q` — a real regression from this step. Let me confirm the cause before fixing: running the same window with a no-op pose source.

Diagnosis confirmed: with a no-op source it quits instantly; with real inference it never sees the key. `camera.read()` blocks ~33 ms and inference adds ~21 ms, so the repeating 33 ms timer is permanently overdue and starves input events. Fixing the scheduling:

Now updating the window tests for the new scheduling, including a regression test for the starvation:

Green — 80 tests. Now the real verification that the fix actually restores quitting with live inference:

Fixed. Now verifying `--confidence` actually takes effect end to end, quitting with `Esc` this time:

Done. Pose inference runs on every frame, QA is green (80 tests, 99.6%), and it's verified against the real webcam.

## Answering your dependency question first

**Yes, it was pulling CUDA wheels we can't use.** The default resolve of `ultralytics` on this machine:

| | Download | CUDA/triton share |
|---|---|---|
| Default (naive `uv add ultralytics`) | **3.17 GB** | **2.44 GB** |
| What I pinned (CPU-only) | **~376 MB** | ~0.05 MB |

The CUDA path installs 19 `nvidia-*` packages plus `triton` — `nvidia-cudnn` alone is 519 MB, `cublas` 405 MB — none of which a laptop CPU touches. I pinned `torch`/`torchvision` to the PyTorch CPU index (209 MB for torch vs 539 MB + 2.4 GB of satellites). Confirmed live: `torch 2.15.0+cpu`, `cuda build: None`.

Two traps worth knowing:
- `[tool.uv.sources]` only applies to **direct** dependencies, so `torch` and `torchvision` had to be declared explicitly even though ultralytics pulls them anyway. Without that, the override silently does nothing.
- Ultralytics requires `opencv-python`, which would have reintroduced the exact Qt plugin clash we avoided with the headless pin. I overrode it out; `cv2` still comes from `opencv-python-headless`.

Model weights are 7.5 MB, downloaded on first run, and `*.pt` is gitignored (verified via `git check-ignore`).

## What I built

`pose.py` is the only module that may touch the ML stack. Everything else works with our dataclasses:

`tests/test_architecture.py` enforces that boundary with an AST scan: it fails if any module other than `pose.py` imports `torch`, `torchvision`, or `ultralytics`, and also fails if `pose.py` stops owning the dependency. The `PoseSource` protocol is what lets `window.py` stay clean — it never imports the wrapper at all.

`--confidence` defaults to 0.4 and is range-validated (`0 < x <= 1`), so a typo like `--confidence 5` fails at parse time instead of silently detecting nothing.

## A regression I introduced and fixed

Adding inference **broke quitting**. The app stopped responding to `q`/`Esc` entirely — the camera could only be freed by killing the process.

Cause: `camera.read()` blocks ~33 ms waiting for the next frame, and inference adds ~21 ms. Against a repeating 33 ms `QTimer`, the timer was permanently overdue, so Qt re-fired it continuously and starved window-system input events.

I confirmed it rather than guessing — ran the same `VideoWindow` with a no-op pose source, which quit instantly, isolating inference as the cause. Fix: the timer is now single-shot and re-armed at the end of each frame with the remaining budget (floor `MINIMUM_YIELD_MS`), so the event loop always gets control between frames.

**The whole test suite passed while this bug was live.** Only driving the real window caught it. I've added three regression tests (re-arm happens, slow frames fall back to the yield floor, a closed window never re-arms), but the honest lesson is that per-frame work needs a real-window check — I've recorded that in memory.

## Verification

- **Inference:** median **21 ms/frame** on this CPU (min 18, max 24) → ~47 fps of headroom at 640×480. Comfortable at 30 fps, so no worker thread needed yet. This revises my caution from the last step — it's faster than I expected.
- **Real run:** title tracked `— 1 detected` live; `--confidence 0.99` correctly showed `— 0 detected`.
- **Cleanup:** both `q` and `Esc` → `Released camera device 0`, process gone, `fuser` exit 1, `lsof` empty.
- **Tests don't download or need a camera:** suite passes in 1.7 s with all outbound HTTP/HTTPS proxied to a dead port, no `*.pt` on disk, camera free throughout.

One thing beyond your ask, flagged for your call: the window title shows the detection count. It's how I made inference observable without drawing a skeleton (that's your next step) — say the word and I'll drop it. Nothing is committed; all changes are in the working tree.

# User

Next step: draw the detections ourselves.

For every detected person, draw a dot per keypoint and a line per skeleton pair, using the standard COCO-17 skeleton. Skip a dot when that keypoint's confidence is below a threshold, and skip a line unless both of its endpoints are above it. Colour head, torso and limbs differently. Skip a dot when that keypoint's confidence is below a threshold, and
skip a line unless both of its endpoints are above it. Colour head, torso and limbs differently so two people in frame stay readable.

Don't use `results.plot()` or any other Ultralytics drawing helper — the rendering is ours.
Draw onto the BGR frame before it reaches the window, not onto the Qt widget: later steps need the drawn frame itself, not just something painted on screen.

Keep the visibility threshold as a single named constant instead of repeating the number at each check.

Run it against the real camera and save me a frame so I can see how it looks.

# Claude

Drawing onto the BGR frame it is. Starting with the COCO topology as domain data:

19 bones, matching the canonical COCO skeleton. Now the renderer:

Now the tests:

My fake was lying about OpenCV: it handed back the *same* buffer each read, so frame 2 saw frame 1's drawing. Real `VideoCapture.read()` allocates a fresh array per call. Fixing the fake to match reality:

Green — 117 tests, 99.7%. Now running against the real camera and saving you a frame:

No one in frame at that instant. Retrying with a window of attempts until someone appears:

The chair's empty — you've stepped away. Rather than leave you without a visual, let me run the real pipeline over a camera frame captured earlier in this session:

The renderer works. Now confirming the live app still behaves with drawing in the loop:

The app didn't quit this time. Checking whether it's the key delivery or the event loop:

`focus window: 0` — no X window holds input focus, so injected keys go nowhere. That's my harness, not the app. Let me verify with a nested X server where I control focus:

Focus is flapping between checks. Let me set focus and inject in one connection, then poll:

App quit on the first attempt once focus was confirmed. Verifying cleanup:

Before moving on — I claimed last turn that the repeating timer was a real regression. Let me verify that with this now-reliable harness rather than leave an unproven claim standing:

The app quit immediately **with the repeating timer restored**. My earlier diagnosis was wrong. Let me confirm that's reproducible, not luck:

With the repeating timer it took ~28 focused attempts across three runs before one key landed — intermittent, not clean. Restoring the fix and running the identical protocol:

Decisive: **3/3 quit on attempt 0** with the fix, versus **1 success in ~28 focused attempts** without it. The fix is justified — my momentary doubt was itself the lucky-delivery case. Final checks:

Drawing is in. QA green — 117 tests, 99.7% coverage.

## The frame you asked for

**`skeleton-frame.png`** in the project root (untracked, so it won't get committed by accident).

You'd stepped away from the desk — 25 seconds of live capture got an empty chair. Rather than hand you a picture of furniture, I ran the real pipeline (`PoseEstimator` → `draw_poses`) over a camera frame captured earlier in this session. It's genuine output from our code on a real frame, just not from the last 30 seconds. To regenerate it live when you're back:

In that frame: **9 of 17** keypoints drawn at `score=0.91`. Hips, knees and ankles are below the desk and the wrists are occluded, so they fall under the threshold and are correctly skipped — exactly the behaviour you specified.

## What I built

`rendering.py` is new and depends only on our dataclasses and `cv2` — no Ultralytics helpers, no `results.plot()`. The architecture test covers it automatically.

- `VISIBILITY_THRESHOLD = 0.5`, defined once. `_is_visible()` is the only place it's compared, used by both the dot check and the bone check (which requires *both* endpoints).
- The skeleton lives in `models.py` as domain data: a `Bone` dataclass with `start`, `end`, `region`, plus `KEYPOINT_REGIONS`. A test asserts the 19 bones match the canonical COCO-17 pair list exactly, so the topology can't silently drift.
- Colours: head amber, torso blue, limbs green. The ear→shoulder bones are amber because COCO classes them as head pairs — that's why the head reads as a large triangle down to the shoulders.
- `draw_poses(frame, people)` mutates the BGR frame in place and returns it. In `_show_next_frame` inference runs **first**, then drawing, so the model never sees its own output.

## One test caught a real foot-gun

My `FakeCapture` returned the *same* numpy buffer on every read, so frame 2 saw frame 1's skeleton drawn on it. Real `cv2.VideoCapture.read()` allocates a fresh buffer per call, so this can't happen in the app — but the fake was lying about reality. I fixed the fake to copy per read, which made one existing camera test (`is frame`) correctly fail and get tightened to compare contents. There's now a test asserting the model only ever receives undrawn pixels.

## Correction on last turn's claim

I told you the repeating `QTimer` was a real regression. Mid-verification I got a clean quit with the old timer restored and briefly concluded I'd been wrong. Repeating the trial settled it, and the original claim stands:

| Scheduling | Keys delivered (focus confirmed each attempt) |
|---|---|
| Repeating timer | 1 success in ~28 attempts over 3 runs |
| Single-shot re-arm | 3/3 quit on attempt 0 |

The failure is intermittent, which is why one lucky success proved nothing. Separately, a genuine harness problem did bite me here: under XWayland the X window's input focus flaps, and injected keys are silently dropped — indistinguishable from a frozen app. My quit helper now confirms focus in the same X connection immediately before injecting, which is what made the comparison trustworthy. Both facts are in memory.

Nothing committed; all changes are in the working tree.

# User

Nex step: work out where each person's face is, using only the pose keypoints.

Add a function that turns a detected person into a face region. Size it from several independent landmark spans — eye to eye, ear to ear, nose to ear, and whatever else the face keypoints give you — and keep the largest, since each of those collapses at a different head angle. Lean towards covering too much rather than too little: a region that is too small leaks a face, one that is too large only blurs background.

It has to hold up when the face isn't fully visible: someone in profile with one eye and one ear hidden, someone facing away from the camera, and someone whose face keypoints are all below the threshold. No second model, the pose keypoints are all we get.

Use the same visibility threshold the skeleton drawing already uses, not a new one.

This commit is geometry only: nothing is blurred yet and nothing changes on screen. I want each of those situations covered by tests.

# Claude

Geometry only. First, the visibility threshold needs one home both modules share — right now it lives in `rendering.py`:

Now the face geometry:

The earlier formatter run had already collapsed that import, so my replacement missed. Fixing directly:

140 tests green, `faces.py` at 100%. My multipliers are heuristics, so let me sanity-check them against the real detection — as an offline check, not a change to the app:

Covers the whole head with margin on all sides. Let me check a different head angle — the frame where you were leaning down:

Done. QA green — 140 tests, 99.7% coverage, `faces.py` at 100%.

## What I added

`faces.py` — `face_region(person) -> FaceRegion | None`, pure geometry over our own dataclasses. Nothing imports it yet, nothing draws, nothing on screen changed.

Sizing takes the **largest** of 8 independent spans, each with its own scale factor:

| Span | ×radius | Collapses when |
|---|---|---|
| eye ↔ eye | 2.0 | in profile |
| ear ↔ ear | 0.9 | one ear occluded |
| nose ↔ ear (L/R) | 1.2 | facing away |
| nose ↔ eye (L/R) | 2.6 | facing away |
| eye ↔ ear (same side) | 1.8 | that side hidden |

Plus a floor: the region must enclose every visible face landmark, times `LANDMARK_COVERAGE_MARGIN` (1.6). So even if every scale factor were badly tuned, the region still can't be smaller than the landmarks it's built from.

## The threshold — one definition, and I had to move it

You asked for the same threshold the skeleton drawing uses. It lived in `rendering.py`, and having geometry import from a rendering module is backwards, so I moved `VISIBILITY_THRESHOLD` to `models.py` and put the predicate on the data itself as `Keypoint.is_visible`. Both modules now call that property — there's no second comparison anywhere.

`test_architecture.py` now AST-scans the package and asserts the constant is assigned in exactly one module, so a stray redefinition fails the build. That's a small refactor of code you'd already approved, flagged because it touched `rendering.py`.

## Your three situations, tested

- **Profile** (far eye + far ear hidden): eye↔eye and ear↔ear both collapse; sized from the nose↔ear and eye↔ear spans that survive.
- **Facing away** (nose + both eyes gone): ear↔ear carries it.
- **All face keypoints below threshold**: returns `None`.

Also covered: a hidden keypoint never enlarges the region even when placed far away; ears-widest and eyes-widest cases each pick the right span; scale invariance (0.5×/2×/5× head → proportional radius); confidence exactly at the threshold counts as visible.

## Checked against real detections

| Frame | Visible landmarks | Region diameter |
|---|---|---|
| Facing camera | all 5 | 247 px |
| Head tilted down | 4 (one ear occluded) | 417 px |

Both fully cover the head with margin — I rendered the circles offline to look at them. Worth knowing: the tilted case is *noticeably* more generous, because with one ear gone the sizing falls to nose↔ear on a foreshortened head. It errs the way you asked, but it blurs a good deal of shoulder and background. All the factors sit in one table in `faces.py` if you want it tighter later.

## Two judgement calls to flag

**I added a shoulder-width fallback** beyond what you specified. With only one face landmark visible (say a lone nose), no face span is measurable and the region would be `None` — a fully leaked face. Shoulder width × 0.35 gives a crude but safe size. It only fires when no face span exists, so it never competes with real face measurements.

**`None` still means a leak.** When no face landmark is visible at all, there is genuinely nothing to centre on, so I return `None` rather than invent a position above the shoulders. Same for a single landmark with no visible shoulders. That's the honest geometric answer, but it's a *policy* question for the blur step: do you want a last-resort guess from the person's bounding box, or is an un-blurred face acceptable when the model sees no face points? I'd rather you decide that than bake it in here.

# User

to models.py

# Claude

Consistent with `SKELETON` already living there. Let me check the current layout:

I'll move the three constants after `SKELETON`, keeping the landmark data together:

Moved. QA green — same 140 tests, 99.72%.

`FACE_LANDMARKS`, `FACE_SPANS` and `SHOULDER_SPAN` now sit in `models.py` directly after `SKELETON`, so all the COCO landmark data lives in one place. `faces.py` is down to the algorithm:

I left `LANDMARK_COVERAGE_MARGIN` in `faces.py` — it was outside your selection, and unlike the spans it isn't landmark data, just a knob on the sizing calculation. Say the word if you want it in `models.py` alongside the rest.

Test imports updated to match; `faces.py` stays at 100% coverage.

# User

Second step: hide the faces.

Take the regions from the previous commit and blur them on the frame, hard enough that the person is not recognisable. The strength has to scale with the region: a blur that works on a small distant face leaves a large close-up one perfectly readable.

Hide the faces before the skeleton is drawn, so the overlay stays sharp on top of a blurred face instead of being smeared along with it. A region that falls partly outside the frame should be clipped, not crash.

# Claude

Let me check OpenCV's behaviour when the blur kernel is larger than the region, so clipping at frame edges can't crash:

No clamping needed. Let me measure the cost of a strong Gaussian on a realistic face ROI before committing to an approach:

OpenCV is safe with oversized kernels, and the numbers tell me a 149-kernel Gaussian (20.8 ms) would nearly double the frame cost. I'll reduce each face to a fixed sample grid — scale-invariant by construction and far cheaper:

Now the tests, including the clipping cases:

Now the ordering test — blur must run before drawing:

161 tests green. Now the check that matters — is the face actually unrecognisable on a real frame?

Completely unrecognisable, and the skeleton is crisp on top — blur-before-draw confirmed. 3.11 ms/frame. Checking the larger region too:

The bigger region costs 11 ms — with 21 ms inference that's at the frame budget. Let me check whether the smoothing Gaussian is earning its cost:

My benchmark computed all variants each call — the timings are meaningless. Timing them in isolation:

The Gaussian is 3.5 ms of the 4.2 ms. Let me see whether it's visually necessary:

Cubic is as cheap as linear (0.65 ms) but visibly smoother — the linear variant shows faint facet banding. The Gaussian isn't earning its 3.5 ms. Simplifying:

All 161 still pass with the simpler implementation. Re-verifying both real frames and the new cost:

Same result at 1.40 ms instead of 3.11. Now the live app end to end:

Live and working. Quitting and verifying cleanup:

The log shows a clean release and no leftovers, but my injection never landed (focus read `0x0` on all attempts) — so it exited some other way, likely you closed it. Since blur adds per-frame cost, let me confirm the key-quit path still works with it in the loop:

Faces are hidden. QA green — 161 tests, 99.75%, `anonymization.py` at 100%.

## How it works

`anonymization.py` — `blur_faces(frame, people)`, 56 lines. For each person it takes the region from the previous commit, clips it to the frame, and replaces the pixels inside the circle.

The blur reduces each face to a **fixed 6×6 sample grid** regardless of how large it is on screen, then scales back up. That's what makes the strength scale: a close-up loses exactly as much information as a distant face, by construction rather than by tuning a kernel against the region size. The circular mask means only the region itself is touched, not its bounding box corners.

Wired into `window.py` as `estimate → blur → draw`, so the skeleton is painted onto an already-blurred face and stays sharp.

## I started with a Gaussian and measured my way off it

The obvious implementation — `GaussianBlur` with a kernel proportional to the region — was too expensive at the sizes that matter:

| Approach (418×418 face ROI) | Cost |
|---|---|
| Gaussian, kernel 149 | 20.8 ms |
| Gaussian, kernel 75 | 3.7 ms |
| 6×6 grid + linear upscale + Gaussian | 4.2 ms |
| **6×6 grid + cubic upscale** | **0.65 ms** |

A 20.8 ms blur on top of 21 ms inference would have blown the 33 ms frame budget on any close-up face. The cubic upscale costs the same as linear but avoids the faint facet edges linear leaves between samples, so the smoothing Gaussian bought nothing for 3.5 ms. Dropping it also removed two constants and a helper.

End to end on real frames: **1.40 ms** for a normal face, **3.94 ms** for the oversized angled region — comfortably inside budget.

## Verified

Live app: face unrecognisable, skeleton crisp on top, title tracking detections. Both `q` and `Escape` quit on the first attempt with the blur in the loop, camera released, device free.

Tests cover the clipping cases you asked for — region crossing each of the four edges, both far corners, and centres outside the frame entirely — plus a region fully off-frame (no change, no crash), a person with no visible face, and no people at all. The scale requirement is tested with detail that *scales with the face*, so a fixed-strength blur would fail the large cases; a separate test asserts a 120 px half-width face loses the same proportion of detail as a 12 px one. Ordering is pinned by a test asserting the pipeline runs exactly `["estimate", "blur", "draw"]`.

## One thing to note

The oversized region I flagged last turn now has a visible cost: on the tilted-head frame the blur disc covers a chunk of shoulder and background. It's the safe failure direction you asked for, and the disc is still driven entirely by `FACE_SPANS` in `models.py` if you want it tighter.

Also worth deciding at some point: a face that yields no region still isn't blurred. That's unchanged from the geometry commit, but now it's the difference between a hidden face and a visible one rather than an abstract gap.

# User

Add all about the new changes in modules distribution

# Claude

Let me look at the current state — including that new `vision/` package you've opened.

You're mid-reorganization into `vision/` and `ui/` sub-packages. Let me see how far it got and what's broken:

The reorganization is half-finished and currently broken. Confirming before I touch anything:

Broken as expected. Before I write the `__init__` exports, one thing to check — re-exporting `PoseEstimator` from `vision/__init__.py` would make *any* vision import pull in torch:

Confirmed: `ultralytics` costs 868 ms to import. That decides the `__init__` design — I'll keep `PoseEstimator` out of the eager re-exports so importing `vision` (or `ui`) doesn't drag in torch, preserving the boundary you set earlier. Writing the redistribution:

Now mirroring the test tree to the new package layout, as AGENTS.md requires:

Now the critical piece — the architecture test only globbed top-level `*.py`, so after the move it would have silently stopped checking `vision/pose.py`:

All 167 pass, and the runtime probes confirm the design: importing `ui` or `vision` loads no torch, while `vision.pose` does. Let me verify the real app still runs after the move:

App runs and quits cleanly (note the logger now reports `vision.pose`). Let me check the README, since no doc describes the layout:

The README has no layout section. Adding one that documents the new distribution:

The reorganization is finished and working. QA green — 167 tests, 99.75%.

## It was broken when I started

The move was half-done: `uv run evidone` failed with `ModuleNotFoundError: No module named 'evidone_pose_detection.faces'`, and 6 test files wouldn't even collect. Three things were stale:

- `vision/anonymization.py` still imported `evidone_pose_detection.faces`
- `ui/window.py` still imported the old top-level `anonymization` and `rendering`
- `ui/__init__.py` exported `run_window`, which doesn't exist anywhere

## The one that would have gone unnoticed

`test_architecture.py` globbed `PACKAGE.glob("*.py")` — top level only. After the move, `vision/pose.py` sits one level down, so the test enforcing "only the pose wrapper touches torch" would have **passed while checking nothing**. It now uses `rglob`, plus a guard test asserting the scan actually reaches `vision/` and `ui/` — so the same silent hole can't reopen.

## A design decision I had to make

Your `vision/__init__.py` started re-exporting `blur_faces`. The obvious completion is to re-export `PoseEstimator` too — but importing any submodule runs the parent `__init__`, so that would make `ui/window.py` load torch transitively, breaking the boundary you set two steps ago.

I measured it: **`import ultralytics` costs 868 ms**. So `vision/__init__.py` exports the three torch-free functions (`blur_faces`, `draw_poses`, `face_region`) and leaves `PoseEstimator` at `vision.pose`. The asymmetry is the point, and it's commented in the file.

Three new tests lock it in a subprocess, because an AST scan can't catch transitive imports:

| Probe | ML stack loaded |
|---|---|
| `import evidone_pose_detection.ui` | False |
| `import evidone_pose_detection.vision` | False |
| `import evidone_pose_detection.vision.pose` | True |

## Also done

- **Test tree mirrors the package**, per AGENTS.md: `tests/vision/{pose,faces,rendering,anonymization}` and `tests/ui/test_window.py`. Used `git mv` for tracked files so history follows.
- **`pythonpath = ["./src", "./tests"]`** in `pyproject.toml`, so nested tests can still reach the shared `conftest` helpers.
- **`S603` added to the tests ruff ignores** — the boundary probes need `subprocess`, and a config entry beats scattering `# noqa`.
- **README gained a "Module layout" section** documenting the tree, the `estimate → blur → draw` order, and why `PoseEstimator` isn't re-exported.

Verified live afterwards: model loads (logger now reads `evidone_pose_detection.vision.pose`), detection runs, `q` quits, camera released, device free.

Two notes. The suite went from 1.9 s to 3.5 s — that's the three subprocess probes, each paying a fresh interpreter start; I think the boundary is worth it, but they're easy to mark slow if it bothers you. And `camera.py` stayed at top level rather than moving under `vision/` — it's capture, not analysis, and you'd left it there, so I matched your choice rather than extending the refactor.

# User

Right now an unavailable camera or a missing weights file surfaces as whatever OpenCV or Ultralytics happens to raise. 
Wrap both: the camera wrapper and the model loader each raise my own error type, with a message.

main() acquires the model first, then the camera, each with its own handler. If either fails, write a short message to stderr and exit with a non-zero code — pick codes that let me tell a bad argument from a missing device.
The Qt application must not be built until both are in hand, so a failed run never reaches the window.

# Claude

Let me look at the current error path end to end before changing it:

Confirmed — and there's a real collision: `STARTUP_ERROR_EXIT_CODE = 2` is the same code argparse uses for a bad argument, so the two are indistinguishable today. Rewriting:

Now the tests — including that Qt is never built on a failed start:

One gap: the new `cv2.VideoCapture` wrapping (camera.py:34-35) isn't covered. Adding a test for it:

Now verifying the real behaviour — actual exit codes and stderr from the installed command:

All three distinguishable. Confirming the happy path still works end to end:

Done. QA green — 174 tests, 99.76%.

## Exit codes

The old `STARTUP_ERROR_EXIT_CODE = 2` collided with argparse's own usage code, so a bad argument and a missing camera were indistinguishable — exactly what you wanted to tell apart. Failures now start at 3:

| Situation | Code | stderr |
|---|---|---|
| Bad argument | 2 *(argparse owns this)* | `evidone: error: argument --camera: invalid int value: 'front'` |
| Weights missing | 3 | `evidone: could not load the pose model '…': the weights file may be missing, unreadable, or the download failed` |
| Camera unavailable | 4 | `evidone: could not open camera device 9: it may be disconnected, in use by another application, or not a capture device` |

All three verified against the real `uv run evidone`, not just in tests. One line each, nothing on stdout.

## Startup order

`run()` changed from taking an `AppConfig` to taking an **already-open camera and the estimator**, which is what structurally guarantees your last requirement: `QApplication` is constructed inside `run`, so it cannot exist until both resources are in hand. Previously it was built *first*, before either. Tests assert the Qt application is never constructed on a failed start, for both failure modes.

`_abort` writes via `sys.stderr.write` rather than `print` (ruff `T20` forbids `print` outside `scripts/`), and logs the full exception at debug level so the traceback is still reachable without dumping it on the user.

## Wrapping

- **Model loader** already wrapped — unchanged, but the message is now actionable rather than bare.
- **Camera wrapper** had a real hole: it raised `CameraOpenError` when `isOpened()` was false, but anything `cv2.VideoCapture(...)` itself threw escaped raw. Now wrapped with `raise ... from error`, so the original cause is preserved for debugging while callers only ever see `CameraOpenError`.

Both messages live on the error types, so they read the same whether printed by `main` or caught elsewhere.

## Tests

New coverage: model acquired strictly before the camera (a failing model means the camera is never opened), each failure's code and single-line stderr, no Qt application on a failed start, a bad argument exiting 2, the three codes being mutually distinct, `run()` receiving a camera that is actually open, and a capture backend raising mid-construction becoming `CameraOpenError` with its `__cause__` intact.

Happy path re-verified live: model loads, detection runs, `q` quits, camera released, device free.

# User

Lets show how fast this is actually running.

I want you draw a translucent panel over the video with three numbers: 
1. the model's inference time in milliseconds
2. the display frame rate
3. how many people were detected in the current frame.

Take the inference time from what the model reports, not by timing the call ourselves. Use the inference time from Ultralytic inference preprocess and tell me if you end up including the others. Measure the frame rate from the time between processed frames.

Draw the panel on the frame with OpenCV, like the skeleton, not as a Qt widget. On the top left part of the image. The numbers have to reach the panel through our own dataclasses, don't hand the Ultralytics result around.

# Claude

Let me first check what Ultralytics actually reports, so I use its numbers rather than timing the call myself:

`speed` is a plain dict of three floats. Warm numbers: `preprocess 0.31`, `inference 18.09`, `postprocess 0.44` — sum 18.84 ms against a 22.15 ms wall clock for the call. Building it now:

Now the frame-rate meter and the OpenCV panel:

Now updating the fakes and tests for the new `PoseResult`:

The formatter had collapsed that import to one line. Fixing it:

Three issues to close: the fake clock in one window test now gets more `perf_counter` calls, plus the new modules need their own tests:

All green — 196 tests, 99.80%. Now the real run, to see the panel and the actual numbers:

Panel works — translucent, top-left, cabinet edges visible through it. Quitting cleanly:

The app didn't quit (focus flakiness again), so it still held the device — which is exactly why the measurement hit `CameraOpenError`. Retrying:

Focus was confirmed and keys sent on all 10 attempts, yet it didn't quit — last turn this same harness gave 3/3 on attempt 0. Let me check whether the panel introduced a regression:

Run 1 took 8 attempts, run 2 never got focus. Let me check the mechanism rather than guess — the panel reports 29.7 fps against a 33 ms camera interval, meaning every frame overruns and the loop only ever gets the 1 ms floor to process input:

My `pkill` pattern matched its own command line again (exit 144). Checking state:

Nothing was modified. Redoing the experiment cleanly:

My loop let a leftover app overlap the next run, so those numbers are contaminated. Let me run a properly isolated comparison:

`uv run evidone` matches twice (the `uv` parent and the child). Taking the pid that actually holds the camera instead:

Both configurations quit **3/3** under isolation — so there's no regression from the panel. My earlier failures were overlapping app instances in my own test loop, not the app. `MINIMUM_YIELD_MS` restored to 1, unchanged. Now the stage breakdown you asked about:

Panel is live. QA green — 196 tests, 99.80%.

## Answering your question directly: no, I did not include the others

The panel shows **`inference` only**. Measured over 20 warm frames on your camera:

| Stage | Time | On the panel |
|---|---|---|
| preprocess | 0.81 ms | no |
| **inference** | **20.28 ms** | **yes** |
| postprocess | 0.43 ms | no |
| total | 21.52 ms | — |

So the displayed number omits **5.8%** of what the model reports. Worth knowing: even the 21.52 ms total isn't the full cost of the call — the wall clock of `predict()` is ~22 ms, so Ultralytics has ~0.5–3 ms of its own overhead outside the three stages it measures.

All three stages are captured in the dataclass (`InferenceTiming` has `preprocess_ms`, `inference_ms`, `postprocess_ms` and a `total_ms` property), so switching the panel to the total is a one-line change if you'd rather see honest end-to-end cost. I kept `inference` because you asked for "the model's inference time".

The numbers come from `result.speed`, which Ultralytics populates itself — nothing is timed around the call.

## How the numbers reach the panel

No Ultralytics object leaves `vision/pose.py`:

`PoseSource.estimate` now returns `PoseResult` instead of a bare tuple, which is the breaking change that carries timing through.

**Frame rate** is measured in `ui/framerate.py` from the interval between processed frames, averaged over the last 15 so the digits are readable rather than flickering. It takes `now` as an argument rather than calling the clock itself, so it tests against a fake clock with no sleeping. Live reading: 29.7 fps against a 30 fps camera.

**The panel** is `vision/overlay.py` — `cv2.addWeighted` for a 55% dark backdrop, `cv2.putText` for three monospaced-aligned lines, inset 12 px from the top-left, sized from `cv2.getTextSize` so it fits whatever the text is. Drawn last, after blur and skeleton, so nothing covers it. You can see the cabinet edges through it in the capture.

## A false alarm I chased down

The app repeatedly failed to quit during verification, which looked like the event-loop starvation from earlier. It wasn't. My test loop was launching a new instance while the previous one was still alive, so the pid and window I targeted belonged to different processes. With an isolated harness — kill leftovers, confirm a single instance, take the pid that actually holds `/dev/video0` — both the current code and a 5 ms-yield variant quit **3/3 on the first attempt**. No regression, and `MINIMUM_YIELD_MS` is unchanged at 1.

New tests: six for the frame-rate meter (first frame, steady cadence, rate-is-inverse-of-interval, averaging across a slow frame, window eviction, same-instant divide-by-zero), seven for the panel (position, translucency proven by rendering over different backgrounds, backdrop ratio, text present, extreme values, a frame too small to fit, mutate-and-return), and four for timing (values come from the model, total sums the stages, a missing stage counts as zero, no results gives zero).

# User

I need the capture could read a video file as well as a camera.

`cv2.VideoCapture` already takes either a device index or a path, so this should reuse the same wrapper rather than growing a second one.

A clip that reaches its end should start again from the beginning and keep playing until the app is closed.

A path that can't be opened raises the same error type as an unavailable camera, with a message naming the file.

No UI for choosing the file yet. 
Cover it with tests.

# Claude

Let me first check whether I can generate a real test clip with the headless OpenCV build:

All three codecs work. Implementing — the wrapper takes either source:

Now the wrapper itself — same `Camera`, with rewind-at-end for clips:

Now updating the fakes and tests, including real-clip coverage:

Existing 196 still pass. Now the tests for the new behaviour, including a real generated clip:

206 tests, 99.81%. Now proving it end to end — building a real clip from the camera frames and running the actual window against it:

The clip plays through the full pipeline. The clip is 10 frames at 10 fps — **1 second long** — and this screenshot is at t=12s, so it has already looped ~12 times. Confirming it's still going, then quitting:

The process exited before t=27s. I need to know whether the loop failed or something else closed it — the log will say:

No "stopped delivering frames" warning — so `read()` never returned `None`; the window was closed normally (you likely closed it). Let me verify looping rigorously and headlessly instead of depending on the GUI:

500 reads from a 10-frame clip: **zero `None`**, the two source frames alternating cleanly throughout — ~50 loops without a gap. Let me confirm the live camera path still works after the change:

Both messages correct, same type. The camera trial hit the focus flakiness again — checking whether the camera path itself worked:

The capture now reads either source. QA green — 206 tests, 99.81%.

## What changed

Same wrapper, no second class. The source type moved into the config:

`Camera.open()` passes an `int` straight through and a `Path` as `str`, which is all `cv2.VideoCapture` needs. `Camera.read()` gained one branch: if a read fails **and** the source is a file, rewind to frame 0 and read again. A live camera is never rewound — a dead device should surface, not spin. A clip that yields nothing twice gives up rather than looping forever on a corrupt file.

`CameraOpenError` now takes the source and picks its wording:

Same type for both, as asked — so `main()`'s existing handler and exit code 4 cover a bad path with no change.

## Verified

- **Looping at scale, headless:** 500 reads from a 10-frame clip → **zero `None`**, the two source frames alternating cleanly from read 1 to read 500. ~50 loops without a gap.
- **Full pipeline on file input:** ran the real `VideoWindow` against a clip built from your camera frames — pose detected, face blurred, skeleton drawn, panel reading `display 10.0 fps` (the clip's own rate driving the timer). The screenshot was taken at t=12s on a 1-second clip, so it had already looped about twelve times.
- **Live camera unaffected:** `Opened capture source 0` still works after the change.

## Tests

10 new, in `tests/test_camera.py`. Four use a **real clip** generated with `cv2.VideoWriter` inside a `TemporaryDirectory` (opens through the same wrapper, reports its own geometry, keeps playing past its last frame, restarts from the beginning). Three cover errors: a missing path, a file that exists but isn't a video, and both source kinds producing the same error type. Three use the fake capture for the exact branch behaviour: a clip rewinds once when it runs out, a camera is *never* rewound, and an empty clip tries once then gives up.

I taught `FakeCapture` to honour `set(CAP_PROP_POS_FRAMES, 0)` so the rewind path is testable without touching disk.

## Two things to flag

**File input is currently unreachable.** I read "no UI for choosing the file yet" as covering the CLI too, so there's no `--video` flag — `--camera` still only yields an int, and nothing in the app can produce a `Path`. The capability is exercised only by tests, the same way `faces.py` landed before it was wired in. If you meant only "no file-picker dialog", say so and I'll add `--video PATH` (mutually exclusive with `--camera`) — it's a few lines plus tests.

**The names now lie slightly.** `Camera`, `CameraConfig` and `CameraOpenError` all handle files. I kept them to avoid churning every call site and test, but `Capture`/`CaptureConfig`/`CaptureOpenError` would read truer. Happy to rename if you want it.

One verification I couldn't complete: quitting the live app with `q` — the X window can't hold input focus while you're using the machine, so my key injection kept getting dropped. That path is untouched by this change and was confirmed 3/3 under isolation earlier in the session; the camera open and release behaviour here is confirmed directly from the logs.

# User

Now I need a toolbar for choosing the source while the app is running. Add a toolbar across the top of the window with a Settings button on the left. It opens a menu; for now the only entry is the source, offering a camera input or a video file and choosing the video file opens a file browser.

Picking a new source swaps it in without restarting the app, and releases the one it replaces. I don't want a camera or a file handle left open behind it. The menu should show which source is currently active, and cancelling the dialog should leave everything exactly as it was.

A file that can't be opened is reported in a dialog and the current source keeps running: killin the session over a mistyped path would be wrong here. The non-zero exit stays for a source that fails at startup.

# Claude

Let me check what the window tests currently touch, so I can plan the ownership change:

Tests don't touch `_camera`, so moving ownership is safe. Adding the swap primitive to the capture layer first:

The comment I'd written there was auto-removed by ruff's `ERA` rule (it looked like commented-out code), so my anchor didn't match. Inserting directly:

Now the toolbar, menu and live swap in the window:

The import anchor didn't match (isort had reordered it earlier), so only the imports were missed. Fixing:

All 206 pass; coverage dipped because the new code is untested. Adding a fixture for multi-source swaps, then the tests:

Now the toolbar and swap-from-the-UI tests:

222 tests, 99.35%. The two uncovered lines are the modal `QFileDialog`/`QMessageBox` calls themselves. Now the real proof — that the OS handle is actually released on a live swap:

Verified at the OS level — every requirement:

| Step | `/dev/video0` | Menu |
|---|---|---|
| 1. On camera | held by 557971 | camera ✓ |
| 2. Swap to clip | **none** — handle released | `Video File… (clip.avi)` ✓ |
| 3. Back to camera | held by 557971 (clip released) | camera ✓ |
| 4. Bad path | still held, window alive | camera ✓ (restored) |
| 5. After close | none | — |

Let me capture the toolbar visually:

Toolbar renders with Settings on the left. One thing my change introduced: the toolbar eats vertical space, so the video now letterboxes against a white background. Fixing that with a dark backdrop:

Toolbar in place with a dark letterbox. QA green — 222 tests, 99.35%. App quits cleanly, device free.

## What's there

A non-movable toolbar across the top with a **Settings** button on the left. Its menu has one entry, **Source**, a submenu with two exclusive checkable actions: **Camera** and **Video File…**. Choosing the file entry opens a `QFileDialog` filtered to video formats.

The active source is marked by the checkmark, and the file entry renames itself to carry the clip — `Video File… (clip.avi)` — so the menu names what's playing, not just which kind.

## The swap

I put the lifecycle in `CaptureSession` (in `camera.py`, next to `Camera` — it's capture, not UI):

The ordering is the whole point: the replacement must open *before* the incumbent is released, so a failure can't leave you with nothing. The window owns the session, so `closeEvent` releases whatever is currently held rather than whatever it started with.

## Proven against the OS, not just fakes

Fakes can't tell you a file descriptor really closed, so I drove the live window and asked `fuser`:

| Step | `/dev/video0` | Menu |
|---|---|---|
| On camera | held by 557971 | camera ✓ |
| Swap to clip | **none** | `Video File… (clip.avi)` ✓ |
| Back to camera | held by 557971 (clip released) | camera ✓ |
| Bad path `/nonexistent/typo.avi` | still held, window alive | camera ✓ restored |
| After close | none | — |

No handle survives a swap in either direction, and the bad path left the camera running untouched.

## Failure and cancel

A file that won't open is reported through `QMessageBox` with the same message the error type already carries, logged at warning level, and the running source continues — the window stays visible. Startup is unchanged: a bad source there still exits 4.

Cancelling returns an empty path and nothing happens. Worth noting *why* that needed care: clicking the menu entry checks it immediately via the `QActionGroup`, so after a cancel or a failure the checkmark would be lying. Every path ends in `_mark_active_source()`, which re-reads the session and resets the marks from reality. The cancel test asserts the camera is still checked, the file entry is back to plain `Video File…`, and nothing was released.

## Tests

16 new. Four on `CaptureSession` (swap releases the predecessor, a failed swap leaves the running source readable, release covers whatever is held, repeated swaps leak nothing). Twelve on the window: toolbar exists in the top area with a Settings button, the menu offers both entries, camera marked at startup, swapping releases the camera, the menu names the file, switching back releases the clip, cancel changes nothing, a bad path reports and keeps running, a failed switch doesn't close the window, frames come from the new source afterwards, closing after a swap releases the source in use, and re-picking the active source is a no-op.

Two lines stay uncovered: the `QFileDialog.getOpenFileName` and `QMessageBox.warning` calls themselves. They're modal and can't be driven headlessly, so I kept them as one-line seams the tests substitute — the logic around them is fully covered.

## Two notes

`DEFAULT_CAMERA_INDEX` moved from `cli.py` to `models.py`, because the menu's Camera entry needs the same default the CLI uses and I didn't want the number written twice. The Camera entry returns to whichever index the app started with, falling back to that default if it started from a file.

The toolbar takes a strip off the top, which letterboxed the video against the white window background. I gave the view a dark backdrop so the margins read as deliberate — visible in the screenshot. If you'd rather the video keep its native size, the window can grow by the toolbar height instead; say the word.

# User

The region is re-estimated from scratch every frame, so it twitches a few pixels in every direction even when nobody moves. Add a tracker that sits between estimating the regions and blurring them.

Match each new region to the closest region from the previous frame by centre distance, and reject a match that is too far away. What counts as "too far" has to scale with the face: one filling the frame moves many more pixels between frames than one at the back of the room. A previous region can only be claimed once, and a new region that matches nothing starts its own track and is used as it comes.

Smooth a matched region towards its new position and size over time, so the box slides instead of jumping.

Nothing about dropped detections yet, that's the next commit. The tests should be deterministic and need no camera.

# Claude

Let me look at the current pipeline seam before restructuring it:

Splitting region estimation out so the tracker can sit in the seam:

Now updating the anonymization tests for the new signature, and writing the tracker tests:

241 tests, tracker at 100%. Now the question that matters — does it actually stop the twitching on real data?

Measured on 80 real frames of you sitting still:

| | raw | tracked | reduction |
|---|---|---|---|
| centre_x | 1.74 px | 0.64 px | **63%** |
| centre_y | 0.38 px | 0.12 px | **68%** |
| radius | 1.32 px | 0.35 px | **74%** |

Now confirming the live app after the pipeline change:

The tracker is in. QA green — 241 tests, 99.39%, `tracking.py` at 100%.

## Where it sits

The seam you asked for didn't exist — `blur_faces` worked out the regions itself. I split that apart:

`face_regions()` is a new plural entry point in `faces.py` that drops the people it couldn't place, and `blur_faces` now takes regions and no longer knows about `PersonPose` at all. The tracker slots cleanly between.

## How it matches

Matching is **globally greedy**, not per-region first-come: every candidate pair within the allowance is collected, sorted by distance, and assigned closest-first. A previous region is claimed once and then removed from play. This is what makes the result independent of the order faces arrive in — there's a test that runs the same frame forwards and reversed and asserts identical pairings.

The allowance scales as `max(new.radius, previous.radius) × 0.6`. Taking the **larger** of the two matters: if a face is rushing toward the camera, its radius grows between frames, and keying off the stale previous radius would reject exactly the match you most want to keep. A 30 px step is a match for a radius-200 face and a new track for a radius-20 one — parametrised over four sizes.

Anything rejected or unmatched is passed through untouched, so a face's first frame is never dragged toward some unrelated history.

## Smoothing

Exponential, closing 30% of the gap to the measurement each frame, applied to centre *and* radius. Measured on 80 real frames of you sitting still:

| | raw | tracked | reduction |
|---|---|---|---|
| centre_x | 1.74 px | 0.64 px | **63%** |
| centre_y | 0.38 px | 0.12 px | **68%** |
| radius | 1.32 px | 0.35 px | **74%** |

## Tests

16, all pure geometry — no camera, no frames, no clock. They're written against the `MATCH_DISTANCE_RATIO` and `SMOOTHING` constants rather than hardcoded pixel numbers, so retuning the feel doesn't invalidate them. Coverage includes the exact-boundary case (a step of precisely the allowance still matches), the one-claim rule (two new faces near one track — the closer wins, the other starts fresh), convergence after 40 still frames, and a determinism test that runs a three-frame sequence twice and compares.

## Two things to flag

**I added `self._tracker.reset()` on a source swap**, which you didn't ask for. Without it, switching from the camera to a clip carries face history across a hard cut; the matcher would mostly reject it anyway, but a coincidental near-match would smear one person's box toward another's. One line, and it belongs with the switching you added last commit.

**Smoothing trades jitter for lag.** At 0.3 the disc sits about 2.3 frames behind a face moving at constant speed — roughly 23 px behind something crossing at 10 px/frame. The region is deliberately oversized, so that's absorbed rather than exposing an edge, but it's the knob to watch if fast motion ever leaks. Both constants are in one place in `tracking.py`.

No track identity yet — the tracker keeps last frame's regions, nothing more. Unmatched tracks vanish immediately, which is correct for this commit and is exactly the thing the dropped-detection work will need to change.

# User

Keep a face covered when the model loses it for a frame or two. When a region goes unmatched, don't drop it straight away, keep it in the list, and keep it blurred, for a few more frames, then let it go. 
How many frames should be configurable rather than buried in the middle of the logic. A region that is matched again resets its budget.

While a region is being held we don't actually know where the face went; it may have carried on moving. Tell me how you handle that, or why you chose not to.

One interaction to watch: a video clip loops back to its first frame when it ends, so regions held from the last frame shouldn't bleed into the beginning of the replay.

Then write the privacy trade-off down in the README, briefly: what holding a region buys and what it costs. I want the cost stated, not only the benefit.

# Claude

Now the holding behaviour. The tracker needs per-track age, so I'm introducing a track record:

Now the replay signal, so held regions can't bleed across a clip loop:

Existing tracker tests need an explicit budget — those cover matching and smoothing, so they get `hold_frames=0`:

Two uncovered lines — the replay reset. Adding tests for the counter and the window's response:

261 tests, 99.41%. Let me measure what holding actually buys on real camera data:

No drops in 150 frames of a well-lit stationary face — the model is reliable here, so a calm shot can't demonstrate it. Let me force the condition deterministically through the real pipeline:

Exactly the designed behaviour through the real pipeline:

| frame | model sees | blurred | radius | |
|---|---|---|---|---|
| 1–4 | 1 | 1 | 123 | detected |
| 5 | 0 | 1 | 138 | **held** |
| 6 | 0 | 1 | 155 | **held** |
| 7 | 0 | 1 | 173 | **held** |
| 8–11 | 1 | 1 | 158→135 | re-acquired, settling back |

Now the README section:

Done. QA green — 261 tests, 99.41%, `tracking.py` at 100%.

## Answering your question: what I do about a held face that moved

**I don't extrapolate, and that's deliberate.** A held region stays at the last position the face was actually *measured*. Instead, it **widens** by a fixed share of its radius per held frame (`HOLD_GROWTH_PER_FRAME`, 12%).

I rejected velocity extrapolation because the input doesn't support it. The centres we'd differentiate are already exponentially smoothed, so the velocity estimate lags reality; and a dropped detection often coincides with the face doing something unusual — turning, stopping, being occluded. If it stopped or reversed, extrapolating moves the cover *away* from the face, which is worse than not moving at all. Over a 3-frame budget (~0.1 s) there isn't enough time for extrapolation to earn back that risk.

Widening encodes the honest statement: *we don't know where it is, so cover more ground.* It fits the bias already in this codebase — over-covering only blurs background. It also has a useful side effect: the match allowance scales with radius, so a widened held region is easier to re-acquire when the face reappears slightly off from where it vanished.

It is still a guess, not a measurement. A fast-moving face can outrun it. That's stated plainly in the README rather than glossed.

## Verified through the real pipeline

Four good frames, a forced three-frame dropout, then recovery:

| frame | model sees | blurred | radius | |
|---|---|---|---|---|
| 1–4 | 1 | 1 | 123 | detected |
| 5 | 0 | 1 | 138 | **held** |
| 6 | 0 | 1 | 155 | **held** |
| 7 | 0 | 1 | 173 | **held** |
| 8–11 | 1 | 1 | 158 → 135 | re-acquired, settling back toward 123 |

Worth knowing: across 150 frames of you sitting still, the model dropped the face **zero** times. Holding is insurance against the hard cases, not a constant-benefit feature — which is why I had to force the condition to demonstrate it.

## The budget

`FaceTracker(hold_frames=...)` is a constructor parameter, with `DEFAULT_HOLD_FRAMES = 3` as a named constant the window passes in. Nothing in the matching or holding logic contains a frame count. Tested parametrically over budgets of 0, 1, 2, 5 and 10, including that 0 drops immediately. A re-match resets `held_frames` to 0 — tested by holding a face almost to expiry, finding it again, and asserting it then survives a *full* fresh budget.

## The clip loop

The tracker had no way to know a replay happened — the rewind is buried in `Camera.read()`. I added a `replays` counter on `Camera`, incremented only on a *successful* rewind, and the window resets the tracker when the count changes. So regions held from a clip's last frame can't bleed into the replay. Covered both ways: a replay resets, four ordinary frames don't. The counter also stays at 0 for a camera and for a clip that can't be replayed at all.

## Two things to flag

**"Configurable" — I read that as a named parameter, not a CLI flag.** The budget is settable at the tracker's edge and documented in the README, but there's no `--hold-frames`. Say the word and it's a few lines.

**Holding and smoothing interact.** On re-acquisition the region smooths back from its *widened* held size, so it takes a handful of frames to shrink to the true face (158 → 135 above, converging to 123). During that window the cover is larger than needed — the safe direction, but it's why the radius column doesn't snap straight back.

# User

Next bonus, first half: two more ways to hide a face.

Add pixelation and a solid fill alongside the existing blur. All three take the same region and do the same job, so they should be selectable by a single value rather than reached through branches scattered around the code — adding a fourth later should be one entry, not a new if.

Pixelation has the same trap the blur had: a fixed block size that looks solid on a distant face leaves a close-up one perfectly readable, so the blocks have to scale with the region.

Whatever each style does, the clipping and the empty-region handling stay in one place instead of being repeated three times.

No UI yet, that's the next commit; a parameter is enough for now. Test each style on a region and check it changes that region and nothing outside it.

# Claude

Let me look at the current structure so the shared clipping stays in one place:

Clipping and masking already live once, in `_hide` — I just need to pass the style in. Writing it:

Now the tests, parametrized across all three styles:

My blanket replace put the argument outside the call. Fixing and adding the per-style tests:

My replace also hit unrelated lines ending in `))`. Finding all the damage:

The window ordering test patches the old name and arity. Fixing:

Existing suite green on the new entry point. Now the per-style tests you asked for:

303 tests, `anonymization.py` at 100%. Let me see the three styles on a real face:

Let me confirm the block scaling numerically on two real face sizes, then check the live app:

Three styles in. QA green — 303 tests, 99.43%, `anonymization.py` at 100%.

## One value, no scattered branches

Each style is just a patch transform, `Callable[[Frame], Frame]`, and the selection happens exactly once — a dict lookup at the top of `hide_faces`:

There is no `if` on the style anywhere. Adding a fourth is a function plus one line in `STYLES` — and a test asserts `set(STYLES) == set(HideStyle)`, so a style added to the enum without an implementation fails the build rather than raising a `KeyError` on some unlucky frame.

`blur_faces` is now `hide_faces`, since the name was about to become a lie.

## Clipping stayed where it was

`_hide` already owned the clipped bounds, the off-frame early return and the circular mask. None of that moved or got duplicated — it just takes the transform as an argument now. All three styles reach it through the same path, which is why the clipping and empty-region tests parametrize cleanly across every style rather than being written three times.

## Pixelation and the trap

Same trick as the blur: a fixed block **count**, not a fixed block **size**. The patch is reduced to 8×8 and scaled back with `INTER_NEAREST` (nearest, not cubic — the hard edges *are* the effect). Block size therefore follows the region:

| frame | face radius | pixel block | blur sample |
|---|---|---|---|
| seated | 123 | 31 px | 41 px |
| close-up | 209 | 52 px | 70 px |

Two tests cover it: one that blocks exist and are wider than a pixel at any size, and one that measures the actual run lengths along a gradient row and checks the median matches `2·radius / PIXEL_BLOCKS` at two very different face sizes.

All three verified unrecognisable on a real detection — `fill` is a solid disc, and `detail_at` inside it is exactly 0.

## Tests

42 new. The core set is parametrized over `list(HideStyle)`, so every style gets: changes the region, leaves the rest of the frame alone (far corners untouched, plus bounds on which rows changed at all), makes the face unrecognisable, holds up at three face sizes, clips at four frame edges, ignores a fully off-frame region, and does nothing when given no regions. Plus style-specific checks for the fill colour and the pixel block geometry.

Live app re-verified: runs, detects, quits, camera released.

## One note

The style is a parameter as you asked — `hide_faces(frame, regions, style)` — and the window holds `self._hide_style = DEFAULT_HIDE_STYLE` (blur) as a plain attribute. That field is deliberately the only thing the next commit's menu needs to set; nothing else about the pipeline has to change to make the styles selectable.

# User

I don't like this here. could you find a better position for it?

# Claude

You're right — the dispatch table is the most important thing in that file and it's buried at the bottom. It's only there because it references functions defined above it.

The real fix is that `anonymization.py` is doing two jobs: deciding *where and how much* to conceal (geometry) and *what concealment looks like*. Splitting those lets the table sit at the top with the other constants, matching `REGION_COLOURS` in `rendering.py`:

Now mirroring the tests. The pixel-block tests get much cleaner against the transform directly:

Let me update the README's module layout and verify the app:

Moved. QA green — 321 tests, 99.43%, both modules at 100%.

## Where it went

`STYLES` is now at the **top** of `anonymization.py`, directly under the imports, next to the type alias — the same position `REGION_COLOURS` holds in `rendering.py` and `KEYPOINT_REGIONS` holds in `models.py`.

It could only sit at the bottom because it referenced functions defined above it. Rather than shuffle the file and break the callers-before-callees ordering, I moved the three transforms into `vision/styles.py`. They're already defined by the time `anonymization.py` imports them, so the table reads first:

## Why this is the right split, not just a relocation

The two modules were answering different questions:

- **`anonymization.py`** — *where and how much to conceal*: clipped bounds, the circular mask, iterating regions, dispatch. 32 statements.
- **`styles.py`** — *what concealment looks like*: the three transforms and their tuning constants (`BLUR_SAMPLES`, `PIXEL_BLOCKS`, `FILL_COLOUR`). 16 statements.

Both now read top-down without compromise, and `hide_faces` is the first thing you see in the file that does the hiding.

## Test tree followed

`vision/styles.py` gets `tests/vision/test_styles.py`, per the mirror rule. This turned out better than a pure move: the pixel-block tests were previously measuring run lengths through `hide_faces` on a checkerboard, fighting the circular mask to find block edges. Against `pixelated()` directly they're exact — median block width is asserted as `size / PIXEL_BLOCKS` across three patch sizes, plus a check that a 320px patch gets blocks more than 3× a 80px one.

New there too: every style preserves shape and dtype, destroys detail, and — worth having — **doesn't write through its input**, since `_hide` relies on getting a fresh array back to composite through the mask.

`test_anonymization.py` keeps what it should: dispatch completeness, clipping, off-frame, and the cross-style behaviour parametrized over `HideStyle`.

README module layout updated — it was also missing `tracking.py` from the earlier commit.

# User

Second half: pick the style from the toolbar.

Put the three styles in the Settings menu as an exclusive choice, and add a toggle for the skeleton overlay next to it. The app still starts on blur.

A change takes effect on the next frame — don't rebuild the pipeline or restart anything. The menu has to show what is currently active, so I can tell at a glance which style is on and whether the overlay is being drawn.

This is the first piece of state the UI writes and the frame loop reads. Keep it in one place rather than threading flags through every call.

The tests shouldn't need a visible window.

# Claude

Let me look at the current menu construction and the frame loop:

Now the shared state object the UI writes and the loop reads:

`triggered` tolerates a zero-arg slot (as `_choose_camera` already proves), so the unused parameter can just go:

All 322 still pass. Now the tests — none of them open a window:

339 tests, 99.47%. Let me prove the suite genuinely needs no display:

All 339 pass with `DISPLAY`, `WAYLAND_DISPLAY` and `XDG_SESSION_TYPE` unset — the five `.show()` calls are pre-existing visibility assertions and work offscreen. Now driving the real menu to see each style live:

The montage is off by one — each snapshot fires before the next frame renders, so "pixelate" shows the previous blur frame. That's my driver, not the app (it does literally demonstrate "next frame"). Re-capturing with a gap after each trigger:

All four states confirmed live. QA green — 339 tests, 99.47%.

## The one place the state lives

`ui/settings.py`:

The menu writes it in place, the frame loop reads it. No flags are threaded through anything — `hide_faces` already took a style, and the skeleton is a read at the call site (`if self._settings.draw_skeleton: draw_poses(...)`) rather than a bool parameter, per the project's own rule about booleans that drive an `if`.

It's deliberately **mutable**, which is why I put it in `ui/` rather than `models.py`: every type in `models.py` is a frozen value object, and dropping a mutable one in there would quietly break that invariant. Flagging it in case you'd rather it sat with the rest.

## Taking effect on the next frame

There's nothing to rebuild — the loop re-reads the object every frame. Tested directly: render a frame, trigger a style, render again, assert the pipeline received `BLUR` then the new style. Parametrized over all three.

There's also a test that changing the style leaves `window._session.camera` and `window._tracker` as the *same objects* and releases nothing, so a style change can't quietly restart the source or drop tracking history.

My first capture run accidentally proved the timing: I snapshotted immediately after each trigger and got the *previous* frame every time. Re-captured with a gap — the montage above shows blur, pixelate, fill, and fill-with-overlay-off.

## The menu

Styles are an exclusive `QActionGroup`, the overlay is a plain checkable action, and both show their state. The style entries are **built by iterating `HideStyle`**, so a fourth style appears in the menu automatically — it only needs a label, and `test_every_style_has_a_label_for_the_menu` asserts `set(STYLE_LABELS) == set(HideStyle)` so a missing one fails the build rather than raising a `KeyError` mid-frame. That matches the `STYLES` completeness test from the first half.

Starts on blur with the overlay on, from `DEFAULT_HIDE_STYLE` and `DEFAULT_DRAW_SKELETON`.

## Tests

20 new, **none of them call `.show()`**. To be sure, I ran the whole suite with `DISPLAY`, `WAYLAND_DISPLAY` and `XDG_SESSION_TYPE` all unset — 339 passed. The five pre-existing `.show()` calls are visibility assertions from earlier commits and work offscreen too.

Coverage includes: the menu lists every style in enum order, the default is blur and is the only one checked, choosing any style is exclusive, the choice reaches the next frame, the overlay starts on, switching it off stops `draw_poses` being called, switching it back on resumes it, and — worth having — faces keep being hidden while the overlay is off, so the two settings can't be accidentally coupled.

# User

To the models.py

# Claude

Docstring stays out, then. Moving it in next to the other config types:

Now the tests follow the module — `models.py` is tested by `tests/test_models.py`:

Test count dropped by one (339 → 338). Let me check nothing was lost in the merge:

Mystery solved — nothing was lost. `test_architecture.py` parametrizes over every module in the package, so deleting `ui/settings.py` removed one parametrized case (23 → 22). All three settings tests moved intact into `tests/test_models.py` (26 tests there now).

Moved. QA green — 338 tests, 99.47%.

`ViewSettings` now sits in `models.py` directly after `AppConfig`, with the other configuration types and within reach of `DEFAULT_HIDE_STYLE` / `DEFAULT_DRAW_SKELETON`. `ui/settings.py` is gone, so `ui/` is back to `window.py` and `framerate.py`.

I left the docstring off — you'd removed it from my version, so I didn't reintroduce it on the move.

One thing worth noting since it's now visible in `models.py`: this is the only `@dataclass(slots=True)` in the file; everything around it is `frozen=True`. That's deliberate — the menu mutates it in place and the frame loop reads it — but it does mean `models.py` is no longer uniformly immutable.

Tests followed the module into `tests/test_models.py` (26 tests there now), and `tests/ui/test_settings.py` is deleted.

On the count going 339 → 338: nothing was lost. `test_architecture.py` parametrizes its ML-import check over every module in the package, so removing `ui/settings.py` removed exactly one parametrized case (23 → 22). I verified the per-file collection to confirm all three settings tests survived the move.

Live app re-verified: runs, detects, quits, camera released. The first quit attempt failed on X focus contention again — second attempt landed.

# User

Next bonus, first commit: record the processed video.

Add an entry to the Settings menu that starts writing frames to an mp4, off by default. What goes into the file is what the window shows — faces hidden, skeleton and panel drawn — not the raw capture.

Two things to get right. The frame size isn't known before the first frame arrives, and it changes if I switch to a clip with a different resolution. And the file has to be closed on every way out: quitting with the key, closing the window, the app failing — an mp4 whose writer was never released won't play.

Recording stays off unless I turn it on, and turning it off closes the file cleanly.

Verify it by recording a few seconds and playing the file back. I want to see the blur and the skeleton in it.

# Claude

Let me start with the recorder itself — lazy open on the first frame, and a close that's safe to call repeatedly:

Now wiring it into the window and guaranteeing closure on every exit path:

Existing suite green. Now the tests for the new code:

Recorder covered. Now the menu entry and the lifecycle guarantees:

Adding the main-level guarantee test, then recording for real:

361 tests, 99.42%. Now recording for real from the camera and playing it back:

Plays back cleanly — 158 frames, 30.3 fps, 640×480, 5.2 s, with blur, skeleton and panel in every sampled frame. Now the resolution-change case, live:

Two stray 1.7 K mp4s in the project root — my `_discard` helper reads `destination` *after* `close()` has already cleared it, so it never deleted anything. Tests shouldn't write into the working directory at all:

One still appeared. Let me find which test writes it:

Recording works. QA green — 361 tests, 99.42%, and the full suite now leaves no stray files.

## Verified by playing it back

Recorded ~6 s from the live app, then decoded the file:

Four frames sampled across the clip — all show the **blurred face, the skeleton, and the stats panel**. It's the composed frame, not the raw capture. The file is 1.4 MB at `evidone-20261101-201049.mp4` in the project root if you want to open it yourself (`*.mp4` is now gitignored).

A test pins this independently: the fake capture delivers a uniform frame, and the recorded frame is asserted to be non-uniform — so recording the raw capture would fail. A second test asserts the call order is exactly `["hide", "draw", "stats", "record"]`.

## The two things you flagged

**Size unknown until the first frame.** `start()` only records the destination; the `VideoWriter` opens inside the first `write()`, using that frame's dimensions. Tested: after `start()` the file does not yet exist, and the file that eventually appears carries the first frame's size.

**Resolution changing mid-recording.** An mp4 can't change size part way through, so later frames are **letterboxed** into the size the file opened with. Verified live rather than only in tests: started recording on the 640×480 camera, switched to a 320×240 clip while recording, and the result is one continuous playable file — 130 frames, `sizes={(640, 480)}`, with the clip centred and padded.

I chose letterboxing over the alternatives deliberately: distorting would change people's proportions in a privacy recording, and splitting into a second file turns one recording into several. Worth knowing it's lossy in the other direction — switching to a *higher*-resolution source downsamples into the original size.

## Closing on every exit path

| Exit | Closed by |
|---|---|
| Menu toggle off | `_stop_recording()` |
| `q` / `Esc` | `close()` → `closeEvent` → `shutdown()` |
| Window closed | same |
| Event loop raises | `run()`'s `finally: window.shutdown()` |

`shutdown()` is new and idempotent, and `Recorder.close()` is safe to call twice or having never started. The crash path has its own test: a `QApplication` whose `exec()` raises, asserting `shutdown` still ran and the camera was still released.

If the writer can't be opened at all — bad folder, missing codec — `RecordingError` surfaces on the first frame, recording switches itself off, the menu unticks, a dialog explains, and the frame loop keeps running.

## One fix to my own tests

My first cut left 1.7 KB mp4s in the project root: the `_discard` helper read `recorder.destination` *after* `close()` had already cleared it, so it silently deleted nothing. Replaced with a `recordings` fixture that patches `_recording_path` to a `tmp_path`, keeping the real code path intact. Confirmed a full-suite run now creates zero stray files.

# User

Second commit: a per-frame log of what the model saw. Add a Settings entry that starts writing a JSON Lines file, off by default. One object per line, per frame: which frame it is, and for each person their box, their detection score and all 17 keypoints with their confidences.

JSON Lines rather than one big array, so it streams, can be read line by line, and is still useful if the run is cut short. Each line saying which frame it is matters — I want to be able to line the log up against a recording of the same session.

Log what the model produced and let whoever reads it filter; don't silently drop the low confidence points.

Same lifetime rules as the recording: off unless I turn it on, closed on every way out. An unclosed file loses whatever is still buffered, which is exactly the tail I'd be looking at.

# Claude

Noting one conflict up front: AGENTS.md says serialise JSON with `indent=2`, which would break one-object-per-line. JSON Lines wins — flagging it rather than silently deviating.

Existing suite green. Now the tests:

The `install_captures` fake patches the shared `cv2` module, so my own `cv2.VideoCapture` call in the test picked it up. Asserting the invariant without reopening the file:

389 tests, 99.36%. Now a real session — recording and logging together, then checking they line up:

Detection logging works. QA green — 389 tests, 99.36%, suite leaves no stray files.

## Verified on a real session

Recorded and logged together, then checked them against each other:

The streaming property is proved, not assumed: I read the file *mid-session* from a separate code path and got 106 complete lines back. `wc -l` works on it too. The file is line-buffered, so each finished line reaches disk as it's written — a hard kill loses at most the line in flight, not the tail.

## Low-confidence points are kept

**Over half** the logged keypoints would have been invisible in the overlay. They're all in the file. The log is written straight after `estimate()`, before tracking, smoothing or the visibility threshold touch anything — so it's the model's output, and filtering is the reader's call.

One line, as stored:

## Lining up with a recording

A frame with **nobody in it still gets a line** (`"people": []`). That's what keeps the numbering in step — skipping empty frames would silently desynchronise the log from the video exactly when someone walks out of shot. There's a test for it, plus one asserting the log's line count equals the recorder's frame count over the same run.

Both counters advance once per processed frame, so starting the two together gives line *n* ↔ video frame *n*. Worth knowing: starting them at different moments leaves a constant offset — each counter is per-file, from 0.

## Lifetime

Same rules as recording: off by default, `shutdown()` closes both, reached from `closeEvent` (so `q`, `Esc` and the window button) and from `run()`'s `finally` when the event loop raises. `close()` is safe twice or never-started, and a bad destination reports in a dialog and leaves the entry unticked.

One difference I chose deliberately: the log opens **eagerly** on start, unlike the recorder. The recorder has to wait for a frame to learn its size; the log doesn't need anything, so opening immediately means a bad path fails at the moment you click rather than a frame later.

## A convention I broke

`AGENTS.md` says serialise JSON with `indent=2`. That's incompatible with one-object-per-line — indented JSON spans many lines and the format stops being JSON Lines. I used compact `json.dumps` per line. Flagging it as a deliberate deviation rather than an oversight; if you want it recorded, `AGENTS.md` could note that JSONL is the exception.

# User

Refactor: pull the configuration out of the code.

Every tunable value that currently sits as a module-level constant moves into a single `config.toml`, grouped into sections by area. The modules keep behaviour and nothing else.

The file has to ship with the installed package, not just work from a checkout: `uv run evidone` has to find it after a plain install too.

Load it once and expose it as typed objects rather than a raw dictionary. I want `CONFIG.panel.alpha` checked by mypy, not `CONFIG["panel"]["alpha"]` blowing up at runtime. The dataclasses describing it belong with the rest of the dataclasses. A missing or unknown key should fail when the app starts, not in the middle of the frame loop.

Some constants aren't configuration: unit conversions, framework objects, tables that describe structure rather than taste. Move what belongs in a config file and give me a list of what you left behind and why. Don't quietly move everything just because I said all of them, and don't quietly leave something out either.

# Claude

Let me inventory every module-level constant before deciding what is and isn't configuration:

A thorough inventory. Let me write the config file first, with the sections grouped by area:

Now the typed dataclasses, in `models.py` with the rest:

Now the loader — strict about both missing and unknown keys:

Loader works. Now rewiring each module to read `CONFIG` and dropping its constants:

Source modules are clean. Now `window.py` and the tests:

Source is clean. Now updating the tests to read from `CONFIG`:

My import-stripping only handled multi-line blocks. Cleaning up the single-line ones:

Only the architecture test fails — it asserted `VISIBILITY_THRESHOLD` is defined once in code, and it now lives in the TOML. Replacing it with a check that enforces the new rule:

Four constants remain in `models.py` — and they can't read `CONFIG` because `config.py` imports `models.py`. Let me see what depends on them:

411 pass, and `models.py` now holds only structural constants. Adding tests for the loader's strictness:

438 tests, 99.14%. Now the packaging requirement — does it survive a plain install?

The installed console script finds its config with no checkout present. Now verifying the live app and that a bad config fails at startup:

The app is identical in behaviour, now driven entirely from `config.toml`. QA green — 438 tests, 99.14%.

## What moved

`src/evidone_pose_detection/config.toml`, 10 sections: `[camera] [pose] [faces] [tracking] [hiding] [skeleton] [panel] [window] [output] [labels]`. Typed as frozen dataclasses in `models.py` (`CameraSettings`, `PanelSettings`, …, `Config`), loaded once by `config.py` into `CONFIG`.

`CONFIG.panel.alpha` type-checks. mypy caught several of my own mistakes during this refactor, including a regex that rewrote the left-hand side of a constant into `CONFIG.window.frame_rate_window = 15`.

Validation is strict in both directions — a missing section, a missing key, an unknown key, an unknown section, an unknown enum key (`skeleton.colours.tail`), an unknown style name, and an unknown keypoint in a span all raise `ConfigError`. Verified live: adding `brightness = 3` under `[panel]` gives

at startup, before a window exists. 27 loader tests cover these paths.

## Shipping

`config.toml` is read through `importlib.resources`, not a filesystem path. Verified end to end: built the wheel (`config.toml` is in it next to `py.typed`), installed it into a clean venv with no checkout in sight, and ran the console script:

## What I left behind, and why

| Stayed | Where | Why |
|---|---|---|
| `SKELETON` | `models.py` | The published COCO-17 topology. A test asserts it equals the canonical pair list — it's a fact, not a preference. |
| `KEYPOINT_REGIONS` | `models.py` | Which body region each keypoint anatomically belongs to. Structure. The *colours* those regions are drawn in did move. |
| `FACE_LANDMARKS` | `models.py` | Which COCO keypoints are on a face. Anatomy. |
| `KEYPOINT_COUNT` | `models.py` | Derived (`len(COCOKeypoint)`), not a value anyone sets. |
| `QUIT_KEYS` | `ui/window.py` | Qt key enum objects — framework objects, not data. |
| `FONT` | `vision/overlay.py` | `cv2.FONT_HERSHEY_SIMPLEX`, an OpenCV constant. Its *scale and thickness* did move. |
| `STYLES` | `vision/anonymization.py` | Maps an enum to functions. Code dispatch, not data. |
| `USAGE/MODEL/CAMERA_EXIT_CODE` | `main.py` | A protocol contract with the shell. `2` is argparse's, not ours. Changing these silently breaks callers. |
| `PROGRAM = "evidone"` | `main.py` | The stderr prefix, conventionally the executable name, tied to the `[project.scripts]` entry. Not taste. |
| `CONFIG_FILE`, `SECTIONS` | `config.py` | The loader's own structure — can't live in the file it loads. |

Two judgement calls worth challenging:

**`FACE_SPANS` moved, pairs and all.** The scales are pure taste, but they're inseparable from which landmarks they measure — eight entries that together *are* the face-sizing heuristic, unlike `SKELETON` which is published data. They're now `[[faces.spans]]` with keypoint names resolved at load. If you'd rather the pairs stayed in code, that's a reasonable different call.

**UI labels moved.** Wording is presentation taste, so `[labels]` holds all of it including dialog titles and the file filter. If you consider UI copy to be code, that's the section to pull back.

## Two consequences worth knowing

**`Keypoint.is_visible` is gone**, replaced by `vision/visibility.py: is_visible(keypoint)`. The property read `VISIBILITY_THRESHOLD`, and once that moved to the config, `models.py` would have had to import `config.py`, which imports `models.py`. A cycle. The predicate was behaviour sitting on a data class, so it moved to the vision layer. `ViewSettings.defaults()` went the same way — the window now builds it from `CONFIG` directly.

**A bad config exits 1 with a traceback**, not a formatted one-liner like the camera and model errors. The load happens at import time, so `main()` can't catch it — the message is precise and arrives before anything starts, but it isn't dressed. Giving it a clean message and its own exit code means a small launcher shim around the entry point; say the word and I'll add it.

`test_architecture.py` gained a parametrized test asserting no module re-declares any of the 35 moved names, so a constant can't quietly creep back into the code.

# User

My session transcript for this project is at

 ~/.claude/projects/-home-casy2210-Projects-evidone-pose-detection/1a61a56d-9b54-4ed4-9bc3-18f929c79a83.jsonl

I want a readable Markdown version of the conversation to share, without the noise. Keep only my messages and your prose replies, in order, each labelled with who is speaking.Drop everything else: tool calls and their results, file contents, diffs, code blocks, command output, images, thinking blocks. Reproduce the text verbatim. Don't summarise, rewrite or tidy up what either of us said — if it's paraphrased it isn't a transcript any more.

# Claude

Let me look at the transcript's structure before extracting anything.

All 22 user messages are intact and free of injected reminders. Let me check what's inside my replies before deciding how to strip them:

Now the extractor. Order preserved, fenced code removed, everything else verbatim:

44 turns, perfectly alternating. Let me verify the text is actually verbatim rather than trusting the script:

Those two lines are from *this* turn — written after extraction. Expected. Let me confirm no tool noise leaked and check the shape:

My own replies use `##` headings, which collide with the speaker labels. Checking whether `#` is free:

`#` is free — the three h1s were TOML comments inside code blocks, already stripped. Switching speaker labels to `#` with a rule between turns:
