"""
detect running and jumping in webcam/video using Mediapipe PoseLandmarker
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import sys
import os
import time
import urllib.request

# ── MediaPipe task imports ───────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Model ────────────────────────────────────────────────────────────
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pose_landmarker_lite.task")

MAX_POSES = 2                
MIN_DETECTION_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

HISTORY_LEN = 60                    

RUN_WINDOW = 25                   
RUN_MIN_ZERO_CROSSINGS = 3      
RUN_MIN_AMPLITUDE = 0.003       
RUN_MIN_FREQ_HZ = 0.8            
RUN_MAX_FREQ_HZ = 5.5             
RUN_REGULARITY_CV = 1.0       
RUN_HIP_VISIBILITY = 0.35         
RUN_CONFIRM_FRAMES = 2           
RUN_DROP_FRAMES = 6          

JUMP_VEL_THRESHOLD = -0.003        
JUMP_DISP_THRESHOLD = 0.035        
JUMP_COOLDOWN_FRAMES = 8     
JUMP_BASELINE_WINDOW = 20         
JUMP_LABEL_HOLD = 6           
JUMP_SHOULDER_COHERENCE = 0.05   

MATCH_DIST = 0.20                
MISSING_TIMEOUT = 20            


class PersonState:
    """Maintains landmark history and classification state for one person."""

    _next_id = 0

    def __init__(self):
        self.id = PersonState._next_id
        PersonState._next_id += 1

        # Landmark history buffers (normalised coordinates, 0-1)
        self.torso_y = deque(maxlen=HISTORY_LEN)     
        self.torso_x = deque(maxlen=HISTORY_LEN)    
        self.lsh_y = deque(maxlen=HISTORY_LEN)   
        self.rsh_y = deque(maxlen=HISTORY_LEN)   
        self.hip_y = deque(maxlen=HISTORY_LEN)       
        self.bounce_y = deque(maxlen=HISTORY_LEN)   

        # Classification state
        self.is_running = False
        self.is_jumping = False
        self._jump_cooldown = 0
        self._jump_label_hold = 0
        self._run_confirm_count = 0
        self._run_drop_count = 0

        # Tracking helpers
        self.last_center: tuple[float, float] | None = None
        self.last_landmarks = None
        self.frames_missing = 0

    def update(self, lms):
        """Push one frame of landmarks into the history buffers.

        `lms` is a list of NormalizedLandmark from Mediapipe PoseLandmarker.
        """
        lsh, rsh = lms[11], lms[12]       
        lhip, rhip = lms[23], lms[24]    

        tx = (lsh.x + rsh.x) / 2.0
        ty = (lsh.y + rsh.y) / 2.0

        self.torso_x.append(tx)
        self.torso_y.append(ty)
        self.lsh_y.append(lsh.y)
        self.rsh_y.append(rsh.y)

        if (lhip.visibility > RUN_HIP_VISIBILITY
                and rhip.visibility > RUN_HIP_VISIBILITY):
            hip_mid_y = (lhip.y + rhip.y) / 2.0
            self.hip_y.append(hip_mid_y)
            self.bounce_y.append(hip_mid_y)
        else:
            self.hip_y.append(None)
            self.bounce_y.append(ty)         

        self.last_center = (tx, ty)
        self.last_landmarks = lms
        self.frames_missing = 0

    def check_running(self, fps: float):
        """Detect running via rhythmic vertical torso/hip bounce.

        When running, the body bobs up and down with each stride.
        We take the best-available vertical signal (hip midpoint if
        visible, else shoulder midpoint), detrend it, then check:
          1. Amplitude is above threshold (not just noise).
          2. Zero-crossing frequency is in the running range.
          3. Crossings are roughly evenly spaced (rhythmic).

        A small hysteresis (confirm / drop counters) prevents
        the label from flickering on and off.
        """
        recent = list(self.bounce_y)[-RUN_WINDOW:]
        recent = [v for v in recent if v is not None]
        n = len(recent)
        if n < int(RUN_WINDOW * 0.6):
            self._run_drop_count += 1
            if self._run_drop_count >= RUN_DROP_FRAMES:
                self.is_running = False
            return

        sig = np.array(recent, dtype=np.float64)

        sig -= np.linspace(sig[0], sig[-1], n)

        amplitude = sig.max() - sig.min()
        if amplitude < RUN_MIN_AMPLITUDE:
            self._run_drop_count += 1
            if self._run_drop_count >= RUN_DROP_FRAMES:
                self.is_running = False
                self._run_confirm_count = 0
            return

        centred = sig - sig.mean()
        sign_changes = np.where(np.diff(np.sign(centred)))[0]
        n_crossings = len(sign_changes)

        if n_crossings < RUN_MIN_ZERO_CROSSINGS:
            self._run_drop_count += 1
            if self._run_drop_count >= RUN_DROP_FRAMES:
                self.is_running = False
                self._run_confirm_count = 0
            return

        duration_s = n / max(fps, 1)
        freq_hz = (n_crossings / 2.0) / duration_s

        if not (RUN_MIN_FREQ_HZ <= freq_hz <= RUN_MAX_FREQ_HZ):
            self._run_drop_count += 1
            if self._run_drop_count >= RUN_DROP_FRAMES:
                self.is_running = False
                self._run_confirm_count = 0
            return

        if n_crossings >= 3:
            intervals = np.diff(sign_changes).astype(float)
            cv = intervals.std() / (intervals.mean() + 1e-9)
            if cv > RUN_REGULARITY_CV:
                self._run_drop_count += 1
                if self._run_drop_count >= RUN_DROP_FRAMES:
                    self.is_running = False
                    self._run_confirm_count = 0
                return

        self._run_drop_count = 0
        self._run_confirm_count += 1
        if self._run_confirm_count >= RUN_CONFIRM_FRAMES:
            self.is_running = True

    def check_jumping(self, fps: float):
        """Detect jumping via upward torso rise + landing confirmation.

        Checks:
          1. Rolling baseline of shoulder-midpoint Y.
          2. Both shoulders must rise.
          3. Upward velocity must exceed threshold.
          4. Upward displacement from baseline must exceed threshold.
        """
        if self._jump_label_hold > 0:
            self._jump_label_hold -= 1
            self.is_jumping = True
            return
        if self._jump_cooldown > 0:
            self._jump_cooldown -= 1
            self.is_jumping = False
            return

        n = len(self.torso_y)
        if n < JUMP_BASELINE_WINDOW:
            self.is_jumping = False
            return

        ty = np.asarray(self.torso_y)

        baseline_end = max(n - 6, JUMP_BASELINE_WINDOW // 2)
        baseline_start = max(0, baseline_end - JUMP_BASELINE_WINDOW)
        baseline = ty[baseline_start:baseline_end].mean()

        current = ty[-3:].mean()              
        displacement = current - baseline       

        if n >= 5:
            velocity = np.diff(ty[-5:]).mean()            
        else:
            self.is_jumping = False
            return

        # both shoulders rise
        if len(self.lsh_y) >= 5 and len(self.rsh_y) >= 5:
            lsh_dy = np.diff(np.array(list(self.lsh_y)[-5:])).mean()
            rsh_dy = np.diff(np.array(list(self.rsh_y)[-5:])).mean()
            if abs(lsh_dy - rsh_dy) > JUMP_SHOULDER_COHERENCE:
                self.is_jumping = False
                return
            if lsh_dy > 0.001 or rsh_dy > 0.001:
                self.is_jumping = False
                return

        if displacement < -JUMP_DISP_THRESHOLD and velocity < JUMP_VEL_THRESHOLD:
            self.is_jumping = True
            self._jump_cooldown = JUMP_COOLDOWN_FRAMES
            self._jump_label_hold = JUMP_LABEL_HOLD
            return

        self.is_jumping = False


def match_poses(
    existing: list[PersonState],
    new_poses: list[list],
) -> list[tuple[PersonState, list]]:
    """Greedy nearest-neighbour matching on torso centres.

    Returns (person, landmarks) pairs. New PersonState objects are
    created for unmatched detections; existing persons that weren't
    matched get their ``frames_missing`` counter bumped.
    """
    centres = []
    for lms in new_poses:
        cx = (lms[11].x + lms[12].x) / 2.0
        cy = (lms[11].y + lms[12].y) / 2.0
        centres.append((cx, cy))

    used_p, used_n = set(), set()
    pairs: list[tuple[float, int, int]] = []

    for i, p in enumerate(existing):
        if p.last_center is None:
            continue
        for j, (cx, cy) in enumerate(centres):
            d = np.hypot(cx - p.last_center[0], cy - p.last_center[1])
            pairs.append((d, i, j))
    pairs.sort()

    matched: list[tuple[PersonState, list]] = []
    for d, pi, nj in pairs:
        if pi in used_p or nj in used_n:
            continue
        if d < MATCH_DIST:
            matched.append((existing[pi], new_poses[nj]))
            used_p.add(pi)
            used_n.add(nj)

    for j, lms in enumerate(new_poses):
        if j not in used_n:
            matched.append((PersonState(), lms))

    for i, p in enumerate(existing):
        if i not in used_p:
            p.frames_missing += 1

    return matched

_SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  
    (11, 23), (12, 24), (23, 24),                        
    (23, 25), (25, 27), (24, 26), (26, 28),             
    (0, 11), (0, 12),                                    
]
_KEY_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

_COLORS = {
    "running": (0, 180, 255),    
    "jumping": (0, 255, 100),  
    "idle":    (160, 160, 160), 
}


def draw_skeleton(frame, lms, h, w, color=(255, 255, 255)):
    for a, b in _SKELETON:
        if a < len(lms) and b < len(lms):
            la, lb = lms[a], lms[b]
            if la.visibility > 0.45 and lb.visibility > 0.45:
                p1 = (int(la.x * w), int(la.y * h))
                p2 = (int(lb.x * w), int(lb.y * h))
                cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
    for idx in _KEY_JOINTS:
        if idx < len(lms) and lms[idx].visibility > 0.45:
            pt = (int(lms[idx].x * w), int(lms[idx].y * h))
            cv2.circle(frame, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)


def draw_labels(frame, person: PersonState, lms, h, w):
    """Draw person ID and activity label above the shoulders."""
    cx = int((lms[11].x + lms[12].x) / 2.0 * w)
    cy = int((lms[11].y + lms[12].y) / 2.0 * h)
    shoulder_width = int(abs(lms[12].x - lms[11].x) * w)
    label_y = max(40, cy - shoulder_width - 20)

    # Collect status tags
    tags: list[tuple[str, tuple]] = []
    if person.is_running:
        tags.append(("RUNNING", _COLORS["running"]))
    if person.is_jumping:
        tags.append(("JUMPING", _COLORS["jumping"]))
    if not tags:
        tags.append(("IDLE", _COLORS["idle"]))

    # Person ID
    cv2.putText(frame, f"P{person.id}", (cx - 12, label_y - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                cv2.LINE_AA)

    for i, (text, colour) in enumerate(tags):
        y = label_y + i * 32
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        x1 = cx - tw // 2 - 6
        cv2.rectangle(frame, (x1, y - th - 6), (x1 + tw + 12, y + 6),
                      colour, -1)
        cv2.putText(frame, text, (x1 + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2,
                    cv2.LINE_AA)


def ensure_model():
    """Download the PoseLandmarker model if it isn't already present."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"Downloading pose model → {MODEL_PATH} …")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.\n")



class PoseDetector:
    """Reusable wrapper around MediaPipe PoseLandmarker for multi-person
    running/jumping detection.

    Usage::

        detector = PoseDetector()
        # in your frame loop:
        persons = detector.process_frame(bgr_frame, frame_index)
        for p in persons:
            print(p.id, p.is_running, p.is_jumping)
        detector.close()
    """

    def __init__(self, max_poses=MAX_POSES, fps=30.0):
        ensure_model()
        self.fps = fps
        self._persons: list[PersonState] = []

        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            num_poses=max_poses,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def process_frame(self, frame_bgr, frame_idx: int) -> list[PersonState]:
        """Run detection on a BGR frame and return active PersonState list.

        Each PersonState has up-to-date ``is_running`` and ``is_jumping``
        flags, plus an ``id`` and ``frames_missing`` counter.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(frame_idx * 1000.0 / self.fps)

        result = self._landmarker.detect_for_video(mp_img, ts_ms)

        if result.pose_landmarks:
            matches = match_poses(self._persons, result.pose_landmarks)
            active: list[PersonState] = []

            for person, lms in matches:
                person.update(lms)
                person.check_running(self.fps)
                person.check_jumping(self.fps)
                active.append(person)

            matched_ids = {id(m[0]) for m in matches}
            for p in self._persons:
                if id(p) not in matched_ids and p.frames_missing < MISSING_TIMEOUT:
                    active.append(p)

            self._persons = active
        else:
            for p in self._persons:
                p.frames_missing += 1
            self._persons = [p for p in self._persons
                             if p.frames_missing < MISSING_TIMEOUT]

        return list(self._persons)

    @property
    def visible_count(self) -> int:
        """Number of persons currently visible (not missing)."""
        return sum(1 for p in self._persons if p.frames_missing == 0)

    def close(self):
        self._landmarker.close()


class SinglePoseDetector:
    """Lightweight single-person pose detector using IMAGE mode.

    Designed for the YOLO-crop workflow: crop each person's bounding box
    from the frame, then call ``detect_crop()`` on each crop individually.
    IMAGE mode means no timestamp tracking is required, and each call is
    fully independent.

    Usage::

        spd = SinglePoseDetector()
        landmarks = spd.detect_crop(cropped_bgr_image)
        if landmarks:
            person_state.update(landmarks)
        spd.close()
    """

    def __init__(self):
        ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            num_poses=1,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def detect_crop(self, crop_bgr):
        """Run pose detection on a single cropped BGR image.

        Returns: list of NormalizedLandmark, or None if no pose found.
        Coordinates are normalised to the crop dimensions (0-1).
        """
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_img)
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            return result.pose_landmarks[0]
        return None

    def close(self):
        self._landmarker.close()

def main():
    # parse input source
    source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.isdigit():
            source = int(arg)
        elif os.path.isfile(arg):
            source = arg
        else:
            print(f"File not found: {arg}")
            sys.exit(1)

    ensure_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_live = isinstance(source, int)

    print(f"Source : {'webcam ' + str(source) if is_live else source}")
    print(f"Frame  : {w}×{h} @ {fps:.0f} fps")
    print(f"Poses  : up to {MAX_POSES}")
    print("Press Q / ESC to quit.\n")

    # create landmarker
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        running_mode=mp_vision.RunningMode.VIDEO,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    # state
    persons: list[PersonState] = []
    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    continue
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000.0 / fps)

            result = landmarker.detect_for_video(mp_img, ts_ms)

            if result.pose_landmarks:
                matches = match_poses(persons, result.pose_landmarks)
                active: list[PersonState] = []

                for person, lms in matches:
                    person.update(lms)
                    person.check_running(fps)
                    person.check_jumping(fps)
                    active.append(person)

                    draw_skeleton(frame, lms, h, w)
                    draw_labels(frame, person, lms, h, w)

                matched_ids = {id(m[0]) for m in matches}
                for p in persons:
                    if id(p) not in matched_ids and p.frames_missing < MISSING_TIMEOUT:
                        active.append(p)

                persons = active
            else:
                for p in persons:
                    p.frames_missing += 1
                persons = [p for p in persons if p.frames_missing < MISSING_TIMEOUT]

            # HUD
            elapsed = time.time() - t0
            live_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            visible = sum(1 for p in persons if p.frames_missing == 0)
            running = sum(1 for p in persons if p.is_running)
            jumping = sum(1 for p in persons if p.is_jumping)

            hud = [
                f"FPS: {live_fps:.0f}",
                f"People: {visible}",
            ]
            if running:
                hud.append(f"Running: {running}")
            if jumping:
                hud.append(f"Jumping: {jumping}")

            for i, line in enumerate(hud):
                cv2.putText(frame, line, (10, 28 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                            cv2.LINE_AA)

            cv2.imshow("Body Motion Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        print(f"\nProcessed {frame_idx} frames.")


if __name__ == "__main__":
    main()
