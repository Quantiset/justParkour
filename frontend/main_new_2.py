"""
Dance-game frontend with live webcam pose-detection overlay.

States:
  INTRO     – "Stand in frame!" prompt; waiting for ≥1 person.
  COUNTDOWN – 5-second countdown; player count locked at end.
  PLAYING   – Avatar HUD with action circles over the video.
"""

import pygame
import cv2
import numpy as np
import time
import os
import sys
import json
import math

# ── Allow importing body.py from the project root ──────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from body_new import PoseDetector, draw_skeleton  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────
WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 30
AVATAR_SIZE   = 80
COUNTDOWN_SECONDS = 5
CIRCLE_RADIUS = 14
SIDEBAR_WIDTH = 220

DEBUG_W = WINDOW_WIDTH  // 4
DEBUG_H = WINDOW_HEIGHT // 4

# ── Colours ─────────────────────────────────────────────────────────
DARK   = (18, 18, 24)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREEN  = (0, 200, 80)
ORANGE = (255, 160, 0)
GOLD   = (255, 215, 0)
BLUE   = (0, 150, 255)
PINK   = (255, 0, 200)
SIDEBAR_BG = (20, 20, 30, 180)

CUE_COLORS = [GREEN, BLUE, PINK, ORANGE]

# ── Pygame init ─────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dance Game – Body Detection")
clock = pygame.time.Clock()

title_font = pygame.font.SysFont("arial", 48, bold=True)
count_font = pygame.font.SysFont("arial", 120, bold=True)
label_font = pygame.font.SysFont("arial", 18, bold=True)

# ── Load avatars ────────────────────────────────────────────────────
ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
AVATAR_FILES = ["creeper.jpg", "skeleton.jpg", "zombie.jpg"]

def load_avatar(index: int) -> pygame.Surface:
    fname = AVATAR_FILES[index % len(AVATAR_FILES)]
    path = os.path.join(ASSET_DIR, fname)
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.smoothscale(img, (AVATAR_SIZE, AVATAR_SIZE))

# ── Video ───────────────────────────────────────────────────────────
VIDEO_PATH = os.path.join(os.path.dirname(__file__), "..", "A.mp4")
cap_video = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap_video.get(cv2.CAP_PROP_FPS)
if video_fps <= 0:
    video_fps = 20.0
video_total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
cached_frame_surf = None

# ── Load jump cues from predicted_actions.jsonl ─────────────────────
JSONL_PATH = os.path.join(os.path.dirname(__file__), "..", "predicted_actions.jsonl")
CUE_LEAD   = 0.15   # seconds before jump to show circle (visual)
CUE_SCORE_LEAD = 0.30 # seconds before jump that a player is given points
CUE_LINGER = 0.35   # seconds after jump to keep circle
CUE_RADIUS = 88
CUE_APPROACH_TIME = 2.0   # seconds before cue that falling indicator starts
MIN_CUE_GAP = 0.5          # minimum seconds between consecutive cue onsets

def load_jump_times(jsonl_path, fps):
    """Extract onset times (seconds) when space first appears, with min gap."""
    raw_times = []
    prev_space = False
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            frame = entry["frame"]
            has_space = "key.keyboard.space" in entry.get("keyboard", {}).get("keys", [])
            if has_space and not prev_space:
                raw_times.append(frame / fps)
            prev_space = has_space
    # Filter out onsets that are too close together
    times = []
    for t in raw_times:
        if not times or (t - times[-1]) >= MIN_CUE_GAP:
            times.append(t)
    return times

jump_times = load_jump_times(JSONL_PATH, video_fps)
print(f"Jump cues loaded: {len(jump_times)} onsets")
if jump_times:
    print(f"  First 5 times: {jump_times[:5]}")

# ── Webcam ──────────────────────────────────────────────────────────
cap_webcam = cv2.VideoCapture(0)
webcam_fps = cap_webcam.get(cv2.CAP_PROP_FPS) or 30.0
webcam_frame_idx = 0

# ── YOLO person counter (INTRO / COUNTDOWN only) ───────────────────
yolo_model = YOLO("yolov8n.pt")

def count_people_yolo(frame_bgr):
    """Return the number of people detected by YOLO in a BGR frame."""
    results = yolo_model(frame_bgr, verbose=False)
    count = 0
    for r in results:
        for cls_id in r.boxes.cls:
            if int(cls_id) == 0:
                count += 1
    return count

# ── Pose detector (created at lock time with exact player count) ───
detector = None

# ── State ──────────────────────────────────────────────────────────
STATE_INTRO     = 0
STATE_COUNTDOWN = 1
STATE_PLAYING   = 2

state = STATE_INTRO
countdown_start: float = 0.0
locked_player_count: int = 0
player_avatars: list[pygame.Surface] = []

player_id_map: dict[int, int] = {}
player_actions: list[dict] = []
player_scores: list[int] = []
player_calories: list[float] = []
player_prev_jumping: list[bool] = []

start_time = time.time()

leaderboard_font = pygame.font.SysFont("arial", 28, bold=True)
leaderboard_title_font = pygame.font.SysFont("arial", 34, bold=True)
score_font = pygame.font.SysFont("arial", 22)


# ── Drawing helpers ─────────────────────────────────────────────────
last_video_frame_idx = -1          # track which frame we last displayed

def draw_video_bg(surface, ct):
    """Seek to the time-correct video frame for properly-paced playback."""
    global cached_frame_surf, last_video_frame_idx
    target_frame = int(ct * video_fps)
    if target_frame >= video_total_frames:
        return False  # Video finished

    if target_frame != last_video_frame_idx:
        current_pos = int(cap_video.get(cv2.CAP_PROP_POS_FRAMES))
        # If the next read would naturally give us the target, just read
        if current_pos == target_frame:
            ret, frame = cap_video.read()
        else:
            # Seek to the target frame
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap_video.read()
            
        if ret:
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (1, 0, 2))
            cached_frame_surf = pygame.surfarray.make_surface(frame)
            last_video_frame_idx = target_frame
            
    if cached_frame_surf:
        surface.blit(cached_frame_surf, (0, 0))
    return True


def draw_webcam_bg(surface, wc_frame):
    """Draw the webcam frame as a full-screen background."""
    if wc_frame is None:
        return
    frame = cv2.resize(wc_frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.transpose(frame, (1, 0, 2))
    surf = pygame.surfarray.make_surface(frame)
    surface.blit(surf, (0, 0))


def draw_centered_text(surface, text, font, color, y, shadow=True):
    if shadow:
        sh = font.render(text, True, BLACK)
        surface.blit(sh, (WINDOW_WIDTH // 2 - sh.get_width() // 2 + 2, y + 2))
    sf = font.render(text, True, color)
    surface.blit(sf, (WINDOW_WIDTH // 2 - sf.get_width() // 2, y))


def draw_player_hud(surface):
    n = locked_player_count
    if n == 0:
        return
    gap = 24
    total_w = n * AVATAR_SIZE + (n - 1) * gap
    start_x = (WINDOW_WIDTH - total_w) // 2
    y = 20

    for i in range(n):
        ax = start_x + i * (AVATAR_SIZE + gap)

        shadow = pygame.Surface((AVATAR_SIZE + 8, AVATAR_SIZE + 8), pygame.SRCALPHA)
        shadow.fill((0, 0, 0, 100))
        surface.blit(shadow, (ax - 4, y + 4))
        surface.blit(player_avatars[i], (ax, y))

        lbl = label_font.render(f"Player {i + 1}", True, WHITE)
        lx = ax + AVATAR_SIZE // 2 - lbl.get_width() // 2
        ly = y + AVATAR_SIZE + 6
        sh = label_font.render(f"Player {i + 1}", True, BLACK)
        surface.blit(sh, (lx + 1, ly + 1))
        surface.blit(lbl, (lx, ly))

        actions = player_actions[i] if i < len(player_actions) else {}
        is_running = actions.get("running", False)
        is_jumping = actions.get("jumping", False)
        circle_x = ax + AVATAR_SIZE // 2
        circle_y = ly + lbl.get_height() + 8 + CIRCLE_RADIUS

        if is_jumping:
            pygame.draw.circle(surface, GREEN, (circle_x, circle_y), CIRCLE_RADIUS)
            pygame.draw.circle(surface, WHITE, (circle_x, circle_y), CIRCLE_RADIUS, 2)
            circle_y += CIRCLE_RADIUS * 2 + 6

        if is_running:
            pygame.draw.circle(surface, ORANGE, (circle_x, circle_y), CIRCLE_RADIUS)
            pygame.draw.circle(surface, WHITE, (circle_x, circle_y), CIRCLE_RADIUS, 2)


def draw_leaderboard(surface):
    """Draw a semi-transparent leaderboard sidebar on the right."""
    n = locked_player_count
    if n == 0:
        return
    # Semi-transparent background
    sidebar = pygame.Surface((SIDEBAR_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    sidebar.fill(SIDEBAR_BG)
    surface.blit(sidebar, (WINDOW_WIDTH - SIDEBAR_WIDTH, 0))

    sx = WINDOW_WIDTH - SIDEBAR_WIDTH + 16
    # Title
    title = leaderboard_title_font.render("Leaderboard", True, GOLD)
    surface.blit(title, (sx, 20))
    # Divider line
    pygame.draw.line(surface, GOLD,
                     (sx, 60), (WINDOW_WIDTH - 16, 60), 2)

    # Sort players by score descending
    ranked = sorted(range(n), key=lambda i: player_scores[i], reverse=True)
    for rank, i in enumerate(ranked):
        y = 75 + rank * 72
        # Avatar
        small_av = pygame.transform.smoothscale(player_avatars[i], (44, 44))
        surface.blit(small_av, (sx, y + 4))
        # Name
        name = leaderboard_font.render(f"Player {i + 1}", True, WHITE)
        surface.blit(name, (sx + 52, y + 2))
        
        # Split Score and Calories
        pts_text = f"{player_scores[i]} pts"
        cal_text = f"| {int(player_calories[i])} cal"
        
        pts_surf = score_font.render(pts_text, True, GREEN)
        surface.blit(pts_surf, (sx + 52, y + 32))
        
        cal_surf = score_font.render(cal_text, True, ORANGE)
        surface.blit(cal_surf, (sx + 52 + pts_surf.get_width() + 6, y + 32))
        
        # Rank medal
        if rank == 0 and player_scores[i] > 0:
            medal = leaderboard_font.render("⭐", True, GOLD)
            surface.blit(medal, (WINDOW_WIDTH - 40, y + 8))


# ── Main loop ───────────────────────────────────────────────────────
running = True
while running:
    clock.tick(FPS)
    ct = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # ── Webcam frame + detection ────────────────────────────────────
    ret_wc, wc_frame = cap_webcam.read()
    if ret_wc:
        wc_frame = cv2.flip(wc_frame, 1)   # mirror horizontally
    persons = []

    # Only run PoseDetector during PLAYING (fast, no YOLO overhead)
    if state == STATE_PLAYING and ret_wc and detector is not None:
        persons = detector.process_frame(wc_frame, webcam_frame_idx)
        webcam_frame_idx += 1

    # ── Draw background ─────────────────────────────────────────────
    screen.fill(DARK)
    video_playing = False
    if state == STATE_PLAYING:
        video_playing = draw_video_bg(screen, ct)
        if not video_playing and ret_wc:
            # Video ended, fallback to webcam background
            draw_webcam_bg(screen, wc_frame)
    elif ret_wc:
        draw_webcam_bg(screen, wc_frame)

    # ── State machine ───────────────────────────────────────────────
    if state == STATE_INTRO:
        draw_centered_text(
            screen, "Stand in frame!", title_font, WHITE, WINDOW_HEIGHT // 2 - 30
        )
        if ret_wc:
            yolo_count = count_people_yolo(wc_frame)
            if yolo_count >= 1:
                state = STATE_COUNTDOWN
                countdown_start = time.time()

    elif state == STATE_COUNTDOWN:
        elapsed = time.time() - countdown_start
        remaining = COUNTDOWN_SECONDS - elapsed

        yolo_count = 0
        if ret_wc:
            yolo_count = count_people_yolo(wc_frame)

        if remaining > 0:
            draw_centered_text(
                screen, str(int(remaining) + 1), count_font, WHITE,
                WINDOW_HEIGHT // 2 - 70,
            )
            draw_centered_text(
                screen,
                f"{yolo_count} player{'s' if yolo_count != 1 else ''} detected",
                title_font, WHITE, WINDOW_HEIGHT // 2 + 50,
            )
        else:
            locked_player_count = max(yolo_count, 1)
            player_avatars = [load_avatar(i) for i in range(locked_player_count)]
            player_actions = [
                {"running": False, "jumping": False}
                for _ in range(locked_player_count)
            ]
            player_scores = [0] * locked_player_count
            player_calories = [0.0] * locked_player_count
            player_prev_jumping = [False] * locked_player_count
            player_inactive_frames = [0] * locked_player_count
            # For calorie counting: tracking activity within 20-frame blocks
            player_active_frames_in_block = [0] * locked_player_count
            player_frames_in_block = [0] * locked_player_count
            player_id_map = {}
            detector = PoseDetector(
                max_poses=locked_player_count, fps=webcam_fps
            )
            start_time = time.time()   # reset so video frame 0 = now
            cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind video
            last_video_frame_idx = -1
            state = STATE_PLAYING

    elif state == STATE_PLAYING:
        id_to_person = {p.id: p for p in persons}
        claimed_ids = set()
        for slot in range(locked_player_count):
            pid = player_id_map.get(slot)
            if pid is not None and pid in id_to_person:
                claimed_ids.add(pid)

        for slot in range(locked_player_count):
            pid = player_id_map.get(slot)
            if pid is not None and pid in id_to_person:
                p = id_to_person[pid]
                player_actions[slot]["running"] = p.is_running
                player_actions[slot]["jumping"] = p.is_jumping
            else:
                reassigned = False
                for p in persons:
                    if p.frames_missing == 0 and p.id not in claimed_ids:
                        player_id_map[slot] = p.id
                        claimed_ids.add(p.id)
                        player_actions[slot]["running"] = p.is_running
                        player_actions[slot]["jumping"] = p.is_jumping
                        reassigned = True
                        break
                if not reassigned:
                    player_actions[slot]["running"] = False
                    player_actions[slot]["jumping"] = False

        draw_player_hud(screen)

        # ── Advanced Scoring logic ──────────────────────────────────────────
        for slot in range(locked_player_count):
            actions = player_actions[slot]
            now_jumping = actions.get("jumping", False)
            now_running = actions.get("running", False)
            
            # Calories calculation (proportional up to ~0.4 per 20 frames)
            player_frames_in_block[slot] += 1
            if now_running or now_jumping:
                player_active_frames_in_block[slot] += 1
                
            if player_frames_in_block[slot] >= 20:
                if player_active_frames_in_block[slot] > 0:
                    proportion = player_active_frames_in_block[slot] / 20.0
                    player_calories[slot] += (0.4 * proportion)
                player_frames_in_block[slot] = 0
                player_active_frames_in_block[slot] = 0

            # Check inactivity penalty (no running, no jumping)
            if not now_running and not now_jumping:
                player_inactive_frames[slot] += 1
                if player_inactive_frames[slot] >= 20:
                    player_scores[slot] -= 5
                    player_inactive_frames[slot] = 0  # reset after deduction
            else:
                player_inactive_frames[slot] = 0

            # Jump onset detection
            if now_jumping and not player_prev_jumping[slot]:
                # Is there a valid cue window active?
                valid_jump = False
                for jt in jump_times:
                    if (jt - CUE_SCORE_LEAD) <= ct <= (jt + CUE_LINGER):
                        valid_jump = True
                        break
                        
                if valid_jump:
                    player_scores[slot] += 10
                else:
                    player_scores[slot] -= 5

            player_prev_jumping[slot] = now_jumping

        draw_leaderboard(screen)

    # ── Debug cam (bottom-left) ─────────────────────────────────────
    if ret_wc:
        debug_frame = wc_frame.copy()
        h, w = debug_frame.shape[:2]
        for p in persons:
            if p.last_landmarks is not None and p.frames_missing == 0:
                draw_skeleton(debug_frame, p.last_landmarks, h, w)
        debug_frame = cv2.resize(debug_frame, (DEBUG_W, DEBUG_H))
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
        debug_frame = np.transpose(debug_frame, (1, 0, 2))
        debug_surf = pygame.surfarray.make_surface(debug_frame)
        pygame.draw.rect(
            screen, WHITE,
            (0, WINDOW_HEIGHT - DEBUG_H - 2, DEBUG_W + 4, DEBUG_H + 4), 2,
        )
        screen.blit(debug_surf, (2, WINDOW_HEIGHT - DEBUG_H))

    # ── Jump cue circle (bottom center) ────────────────────────────
    if state == STATE_PLAYING and video_playing:
        cue_x = WINDOW_WIDTH // 2
        cue_y = WINDOW_HEIGHT - CUE_RADIUS - 30

        # Target ring backdrop is transparent now; drawn at the end as an outline

        # Draw stacked approaching + lingering cues
        for idx, jt in enumerate(jump_times):
            arrive_time = jt - CUE_LEAD
            approach_start_t = arrive_time - CUE_APPROACH_TIME
            
            # Active in approach window or linger window
            if approach_start_t <= ct <= (jt + CUE_LINGER):
                if ct < arrive_time:
                    progress = (ct - approach_start_t) / CUE_APPROACH_TIME
                else:
                    progress = 1.0  # Fully filled if lingering
                
                # Circle fills from inside out (radius starts at 0, grows to CUE_RADIUS)
                fill_r = int(CUE_RADIUS * progress)
                
                color = CUE_COLORS[idx % len(CUE_COLORS)]
                
                if fill_r > 0:
                    pygame.draw.circle(screen, color, (cue_x, cue_y), fill_r)
                    
                # Sunburst effect when fully filled, for ~5 frames (5/30 seconds)
                if ct >= arrive_time and ct <= arrive_time + (5 / FPS):
                    # How far along the 5-frame burst we are
                    burst_prog = (ct - arrive_time) / (5 / FPS)
                    line_len_base = 25
                    # Flare expanding outwards
                    flare_start = CUE_RADIUS + int(15 * burst_prog)
                    flare_end = flare_start + int(line_len_base * (1.0 - burst_prog))
                    
                    if flare_end > flare_start:
                        for angle_deg in range(0, 360, 30): # 12 rays
                            angle_rad = math.radians(angle_deg)
                            start_x = cue_x + int(flare_start * math.cos(angle_rad))
                            start_y = cue_y + int(flare_start * math.sin(angle_rad))
                            end_x = cue_x + int(flare_end * math.cos(angle_rad))
                            end_y = cue_y + int(flare_end * math.sin(angle_rad))
                            pygame.draw.line(screen, (255, 50, 50), (start_x, start_y), (end_x, end_y), 5)
                        
        # Foremost border
        pygame.draw.circle(screen, WHITE, (cue_x, cue_y), CUE_RADIUS, 3)

    pygame.display.flip()

# ── Cleanup ─────────────────────────────────────────────────────────
if detector is not None:
    detector.close()
cap_webcam.release()
cap_video.release()
pygame.quit()