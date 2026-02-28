import subprocess
import os
import sys
import math
import time
from dotenv import load_dotenv

import re
import random
import threading
import json
import pygame
import cv2
import numpy as np
import requests
import yt_dlp
import boto3
import imageio_ffmpeg as ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from body_new import PoseDetector, draw_skeleton  # noqa: E402
from ultralytics import YOLO  # noqa: E402

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_KEY"),
    aws_secret_access_key=os.getenv("SECRET_AWS_KEY"),
    region_name='us-east-1'
)
bucket_name = "minecraft-videos-dance"

GEMINI_KEY = os.getenv("GEMINI_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 30
AVATAR_SIZE   = 80
COUNTDOWN_SECONDS = 5
CIRCLE_RADIUS = 14
SIDEBAR_WIDTH = 220

DEBUG_W = WINDOW_WIDTH  // 4
DEBUG_H = WINDOW_HEIGHT // 4

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

WINDOW_WIDTH  = 1280
WINDOW_HEIGHT = 720
FPS           = 30
AVATAR_SIZE   = 80
COUNTDOWN_SECONDS = 5
CIRCLE_RADIUS = 14
SIDEBAR_WIDTH = 220

DEBUG_W = WINDOW_WIDTH  // 4
DEBUG_H = WINDOW_HEIGHT // 4

DARK   = (18, 18, 24)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
GREEN  = (0, 200, 80)
ORANGE = (255, 160, 0)
GOLD   = (255, 215, 0)
SIDEBAR_BG = (20, 20, 30, 180)


VOICE_LINES = [
    "chicken jockey",
    "water bucket, release",
    "amaezing",
    "i am steve",
    "this... is a crafting table",
    "they love crushing loaf",
]

AUDIO_CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio")
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

last_congrats_time = -999.0
CONGRATS_COOLDOWN = 5.0
GEMINI_VOICE_COOLDOWN = 10.0
last_gemini_voice_time = -999.0

def line_to_filename(line: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", line.lower()).strip("_")
    return safe + ".mp3"

def pregenerate_voice_lines():
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    if not ELEVENLABS_API_KEY:
        print("[TTS] No ELEVENLABS_API_KEY found, skipping pregeneration.")
        return
    VOICE_ID = "z2RqfzHxVAbH6LCC7Jc3"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    for line in VOICE_LINES:
        path = os.path.join(AUDIO_CACHE_DIR, line_to_filename(line))
        if not os.path.exists(path):
            print(f"[TTS] Generating: '{line}' -> {os.path.basename(path)}")
            try:
                resp = requests.post(url, json={
                    "text": line,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
                }, headers=headers)
                resp.raise_for_status()
                with open(path, "wb") as f:
                    f.write(resp.content)
                print(f"[TTS] Saved: {path}")
            except Exception as e:
                print(f"[TTS] FAILED to generate '{line}': {e}")
        else:
            print(f"[TTS] Already cached: {os.path.basename(path)}")

def _play_audio_file(path: str):
    """Load and play an MP3 file via pygame mixer (call from any thread)."""
    try:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        print(f"[TTS] Playing: {os.path.basename(path)}")
    except Exception as e:
        print(f"[TTS error] {e}")

def _speak_text_via_elevenlabs(text: str):
    """Call ElevenLabs with arbitrary text and play it (blocks until done)."""
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
    VOICE_ID = "z2RqfzHxVAbH6LCC7Jc3"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, json={
            "text": text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }, headers=headers)
        resp.raise_for_status()
        # Write to a uniquely-named temp file to avoid Windows lock conflicts
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False,
                                         dir=AUDIO_CACHE_DIR, prefix="gemini_") as tmp:
            tmp.write(resp.content)
            path = tmp.name
        _play_audio_file(path)
        # Wait for playback then clean up
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)
        try:
            os.remove(path)
        except Exception:
            pass
    except Exception as e:
        print(f"[Gemini TTS error] {e}")

def ask_gemini_for_voice_line(scores: list, calories: list, ct: float) -> str | None:
    """Ask Gemini for a short encouraging in-game voice line. Returns text or None."""
    if not GEMINI_KEY:
        return None
    n = len(scores)
    score_summary = ", ".join(f"Player {i+1}: {scores[i]} pts" for i in range(n))
    prompt = (
        f"You are a hype commentator for a Minecraft parkour dance game called JustParkour. "
        f"The game has been running for {ct:.0f} seconds. "
        f"Current scores: {score_summary}. "
        f"Write ONE short, punchy, encouraging in-game voice line (max 8 words). "
        f"Make it fun and Minecraft-themed. Return only the voice line, nothing else."
    )
    try:
        resp = requests.post(
            GEMINI_API_URL,
            params={"key": GEMINI_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text
    except Exception as e:
        print(f"[Gemini] Error: {e}")
        return None

def fetch_game_summary(scores: list, calories: list, game_duration: float) -> list[str]:
    """Ask Gemini for a post-game summary and improvement tips. Returns list of lines."""
    if not GEMINI_KEY:
        return ["Game over!", "Nice effort from all players."]
    n = len(scores)
    score_summary = ", ".join(f"Player {i+1}: {scores[i]} pts, {calories[i]:.1f} cal" for i in range(n))
    prompt = (
        f"A Minecraft parkour dance game called JustParkour just ended after {game_duration:.0f} seconds. "
        f"Results: {score_summary}. "
        f"Write a fun, brief post-game summary (2-3 sentences) and 1-2 bullet points on what players can improve. "
        f"Keep it under 60 words total. Use Minecraft-themed language. "
        f"Format: summary paragraph, then bullet points starting with '•'."
    )
    try:
        resp = requests.post(
            GEMINI_API_URL,
            params={"key": GEMINI_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=8
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Split into wrapped lines for rendering
        lines = []
        for paragraph in raw.split("\n"):
            paragraph = paragraph.strip()
            if paragraph:
                lines.append(paragraph)
        return lines
    except Exception as e:
        print(f"[Gemini summary] Error: {e}")
        return ["Game over! Great effort, everyone.", "• Keep jumping on the beat!", "• Stay active to avoid score penalties."]

def play_congratulations(scores: list, calories: list, ct: float):
    """Play a voice line on a correct jump. 40% chance = Gemini-generated."""
    global last_congrats_time, last_gemini_voice_time
    now = time.time()

    # Respect the longer cooldown if a Gemini line was recently played
    effective_cooldown = CONGRATS_COOLDOWN
    if now - last_gemini_voice_time < GEMINI_VOICE_COOLDOWN:
        return  # Still in Gemini silence window
    if now - last_congrats_time < effective_cooldown:
        return

    use_gemini = random.random() < 0.00

    if use_gemini:
        last_congrats_time = now
        last_gemini_voice_time = now
        scores_copy = list(scores)
        cals_copy = list(calories)

        def _run_gemini():
            text = ask_gemini_for_voice_line(scores_copy, cals_copy, ct)
            if text:
                print(f"[Gemini voice] '{text}'")
                _speak_text_via_elevenlabs(text)
            else:
                # Fallback to cached line if Gemini fails
                _play_cached_line()

        threading.Thread(target=_run_gemini, daemon=True).start()
    else:
        last_congrats_time = now
        threading.Thread(target=_play_cached_line, daemon=True).start()

def _play_cached_line():
    line = random.choice(VOICE_LINES)
    path = os.path.join(AUDIO_CACHE_DIR, line_to_filename(line))
    if not os.path.exists(path):
        print(f"[TTS] Cache miss for '{line}'")
        return
    _play_audio_file(path)

pygame.init()
pygame.mixer.init()
pregenerate_voice_lines()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.scrap.init()
pygame.display.set_caption("JustParkour")
clock = pygame.time.Clock()

title_font = pygame.font.SysFont("arial", 48, bold=True)
count_font = pygame.font.SysFont("arial", 120, bold=True)
label_font = pygame.font.SysFont("arial", 18, bold=True)

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
AVATAR_FILES = ["creeper.jpg", "skeleton.jpg", "zombie.jpg"]

def load_avatar(index: int) -> pygame.Surface:
    fname = AVATAR_FILES[index % len(AVATAR_FILES)]
    path = os.path.join(ASSET_DIR, fname)
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.smoothscale(img, (AVATAR_SIZE, AVATAR_SIZE))

PRE_VIDEO_PATH = "video.mp4"

VIDEO_URL = ""
VIDEO_NAME = ""
VIDEO_PATH = ""
cap_video = None
video_fps = 0.0
video_total_frames = 0
cached_frame_surf = None
jump_times = []

JSONL_PATH = os.path.join(os.path.dirname(__file__), "predicted_actions.jsonl")
CUE_LEAD   = 0.15
CUE_SCORE_LEAD = 0.30
CUE_LINGER = 0.35
CUE_RADIUS = 88
CUE_APPROACH_TIME = 2.0
MIN_CUE_GAP = 0.5

def download_youtube(url: str, output: str = PRE_VIDEO_PATH):
    ydl_opts = {
        'outtmpl': output,
        'format': 'best[ext=mp4]',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def convert_video(input_path: str, output_path: str):
    command = [
        ffmpeg.get_ffmpeg_exe(),
        '-i', input_path,
        '-vf', 'fps=20,scale=640:360',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'ultrafast',
        output_path
    ]
    subprocess.run(command, check=True)

def upload_to_s3(file_path: str, key: str):
    s3.upload_file(file_path, bucket_name, key)

def init_video(url: str):
    global VIDEO_URL, VIDEO_NAME, VIDEO_PATH
    global cap_video, video_fps, video_total_frames, jump_times
    
    VIDEO_URL = url
    VIDEO_NAME = VIDEO_URL.split("v=")[-1] if "v=" in VIDEO_URL else VIDEO_URL.split("/")[-1]
    VIDEO_PATH = f"{VIDEO_NAME}.mp4"

    # ── Ensure video is ready ────────────────────────────────────────────
    if not os.path.exists(VIDEO_PATH):
        if os.path.exists(PRE_VIDEO_PATH):
            os.remove(PRE_VIDEO_PATH)
        if os.path.exists(VIDEO_PATH):
            os.remove(VIDEO_PATH)

        print("Downloading video...")
        download_youtube(VIDEO_URL, PRE_VIDEO_PATH)

        print("Converting video...")
        convert_video(PRE_VIDEO_PATH, VIDEO_PATH)

        print("Uploading video to S3...")
        upload_to_s3(VIDEO_PATH, VIDEO_NAME + ".mp4")
    else:
        print("Video already exists locally, skipping download and conversion.")

    response = requests.get("http://ec2-100-54-112-34.compute-1.amazonaws.com:8000/download-result?id=" + VIDEO_NAME)
    response.raise_for_status()
    with open(JSONL_PATH, "wb") as f:
        f.write(response.content)

    cap_video = cv2.VideoCapture(VIDEO_PATH)
    video_fps = cap_video.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 20.0
    video_total_frames = int(cap_video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    jump_times = load_jump_times(JSONL_PATH, video_fps)


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
STATE_INPUT     = -1
STATE_LOADING   = -2
STATE_INTRO     = 0
STATE_COUNTDOWN = 1
STATE_PLAYING   = 2
STATE_ENDSCREEN = 3

end_summary_text: list[str] = []
end_summary_ready = False
restart_button_rect = pygame.Rect(WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT - 100, 240, 50)

state = STATE_INPUT
input_url_text = ""

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
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            
            # STATE_INPUT handling
            if state == STATE_INPUT:
                if event.key == pygame.K_RETURN:
                    if input_url_text.strip():
                        state = STATE_LOADING
                elif event.key == pygame.K_BACKSPACE:
                    input_url_text = input_url_text[:-1]
                elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_CTRL or pygame.key.get_mods() & pygame.KMOD_META):
                    try:
                        import pyperclip
                        input_url_text += pyperclip.paste()
                    except ImportError:
                        try:
                            # Fallback to pygame scrap
                            clip = pygame.scrap.get(pygame.SCRAP_TEXT)
                            if clip:
                                input_url_text += clip.decode("utf-8").strip('\x00')
                        except Exception:
                            pass
                else:
                    input_url_text += event.unicode
                    
            elif event.key == pygame.K_SPACE:
                # Force state progression
                if state == STATE_INTRO:
                    state = STATE_COUNTDOWN
                    countdown_start = time.time()
                elif state == STATE_PLAYING:
                    # Reset
                    state = STATE_INPUT
                    input_url_text = ""
                    player_avatars.clear()
                    player_id_map.clear()
                    player_actions.clear()
                    player_scores.clear()
                    player_calories.clear()
                    player_prev_jumping.clear()
                    player_inactive_frames.clear()
                    detector = None
                    if cap_video is not None:
                        cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        last_video_frame_idx = -1
                    start_time = time.time()

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
    # Input state is purely black. Loading state shows webcam.
    screen.fill(BLACK if state == STATE_INPUT else DARK)
    video_playing = False
    
    if state == STATE_PLAYING:
        video_playing = draw_video_bg(screen, ct)
        if not video_playing and ret_wc:
            # Video ended, fallback to webcam background
            draw_webcam_bg(screen, wc_frame)
    elif state != STATE_INPUT and ret_wc:
        draw_webcam_bg(screen, wc_frame)

    # ── State machine ───────────────────────────────────────────────
    if state == STATE_INPUT:
        # Pure black layout without webcam
        input_rect = pygame.Rect(WINDOW_WIDTH // 2 - 400, WINDOW_HEIGHT // 2 - 30, 800, 60)
        pygame.draw.rect(screen, (40, 40, 50, 200), input_rect)
        pygame.draw.rect(screen, WHITE, input_rect, 2)
        
        prompt_surf = leaderboard_title_font.render("Enter YouTube URL (Minecraft Parkour):", True, WHITE)
        screen.blit(prompt_surf, (input_rect.x, input_rect.y - 50))
        
        url_surf = leaderboard_font.render(input_url_text, True, (200, 200, 200))
        text_x = input_rect.x + 10
        if url_surf.get_width() > input_rect.width - 20:
            text_x = input_rect.right - url_surf.get_width() - 10
        screen.blit(url_surf, (text_x, input_rect.y + 15))

    elif state == STATE_LOADING:
        # Draw webcam + loading text
        loading_surf = leaderboard_title_font.render("Loading video... Please wait.", True, ORANGE)
        # Add a subtle dark backdrop for readability over webcam
        bg_rect = loading_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
        bg_rect.inflate_ip(40, 20)
        pygame.draw.rect(screen, (0, 0, 0, 150), bg_rect)
        screen.blit(loading_surf, loading_surf.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)))
        pygame.display.flip()
        
        try:
            init_video(input_url_text.strip())
            state = STATE_COUNTDOWN
            countdown_start = time.time()
        except Exception as e:
            print(f"Error loading video: {e}")
            state = STATE_INPUT
            input_url_text = ""
                
    elif state == STATE_INTRO:
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
                    play_congratulations(player_scores, player_calories, ct)
                else:
                    player_scores[slot] -= 5

            player_prev_jumping[slot] = now_jumping

        draw_leaderboard(screen)

        if not video_playing:
            state = STATE_ENDSCREEN
            end_summary_ready = False
            end_summary_text = ["Calculating results..."]
            _scores_snap = list(player_scores)
            _cals_snap = list(player_calories)
            _dur_snap = ct
            def _fetch_summary():
                global end_summary_text, end_summary_ready
                end_summary_text = fetch_game_summary(_scores_snap, _cals_snap, _dur_snap)
                end_summary_ready = True
            threading.Thread(target=_fetch_summary, daemon=True).start()
    
    elif state == STATE_ENDSCREEN:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 210))
        screen.blit(overlay, (0, 0))

        title_surf = title_font.render("JustParkour — Game Over!", True, GOLD)
        screen.blit(title_surf, (WINDOW_WIDTH // 2 - title_surf.get_width() // 2, 40))

        y = 110
        for i, (sc, cal) in enumerate(zip(player_scores, player_calories)):
            line = leaderboard_font.render(
                f"Player {i+1}:  {sc} pts  |  {int(cal)} cal burned", True, WHITE
            )
            screen.blit(line, (WINDOW_WIDTH // 2 - line.get_width() // 2, y))
            y += 40

        pygame.draw.line(screen, GOLD, (100, y + 10), (WINDOW_WIDTH - 100, y + 10), 2)
        y += 28

        summary_font = pygame.font.SysFont("arial", 22)
        if not end_summary_ready:
            wait_surf = summary_font.render("Asking Gemini for your recap...", True, ORANGE)
            screen.blit(wait_surf, (WINDOW_WIDTH // 2 - wait_surf.get_width() // 2, y))
        else:
            for line in end_summary_text:
                color = ORANGE if line.startswith("•") else WHITE
                line_surf = summary_font.render(line, True, color)
                # Word-wrap: if too wide, truncate (simple approach for fixed font)
                if line_surf.get_width() > WINDOW_WIDTH - 200:
                    # Split roughly in half
                    words = line.split()
                    mid = len(words) // 2
                    for part in [" ".join(words[:mid]), " ".join(words[mid:])]:
                        ps = summary_font.render(part, True, color)
                        screen.blit(ps, (WINDOW_WIDTH // 2 - ps.get_width() // 2, y))
                        y += 32
                else:
                    screen.blit(line_surf, (WINDOW_WIDTH // 2 - line_surf.get_width() // 2, y))
                    y += 32

        pygame.draw.rect(screen, GREEN, restart_button_rect, border_radius=10)
        pygame.draw.rect(screen, WHITE, restart_button_rect, 2, border_radius=10)
        btn_surf = leaderboard_font.render("Play Again", True, BLACK)
        screen.blit(btn_surf, (
            restart_button_rect.centerx - btn_surf.get_width() // 2,
            restart_button_rect.centery - btn_surf.get_height() // 2
        ))

        # Handle button click
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if restart_button_rect.collidepoint(mx, my):
                # Full reset back to URL input
                state = STATE_INPUT
                input_url_text = ""
                player_avatars.clear()
                player_id_map.clear()
                player_actions.clear()
                player_scores.clear()
                player_calories.clear()
                player_prev_jumping.clear()
                if 'player_inactive_frames' in dir():
                    player_inactive_frames.clear()
                end_summary_text = []
                end_summary_ready = False
                detector = None
                if cap_video is not None:
                    cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                last_video_frame_idx = -1
                start_time = time.time()

    # ── Debug cam (bottom-left) ─────────────────────────────────────
    if ret_wc and state not in [STATE_INPUT, STATE_LOADING]:
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
if cap_webcam is not None:
    cap_webcam.release()
if cap_video is not None:
    cap_video.release()
pygame.quit()