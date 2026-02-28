import pygame
import cv2
import numpy as np
import time
import math
import os
import yt_dlp

VIDEO_URL    = "https://www.youtube.com/watch?v=lekKHbYQGxM"
WINDOW_WIDTH  = 960
WINDOW_HEIGHT = 540
FPS           = 60
AVATAR_SIZE   = 72

WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0  )
DARK   = (10,  10,  20 )
RED    = (255, 60,  80 )
YELLOW = (255, 230, 0  )
CYAN   = (0,   230, 255)
PINK   = (255, 80,  200)
GREEN  = (80,  255, 160)

PLAYER_COLORS = [
    (0,   200, 255),
    (255, 80,  200),
    (255, 220, 0  ),
    (80,  255, 160),
]
VIDEO_PATH = "video.mp4"

ydl_opts = {
    'outtmpl': 'video.%(ext)s',
    'format': 'best[ext=mp4]',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([VIDEO_URL])

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Dance Game")
clock = pygame.time.Clock()

try:
    score_font = pygame.font.SysFont("Impact", 20)
    hit_font   = pygame.font.SysFont("Impact", 56)
    small_font = pygame.font.SysFont("Arial",  14)
    init_font  = pygame.font.SysFont("Impact", 38)
    amaz_font  = pygame.font.SysFont("Impact", 22)
except Exception:
    score_font = hit_font = small_font = init_font = amaz_font = pygame.font.SysFont(None, 32)

players = ["Alice", "Bob", "You"]
profiles = ["creeper.jpg", "skeleton.jpg", "zombie.jpg"] 
scores  = {p: 0 for p in players}

def load_avatar(index):
    path = f"assets/{profiles[index]}"
    if os.path.exists(path):
        try:
            img = pygame.image.load(path).convert_alpha()
            return pygame.transform.smoothscale(img, (AVATAR_SIZE, AVATAR_SIZE))
        except Exception:
            pass
    surf = pygame.Surface((AVATAR_SIZE, AVATAR_SIZE), pygame.SRCALPHA)
    color = PLAYER_COLORS[index % len(PLAYER_COLORS)]
    surf.fill((*color, 210))
    letter = init_font.render(players[index][0].upper(), True, BLACK)
    surf.blit(letter, (AVATAR_SIZE // 2 - letter.get_width() // 2,
                        AVATAR_SIZE // 2 - letter.get_height() // 2))
    return surf

avatars = [load_avatar(i) for i in range(len(players))]

cap = cv2.VideoCapture(VIDEO_PATH)
video_fps = cap.get(cv2.CAP_PROP_FPS)
if video_fps <= 0 or video_fps > 120:
    video_fps = 30.0
frame_duration    = 1.0 / video_fps
last_frame_time   = -999.0
cached_frame_surf = None

ARROW_TARGET_Y = WINDOW_HEIGHT - 120
ARROW_START_Y  = WINDOW_HEIGHT + 70
ARROW_X        = WINDOW_WIDTH  // 2
ARROW_TRAVEL   = 1.2
ARROW_WINDOW   = 0.35

jump_times = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31]

start_time = time.time()

ai_offsets = {}
for i, name in enumerate(players):
    if name != "You":
        ai_offsets[name] = 0.1 * (1 if i % 2 == 0 else -1)

amazing_popups = [{"active": False, "timer": 0.0} for _ in players]

class Arrow:
    def __init__(self, spawn_game_time):
        self.spawn_time     = spawn_game_time
        self.hit            = False
        self.missed         = False
        self.hit_time       = None
        self.ai_press_times = {}
        self.ai_scored      = {}

    def get_y(self, ct):
        t      = min((ct - self.spawn_time) / ARROW_TRAVEL, 1.0)
        t_ease = 1 - (1 - t) ** 3
        return ARROW_START_Y + t_ease * (ARROW_TARGET_Y - ARROW_START_Y)

    def is_in_window(self, ct):
        return abs((ct - self.spawn_time) - ARROW_TRAVEL) < ARROW_WINDOW

    def is_expired(self, ct):
        return (ct - self.spawn_time) > ARROW_TRAVEL + ARROW_WINDOW + 0.5

active_arrows    = []
used_spawn_times = set()

hit_flash = {"active": False, "timer": 0.0, "label": "PERFECT!", "color": YELLOW}

def draw_arrow_shape(surface, cx, cy, color, scale=1.0, alpha=255):
    w      = int(40 * scale)
    sw     = int(15 * scale)
    head_h = int(55 * scale)
    sh     = int(45 * scale)
    pts = [
        (cx,      cy - head_h),
        (cx - w,  cy),
        (cx - sw, cy),
        (cx - sw, cy + sh),
        (cx + sw, cy + sh),
        (cx + sw, cy),
        (cx + w,  cy),
    ]
    if alpha < 255:
        s = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(s, (*color, alpha), pts)
        surface.blit(s, (0, 0))
    else:
        pygame.draw.polygon(surface, color, pts)

def draw_ghost_arrow(surface, cx, cy):
    w = 40; sw = 15; head_h = 55; sh = 45
    pts = [
        (cx,      cy - head_h),
        (cx - w,  cy),
        (cx - sw, cy),
        (cx - sw, cy + sh),
        (cx + sw, cy + sh),
        (cx + sw, cy),
        (cx + w,  cy),
    ]
    gs = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(gs, (255, 255, 255, 20), pts)
    pygame.draw.polygon(gs, (255, 60, 80, 90), pts, 3)
    surface.blit(gs, (0, 0))

def draw_hud(surface, ct):
    n     = len(players)
    pad   = 18
    gap   = 16
    total_w = n * AVATAR_SIZE + (n - 1) * gap
    start_x = (WINDOW_WIDTH - total_w) // 2

    for i, name in enumerate(players):
        color  = PLAYER_COLORS[i % len(PLAYER_COLORS)]
        ax     = start_x + i * (AVATAR_SIZE + gap)
        ay     = pad

        shadow_surf = pygame.Surface((AVATAR_SIZE + 8, AVATAR_SIZE + 8), pygame.SRCALPHA)
        shadow_surf.fill((0, 0, 0, 90))
        surface.blit(shadow_surf, (ax - 4, ay + 4))

        surface.blit(avatars[i], (ax, ay))

        pygame.draw.rect(surface, color, (ax, ay, AVATAR_SIZE, AVATAR_SIZE), 2)

        score_str  = str(scores[name])
        sh_surf    = score_font.render(score_str, True, BLACK)
        sc_surf    = score_font.render(score_str, True, WHITE)
        tx = ax + AVATAR_SIZE // 2 - sc_surf.get_width() // 2
        ty = ay + AVATAR_SIZE + 4
        surface.blit(sh_surf, (tx + 1, ty + 1))
        surface.blit(sc_surf, (tx,     ty))

        # Name label above (small, colored, shadowed)
        nm_sh = small_font.render(name.upper(), True, BLACK)
        nm_sf = small_font.render(name.upper(), True, color)
        nx = ax + AVATAR_SIZE // 2 - nm_sf.get_width() // 2
        ny = ay - nm_sf.get_height() - 2
        surface.blit(nm_sh, (nx + 1, ny + 1))
        surface.blit(nm_sf, (nx,     ny))

        popup = amazing_popups[i]
        if popup["active"]:
            age   = ct - popup["timer"]
            if age > 1.2:
                popup["active"] = False
            else:
                alpha  = int(255 * max(0, 1 - age / 1.2))
                rise   = int(age * 30)
                a_surf = amaz_font.render("Amazing!", True, YELLOW)
                a_surf = pygame.transform.rotate(a_surf, 30)
                a_surf.set_alpha(alpha)
                px = ax + AVATAR_SIZE // 2 - a_surf.get_width() // 2
                py = ay + AVATAR_SIZE // 2 - a_surf.get_height() // 2 - rise
                surface.blit(a_surf, (px, py))

def draw_hit_flash(surface, flash_state, ct):
    if not flash_state["active"]:
        return
    age = ct - flash_state["timer"]
    if age > 0.55:
        flash_state["active"] = False
        return
    alpha = int(255 * (1 - age / 0.55))
    scale = 1.0 + age * 2.0
    lbl   = hit_font.render(flash_state["label"], True, flash_state["color"])
    lbl   = pygame.transform.scale(lbl, (int(lbl.get_width() * scale),
                                         int(lbl.get_height() * scale)))
    lbl.set_alpha(alpha)
    surface.blit(lbl, (ARROW_X - lbl.get_width()  // 2,
                        ARROW_TARGET_Y - 110 - lbl.get_height() // 2))

running = True
while running:
    clock.tick(FPS)
    ct = time.time() - start_time

    # --- Events ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                for arrow in active_arrows:
                    if not arrow.hit and not arrow.missed and arrow.is_in_window(ct):
                        arrow.hit      = True
                        arrow.hit_time = ct
                        scores["You"] += 10
                        hit_flash.update(active=True, timer=ct,
                                         label="PERFECT!", color=YELLOW)
                        you_idx = players.index("You")
                        amazing_popups[you_idx] = {"active": True, "timer": ct}
                        break

    # --- Spawn arrows ---
    for jt in jump_times:
        spawn_t = jt - ARROW_TRAVEL
        if spawn_t not in used_spawn_times and ct >= spawn_t:
            used_spawn_times.add(spawn_t)
            a = Arrow(spawn_t)
            for ai_name, offset in ai_offsets.items():
                a.ai_press_times[ai_name] = jt + offset
                a.ai_scored[ai_name]      = False
            active_arrows.append(a)

    # --- AI scoring ---
    for arrow in active_arrows:
        for ai_name, press_t in arrow.ai_press_times.items():
            if not arrow.ai_scored[ai_name] and ct >= press_t:
                elapsed = press_t - arrow.spawn_time
                if abs(elapsed - ARROW_TRAVEL) < ARROW_WINDOW + 0.2:
                    scores[ai_name] += 10
                    ai_idx = players.index(ai_name)
                    amazing_popups[ai_idx] = {"active": True, "timer": ct}
                arrow.ai_scored[ai_name] = True

    # --- Expire arrows ---
    for arrow in active_arrows:
        if not arrow.hit and arrow.is_expired(ct):
            arrow.missed = True
    active_arrows = [a for a in active_arrows
                     if not (a.missed and a.is_expired(ct))]

    # --- Video frame (native FPS) ---
    if ct - last_frame_time >= frame_duration:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cached_frame_surf = pygame.surfarray.make_surface(np.rot90(frame))
        last_frame_time = ct

    # --- Draw ---
    screen.fill(DARK)

    if cached_frame_surf:
        screen.blit(cached_frame_surf, (0, 0))

    draw_ghost_arrow(screen, ARROW_X, ARROW_TARGET_Y)

    for arrow in active_arrows:
        if arrow.missed:
            continue
        if arrow.hit:
            age = ct - arrow.hit_time
            if age < 0.28:
                sc    = 1.0 + age * 3.5
                alpha = int(255 * (1 - age / 0.28))
                draw_arrow_shape(screen, ARROW_X, ARROW_TARGET_Y,
                                 YELLOW, scale=sc, alpha=alpha)
            continue
        y      = arrow.get_y(ct)
        in_win = arrow.is_in_window(ct)
        color  = YELLOW if in_win else RED
        draw_arrow_shape(screen, ARROW_X, int(y), color)
        draw_arrow_shape(screen, ARROW_X, int(y), color, scale=1.18, alpha=50)

    draw_hud(screen, ct)

    draw_hit_flash(screen, hit_flash, ct)

    hint = small_font.render("SPACE to hit!", True, (200, 200, 200))
    screen.blit(hint, (ARROW_X - hint.get_width() // 2, WINDOW_HEIGHT - 22))

    pygame.display.flip()

pygame.quit()
cap.release()