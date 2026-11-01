"""Undas puzzle reels for GASERA, drawn entirely in code (PIL + numpy + ffmpeg).

    python make_undas.py kandila   -> out/undas_kandila.mp4   (count the candles that get lit)
    python make_undas.py mata      -> out/undas_mata.mp4      (count the pairs of eyes)
    add --preview for a 540x960 / 15 fps draft

1080x1920, 30 fps, H.264 + AAC, 15 s. Night-village palette taken from the GASERA reels:
navy sky, pale moon, pine line, nipa huts with warm windows, brown ground, cream tombs.
Captions follow the house style: heavy yellow text with a thick black outline, plus a small
white English line. Text stays out of the top 180 px, the bottom 420 px and the right rail.
"""
import math
import os
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_BLACK = '/usr/share/fonts/opentype/inter/Inter-Black.otf'
FONT_BOLD = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
DUR = 15.0
YELLOW = (255, 210, 63)
WHITE = (255, 255, 255)

W, H, FPS = 1080, 1920, 30


# ---------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out_back(t):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def radial_sprite(radius, color, power=2.0):
    """RGBA glow: color with alpha falling off from the centre."""
    s = radius * 2
    y, x = np.mgrid[0:s, 0:s]
    d = np.sqrt((x - radius + 0.5) ** 2 + (y - radius + 0.5) ** 2) / radius
    a = np.clip(1 - d, 0, 1) ** power
    arr = np.zeros((s, s, 4), np.uint8)
    arr[..., 0], arr[..., 1], arr[..., 2] = color
    arr[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')


def paste_glow(img, sprite, cx, cy, alpha=1.0):
    if alpha <= 0:
        return
    sp = sprite
    if alpha < 1:
        sp = sprite.copy()
        a = np.asarray(sp.getchannel('A'), np.float32) * alpha
        sp.putalpha(Image.fromarray(a.astype(np.uint8)))
    img.alpha_composite(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)))


# ---------------------------------------------------------------- scene
class Scene:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.horizon = 1060
        self.tombs = []          # (x, y, scale)
        self.candles = []        # (x, y, scale)
        self.bg = self.build_background()
        self.glow_small = radial_sprite(105, (255, 170, 60), 2.2)
        self.glow_big = radial_sprite(160, (255, 150, 50), 2.6)
        self.fly_glow = radial_sprite(22, (210, 255, 120), 1.6)
        self.eye_glow = radial_sprite(34, (255, 235, 120), 1.8)
        self.stars = [(self.rng.uniform(0, W), self.rng.uniform(0, 820), self.rng.uniform(1.5, 3.5),
                       self.rng.uniform(0, 6.28)) for _ in range(70)]
        self.flies = [(self.rng.uniform(80, 1000), self.rng.uniform(900, 1500), self.rng.uniform(0, 6.28),
                       self.rng.uniform(0.3, 0.7)) for _ in range(9)]

    def build_background(self):
        S = 2  # supersample
        w, h = W * S, H * S
        img = Image.new('RGB', (w, h))
        px = np.zeros((h, w, 3), np.float32)
        # sky gradient
        top, bot = np.array([16, 20, 52]), np.array([46, 50, 96])
        hz = self.horizon * S
        g = np.linspace(0, 1, hz)[:, None]
        px[:hz] = (top * (1 - g) + bot * g)[:, None, :]
        # ground gradient
        g2 = np.linspace(0, 1, h - hz)[:, None]
        gt, gb = np.array([104, 74, 52]), np.array([70, 48, 34])
        px[hz:] = (gt * (1 - g2) + gb * g2)[:, None, :]
        img = Image.fromarray(px.astype(np.uint8))
        d = ImageDraw.Draw(img)
        # moon (cut by the right edge like the reference reels)
        d.ellipse([w - 330 * S, 40 * S, w + 130 * S, 500 * S], fill=(246, 232, 184))
        # far pine line
        for i in range(-1, 26):
            x = i * 46 * S + int(self.rng.integers(-10, 10)) * S
            ht = int(self.rng.integers(170, 290)) * S
            d.polygon([(x, hz + 6 * S), (x + 42 * S, hz - ht), (x + 84 * S, hz + 6 * S)], fill=(24, 36, 54))
        # nipa huts (left), stilts + warm windows
        for hx, hy, sc in [(40, 1010, 1.0), (300, 985, 0.75), (520, 1000, 0.6)]:
            hx, hy = hx * S, hy * S
            bw, bh = int(230 * sc * S), int(120 * sc * S)
            for k in range(4):
                sx = hx + int(k * bw / 3)
                d.rectangle([sx - 4 * S, hy, sx + 4 * S, hy + int(90 * sc * S)], fill=(92, 70, 48))
            d.rectangle([hx, hy - bh, hx + bw, hy], fill=(206, 170, 112))
            d.polygon([(hx - 30 * sc * S, hy - bh + 6 * S), (hx + bw / 2, hy - bh - 110 * sc * S),
                       (hx + bw + 30 * sc * S, hy - bh + 6 * S)], fill=(226, 196, 132))
            ww = int(56 * sc * S)
            d.rectangle([hx + bw / 2 - ww / 2, hy - bh + 30 * sc * S, hx + bw / 2 + ww / 2,
                         hy - bh + 30 * sc * S + ww], fill=(255, 205, 110))
        # balete tree (right): trunk, canopy, hanging roots
        tx = 860 * S
        d.rectangle([tx, 520 * S, tx + 150 * S, hz + 40 * S], fill=(46, 34, 30))
        for cx, cy, r in [(900, 470, 230), (1060, 520, 200), (760, 560, 170), (980, 330, 180)]:
            d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=(22, 40, 34))
        for i in range(14):
            rx = (700 + i * 28) * S
            d.line([(rx, 600 * S), (rx + int(self.rng.integers(-15, 15)) * S, (760 + int(self.rng.integers(0, 260))) * S)],
                   fill=(38, 30, 28), width=4 * S)
        # tombs: rows in perspective
        rows = [(1180, 0.55, 6), (1310, 0.72, 5), (1460, 0.9, 4)]
        for y, sc, n in rows:
            xs = np.linspace(110, 860, n) + self.rng.uniform(-30, 30, n)
            for x in xs:
                self.tombs.append((float(x), float(y), sc))
        for x, y, sc in self.tombs:
            tw, th = 120 * sc, 150 * sc
            X, Y = x * S, y * S
            d.ellipse([X - tw / 2 * S, Y - th * S - tw / 2 * S, X + tw / 2 * S, Y - th * S + tw / 2 * S], fill=(238, 226, 200))
            d.rectangle([X - tw / 2 * S, Y - th * S, X + tw / 2 * S, Y], fill=(238, 226, 200))
            d.rectangle([X - tw / 2 * S, Y - 10 * S, X + tw / 2 * S, Y], fill=(196, 182, 156))
            cw = 10 * sc
            d.rectangle([X - cw * S, Y - th * 0.85 * S, X + cw * S, Y - th * 0.35 * S], fill=(176, 160, 134))
            d.rectangle([X - cw * 3 * S, Y - th * 0.72 * S, X + cw * 3 * S, Y - th * 0.62 * S], fill=(176, 160, 134))
            # two candle spots in front of each tomb
            for off in (-0.32, 0.32):
                self.candles.append((x + off * tw * 1.25, y + 26 * sc, sc))
        img = img.resize((W, H), Image.LANCZOS)
        return img.convert('RGBA')

    # dynamic pieces -------------------------------------------------------
    def draw_candle(self, img, d, x, y, sc, lit, t, k):
        cw, ch = 30 * sc, 74 * sc
        d.rectangle([x - cw / 2, y - ch, x + cw / 2, y], fill=(250, 244, 226))
        d.line([(x, y - ch), (x, y - ch - 7 * sc)], fill=(40, 30, 20), width=max(2, int(3 * sc)))
        if lit <= 0:
            return
        f = 1 + 0.12 * math.sin(t * 17 + k * 2.1) + 0.06 * math.sin(t * 31 + k)
        s = sc * ease_out_back(clamp(lit)) * f
        fy = y - ch - 6 * sc
        paste_glow(img, self.glow_small, x, fy - 10 * s, 0.85 * clamp(lit))
        fh, fw = 54 * s, 20 * s
        d.polygon([(x, fy - fh), (x - fw, fy - fh * 0.35), (x - fw * 0.7, fy), (x + fw * 0.7, fy),
                   (x + fw, fy - fh * 0.35)], fill=(255, 150, 40))
        d.polygon([(x, fy - fh * 0.7), (x - fw * 0.5, fy - fh * 0.25), (x, fy - 2), (x + fw * 0.5, fy - fh * 0.25)],
                  fill=(255, 236, 140))

    def draw_ambient(self, img, d, t):
        for sx, sy, r, ph in self.stars:
            a = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(t * 2.2 + ph))
            c = int(255 * a)
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(c, c, int(c * 0.9)))
        for fx, fy, ph, sp in self.flies:
            x = fx + 60 * math.sin(t * sp + ph)
            y = fy + 40 * math.sin(t * sp * 1.7 + ph * 2)
            a = clamp(0.5 + 0.5 * math.sin(t * 3 + ph * 3))
            paste_glow(img, self.fly_glow, x, y, a)

    def draw_kapre(self, img, d, t, t0, cigar=True):
        """Kapre in the balete tree: eyes open, then cigar ember brightens with a puff."""
        k = clamp((t - t0) / 0.6)
        if k <= 0:
            return
        ex, ey = 905, 520
        body = Image.new('RGBA', img.size, (0, 0, 0, 0))
        bd = ImageDraw.Draw(body)
        a = int(235 * k)
        bd.ellipse([ex - 105, ey - 115, ex + 105, ey + 95], fill=(12, 10, 10, a))          # head
        bd.ellipse([ex - 175, ey + 40, ex + 175, ey + 330], fill=(12, 10, 10, a))         # shoulders
        bd.polygon([(ex - 150, ey + 200), (ex + 150, ey + 200), (ex + 110, ey + 520), (ex - 110, ey + 520)],
                   fill=(12, 10, 10, a))
        for i in range(9):  # shaggy hair fringe
            hx = ex - 110 + i * 27
            bd.polygon([(hx, ey - 60), (hx + 26, ey - 60), (hx + 13, ey + 150 + (i % 3) * 25)], fill=(12, 10, 10, a))
        body = body.filter(ImageFilter.GaussianBlur(3))
        img.alpha_composite(body)
        paste_glow(img, self.eye_glow, ex - 34, ey, k)
        paste_glow(img, self.eye_glow, ex + 34, ey, k)
        blink = 1.0 if (t - t0) % 2.4 > 0.12 else 0.15
        for dx in (-34, 34):
            d.ellipse([ex + dx - 22, ey - 16 * blink, ex + dx + 22, ey + 16 * blink], fill=(255, 240, 150))
            d.ellipse([ex + dx - 8, ey - 8 * blink, ex + dx + 8, ey + 8 * blink], fill=(30, 20, 10))
        if cigar:
            c = clamp((t - t0 - 0.5) / 0.4)
            pulse = 0.6 + 0.4 * math.sin(t * 5)
            cx, cy = ex + 60, ey + 70
            d.line([(ex + 10, ey + 72), (cx, cy)], fill=(120, 80, 50), width=10)
            paste_glow(img, self.glow_big, cx, cy, c * pulse * 0.9)
            d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], fill=(255, 120, 40) if c > 0 else (90, 60, 40))
            for i in range(3):  # smoke puffs
                p = ((t - t0) * 0.5 + i / 3) % 1
                r = 14 + 30 * p
                a = int(110 * (1 - p) * c)
                if a > 0:
                    ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
                    ImageDraw.Draw(ov).ellipse([cx + 10 - r, cy - 30 - 160 * p - r, cx + 10 + r, cy - 30 - 160 * p + r],
                                               fill=(220, 220, 225, a))
                    img.alpha_composite(ov)

    def draw_eyes(self, img, d, x, y, k, blink, size=1.0):
        if k <= 0:
            return
        sep = 22 * size
        for dx in (-sep, sep):
            paste_glow(img, self.eye_glow, x + dx, y, k * 0.8)
            d.ellipse([x + dx - 10 * size, y - 8 * size * blink, x + dx + 10 * size, y + 8 * size * blink],
                      fill=(255, 238, 140))
            d.ellipse([x + dx - 3.5 * size, y - 3.5 * size * blink, x + dx + 3.5 * size, y + 3.5 * size * blink],
                      fill=(25, 18, 10))


# ---------------------------------------------------------------- captions
class Captions:
    def __init__(self):
        self.cache = {}

    def block(self, big, small, size):
        key = (big, small, size)
        if key in self.cache:
            return self.cache[key]
        fb = ImageFont.truetype(FONT_BLACK, size)
        fs = ImageFont.truetype(FONT_BOLD, int(size * 0.42))
        lines = big.split('\n')
        tmp = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        widths = [tmp.textbbox((0, 0), ln, font=fb, stroke_width=12)[2] for ln in lines]
        lh = int(size * 1.12)
        sw = tmp.textbbox((0, 0), small, font=fs, stroke_width=6)[2] if small else 0
        bw = max(widths + [sw]) + 20
        bh = lh * len(lines) + (int(size * 0.62) if small else 0) + 30
        im = Image.new('RGBA', (bw, bh), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        y = 6
        for ln, wd in zip(lines, widths):
            d.text(((bw - wd) / 2, y), ln, font=fb, fill=self.color, stroke_width=12, stroke_fill=(0, 0, 0))
            y += lh
        if small:
            d.text(((bw - sw) / 2, y + 4), small, font=fs, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
        self.cache[key] = im
        return im

    def draw(self, img, big, small, t_in, t, y, color=YELLOW, size=78):
        self.color = color
        im = self.block(big, small, size)
        k = clamp((t - t_in) / 0.28)
        s = 0.6 + 0.4 * ease_out_back(k)
        if s != 1:
            im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
        x = (W - 60 - im.width) // 2  # nudge left of the right-hand UI rail
        img.alpha_composite(im, (max(10, x), int(y - im.height / 2)))


def logo(img):
    f = ImageFont.truetype(FONT_BLACK, 34)
    d = ImageDraw.Draw(img)
    d.text((48, 196), 'GASERA', font=f, fill=(242, 196, 92), stroke_width=4, stroke_fill=(0, 0, 0))


# ---------------------------------------------------------------- audio
def make_audio(path, events, dur=DUR, sr=44100, seed=1):
    """Night ambience (crickets + low drone) with a soft chime at each event time.
    events: list of (time, kind) where kind is 'chime', 'pop' or 'kapre'."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = 0.04 * np.sin(2 * np.pi * 55 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * t))
    out += 0.025 * np.sin(2 * np.pi * 82.4 * t)
    # crickets: 4.6 kHz carrier gated in short chirps
    for c in range(3):
        rate = 2.4 + c * 0.7
        gate = (np.sin(2 * np.pi * rate * t + c) > 0.6).astype(np.float32)
        trill = (np.sin(2 * np.pi * 38 * t) > 0).astype(np.float32)
        out += 0.012 * gate * trill * np.sin(2 * np.pi * (4400 + 300 * c) * t)
    out += 0.004 * rng.standard_normal(n)
    for et, kind in events:
        i0 = int(et * sr)
        if i0 >= n:
            continue
        L = min(n - i0, int(1.6 * sr))
        tt = np.arange(L) / sr
        if kind == 'chime':
            f = rng.choice([784, 880, 988, 1047, 1175])
            s = 0.12 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 4) + 0.05 * np.sin(2 * np.pi * f * 2 * tt) * np.exp(-tt * 7)
        elif kind == 'pop':
            s = 0.10 * np.sin(2 * np.pi * (520 + 300 * np.exp(-tt * 20)) * tt) * np.exp(-tt * 10)
        else:  # kapre: low swell + rumble
            env = np.minimum(tt / 0.3, 1) * np.exp(-tt * 1.6)
            s = 0.22 * env * (np.sin(2 * np.pi * 62 * tt) + 0.5 * np.sin(2 * np.pi * 93 * tt))
        out[i0:i0 + L] += s
    fade = np.minimum(1, np.minimum(t / 0.5, (dur - t) / 0.8))
    out = np.clip(out * fade, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((out * 32767).astype('<i2').tobytes())


# ---------------------------------------------------------------- videos
def plan_kandila(scene):
    """12 candles get lit (some in pairs), 2 stay dark. Answer: 12."""
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(scene.candles))
    lit_ids, dark_ids = idx[:12], idx[12:14]
    times = [2.9, 3.6, 4.2, 4.2, 5.0, 5.7, 6.1, 6.9, 7.5, 7.5, 8.4, 9.2]
    plan = {int(i): tm for i, tm in zip(lit_ids, times)}
    shown = set(plan) | {int(i) for i in dark_ids}
    return plan, shown


def render(kind, preview=False):
    os.makedirs('out', exist_ok=True)
    scene = Scene(seed=3)
    caps = Captions()
    fps = 15 if preview else FPS
    ow, oh = (540, 960) if preview else (W, H)
    name = f'undas_{kind}' + ('_preview' if preview else '')
    wav = f'out/{name}.wav'
    mp4 = f'out/{name}.mp4'

    if kind == 'kandila':
        plan, shown = plan_kandila(scene)
        events = [(tm, 'chime') for tm in plan.values()] + [(10.2, 'kapre')]
        answer = 12
    else:
        # pairs of eyes: (x, y, t_on, t_off, size); the last one is the Kapre (big, in the tree)
        eyes = [(150, 1000, 2.8, 5.2, 1.0), (470, 900, 3.4, 6.6, 0.9), (640, 1130, 4.1, 6.0, 0.8),
                (230, 1260, 4.9, 7.7, 1.0), (560, 1400, 5.6, 8.4, 1.1), (90, 1430, 6.4, 9.3, 1.1),
                (760, 1240, 7.2, 9.6, 0.9), (380, 1160, 8.0, 10.0, 0.9)]
        events = [(e[2], 'pop') for e in eyes] + [(10.2, 'kapre')]
        answer = len(eyes) + 1
    make_audio(wav, events)

    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{ow}x{oh}', '-r', str(fps),
           '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nframes = int(DUR * fps)
    for fi in range(nframes):
        t = fi / fps
        img = scene.bg.copy()
        d = ImageDraw.Draw(img)
        scene.draw_ambient(img, d, t)
        if kind == 'kandila':
            for i in sorted(shown, key=lambda j: scene.candles[j][1]):
                x, y, sc = scene.candles[i]
                lit = clamp((t - plan[i]) / 0.25) if i in plan else 0
                scene.draw_candle(img, d, x, y, sc, lit, t, i)
        else:
            for i in (1, 6, 11, 17):  # a few steady candles for mood
                x, y, sc = scene.candles[i]
                scene.draw_candle(img, d, x, y, sc, 1, t, i)
            for (x, y, on, off, sz) in eyes:
                if on <= t <= off:
                    k = clamp((t - on) / 0.25) * clamp((off - t) / 0.25)
                    bl = 0.15 if ((t - on) % 1.9) < 0.1 else 1.0
                    scene.draw_eyes(img, d, x, y, k, bl, sz * 1.7)
        scene.draw_kapre(img, d, t, 10.2, cigar=True)

        # slow push-in toward the cemetery
        z = 1 + 0.06 * (t / DUR)
        if z > 1.001:
            cw, ch = W / z, H / z
            x0, y0 = (W - cw) / 2, (H - ch) * 0.7
            img = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR)

        # captions (house style)
        logo(img)
        if kind == 'kandila':
            if t < 2.7:
                caps.draw(img, 'UNDAS SA GASERA', '', 0.1, t, 300, WHITE, 52)
                caps.draw(img, 'BILANGIN ANG\nMGA KANDILA!', 'Count the candles!', 0.4, t, 450)
            elif t < 10.1:
                caps.draw(img, 'ILAN ANG\nNASINDIHAN?', 'How many get lit?', 2.7, t, 400)
            elif t < 12.6:
                caps.draw(img, 'TEKA... MAY ISA\nPA SA PUNO?', 'Wait... one more in the tree?', 10.1, t, 400, WHITE)
            else:
                caps.draw(img, 'KANDILA LANG\nANG BILANGIN!', 'Candles only. Comment your answer!', 12.6, t, 400)
        else:
            if t < 2.7:
                caps.draw(img, 'UNDAS SA GASERA', '', 0.1, t, 300, WHITE, 52)
                caps.draw(img, 'MAY NAKATINGIN\nSA \'YO...', 'Someone is watching you...', 0.4, t, 450)
            elif t < 10.1:
                caps.draw(img, 'ILANG PARES\nNG MATA?', 'How many pairs of eyes?', 2.7, t, 400)
            elif t < 12.6:
                caps.draw(img, 'PATI SI KAPRE!', 'Even Kapre!', 10.1, t, 400, WHITE)
            else:
                caps.draw(img, 'ILAN ANG NAKITA\nMO? I-COMMENT!', 'How many did you spot? Comment!', 12.6, t, 400)

        frame = img.convert('RGB')
        if preview:
            frame = frame.resize((ow, oh), Image.BILINEAR)
        ff.stdin.write(frame.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f'{mp4}  answer={answer}')
    return mp4, answer


if __name__ == '__main__':
    kind = sys.argv[1]
    render(kind, preview='--preview' in sys.argv)
