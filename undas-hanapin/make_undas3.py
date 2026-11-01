"""Nov 2 batch of GASERA Undas reels, drawn in code. Reuses make_undas.py.

    python make_undas3.py sagot_kandila   answer reveal: counts the 12 candles as they light
    python make_undas3.py sagot_mata      answer reveal: counts the 9 pairs of eyes
    python make_undas3.py crowd_nuno      find the Nuno in the Undas crowd (reveal at the end)
    python make_undas3.py crowd_duwende   find the duwende in the Undas crowd (reveal at the end)
    python make_undas3.py diff_a          spot 5 differences (reveal at the end)
    python make_undas3.py diff_b          spot 5 differences, second set
"""
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import make_undas as U
from make_undas import W, H, FPS, DUR, YELLOW, WHITE, clamp

TEAL = (22, 120, 128)


def badge(img, text, y, size=60):
    f = ImageFont.truetype(U.FONT_BLACK, size)
    d = ImageDraw.Draw(img)
    tw = d.textbbox((0, 0), text, font=f)[2]
    x = (W - 60 - tw) // 2
    d.rounded_rectangle([x - 26, y - 12, x + tw + 26, y + size + 14], radius=18, fill=TEAL)
    d.text((x, y - 4), text, font=f, fill=WHITE)


def ring(d, x, y, r, k, color=(255, 70, 70)):
    if k <= 0:
        return
    rr = r * (1.6 - 0.6 * clamp(k))
    d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=color, width=9)


def encode(name, events, draw_frame):
    wav, mp4 = f'out/{name}.wav', f'out/{name}.mp4'
    U.make_audio(wav, events)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(int(DUR * FPS)):
        ff.stdin.write(draw_frame(fi / FPS).convert('RGB').tobytes())
    ff.stdin.close()
    ff.wait()
    print(mp4)


# ------------------------------------------------------------- answer reveals
def sagot(kind):
    scene = U.Scene(seed=3)
    caps = U.Captions()
    if kind == 'kandila':
        plan, shown = U.plan_kandila(scene)
        order = sorted(plan.items(), key=lambda kv: kv[1])
        # slower, one at a time so viewers can count along
        times = {i: 2.8 + 0.62 * n for n, (i, _) in enumerate(order)}
        total = len(times)
        events = [(tm, 'chime') for tm in times.values()]
    else:
        eyes = [(150, 1000, 1.0), (470, 900, 0.9), (640, 1130, 0.8), (230, 1260, 1.0), (560, 1400, 1.1),
                (90, 1430, 1.1), (760, 1240, 0.9), (380, 1160, 0.9)]
        times = {n: 2.8 + 0.8 * n for n in range(len(eyes))}
        kapre_t = 2.8 + 0.8 * len(eyes)
        total = len(eyes) + 1
        events = [(tm, 'pop') for tm in times.values()] + [(kapre_t, 'kapre')]

    def frame(t):
        img = scene.bg.copy()
        d = ImageDraw.Draw(img)
        scene.draw_ambient(img, d, t)
        if kind == 'kandila':
            n = sum(1 for tm in times.values() if t >= tm)
            for i in sorted(shown, key=lambda j: scene.candles[j][1]):
                x, y, sc = scene.candles[i]
                lit = clamp((t - times[i]) / 0.25) if i in times else 0
                scene.draw_candle(img, d, x, y, sc, lit, t, i)
                if i in times and 0 <= t - times[i] < 0.9:
                    ring(d, x, y - 60 * sc, 60, (t - times[i]) / 0.3)
            scene.draw_kapre(img, d, t, 11.0, cigar=True)
        else:
            n = sum(1 for tm in times.values() if t >= tm) + (1 if t >= kapre_t else 0)
            for i in (1, 6, 11, 17):
                x, y, sc = scene.candles[i]
                scene.draw_candle(img, d, x, y, sc, 1, t, i)
            for k, (x, y, sz) in enumerate(eyes):
                if t >= times[k]:
                    scene.draw_eyes(img, d, x, y, clamp((t - times[k]) / 0.25), 1.0, sz * 1.7)
                    if t - times[k] < 0.9:
                        ring(d, x, y, 70 * sz, (t - times[k]) / 0.3)
            scene.draw_kapre(img, d, t, kapre_t, cigar=True)
        U.logo(img)
        if t < 2.7:
            caps.draw(img, 'SAGOT!', 'Answer time!', 0.1, t, 330, YELLOW, 110)
            caps.draw(img, 'BILANGIN NATIN...' if kind == 'kandila' else 'ISA-ISAHIN NATIN...',
                      "Let's count together...", 0.5, t, 470, WHITE, 60)
        elif n < total:
            caps.draw(img, 'ILAN NA?', 'How many so far?', 2.7, t, 360, WHITE, 64)
        else:
            done = (min(times.values()) if False else 0)
            caps.draw(img, f'{total} {"KANDILA" if kind == "kandila" else "PARES NG MATA"}!',
                      'Tama ka ba? Were you right?', 0, t, 360, YELLOW, 72)
        badge(img, f'{n}/{total}', 520)
        return img

    encode(f'undas_sagot_{kind}', events, frame)


# ------------------------------------------------------------- crowd puzzle
def person(d, x, y, s, shirt, skin, hair, bob, umbrella=None):
    y = y + bob
    d.rounded_rectangle([x - 26 * s, y - 92 * s, x + 26 * s, y], radius=int(14 * s), fill=shirt)
    d.ellipse([x - 21 * s, y - 136 * s, x + 21 * s, y - 92 * s], fill=skin)
    d.chord([x - 22 * s, y - 140 * s, x + 22 * s, y - 104 * s], 180, 360, fill=hair)
    if umbrella:
        d.line([(x + 20 * s, y - 70 * s), (x + 20 * s, y - 190 * s)], fill=(40, 40, 40), width=max(2, int(4 * s)))
        d.pieslice([x + 20 * s - 70 * s, y - 230 * s, x + 20 * s + 70 * s, y - 150 * s], 180, 360, fill=umbrella)


def nuno(d, x, y, s, bob):
    y = y + bob
    d.rounded_rectangle([x - 20 * s, y - 60 * s, x + 20 * s, y], radius=int(10 * s), fill=(150, 104, 64))
    d.ellipse([x - 17 * s, y - 94 * s, x + 17 * s, y - 60 * s], fill=(222, 182, 140))
    d.polygon([(x - 16 * s, y - 72 * s), (x + 16 * s, y - 72 * s), (x, y - 36 * s)], fill=(245, 245, 240))  # beard
    d.polygon([(x - 34 * s, y - 88 * s), (x, y - 118 * s), (x + 34 * s, y - 88 * s)], fill=(196, 160, 90))  # salakot


def duwende(d, x, y, s, bob):
    y = y + bob
    d.rounded_rectangle([x - 18 * s, y - 56 * s, x + 18 * s, y], radius=int(9 * s), fill=(60, 130, 70))
    d.ellipse([x - 16 * s, y - 88 * s, x + 16 * s, y - 56 * s], fill=(230, 190, 150))
    d.polygon([(x - 18 * s, y - 80 * s), (x + 18 * s, y - 80 * s), (x + 4 * s, y - 132 * s)], fill=(210, 40, 40))  # red hat


def crowd(kind):
    scene = U.Scene(seed=3)
    caps = U.Captions()
    rng = np.random.default_rng(21 if kind == 'nuno' else 33)
    shirts = [(220, 220, 225), (40, 40, 48), (200, 60, 60), (60, 90, 170), (240, 200, 80), (90, 150, 90),
              (250, 250, 250), (30, 30, 30), (150, 60, 140)]
    skins = [(222, 182, 140), (198, 150, 110), (236, 200, 160), (176, 126, 90)]
    umbrellas = [(30, 30, 30), (200, 50, 60), (50, 90, 170), (240, 200, 70)] + [None] * 10
    people = []
    for _ in range(150):
        y = rng.uniform(1080, 1640)
        s = 0.45 + 0.75 * (y - 1080) / 560
        x = rng.uniform(40, 1040)
        people.append((x, y, s, shirts[rng.integers(len(shirts))], skins[rng.integers(len(skins))],
                       (25, 20, 18), umbrellas[rng.integers(len(umbrellas))], rng.uniform(0, 6.28)))
    # hide the target mid-crowd, partly behind someone
    tx, ty = (318, 1352) if kind == 'nuno' else (742, 1228)
    ts = 0.45 + 0.75 * (ty - 1080) / 560
    def blocks(p):
        x, y, s = p[0], p[1], p[2]
        if p[6] is not None and abs(x + 20 * s - tx) < 90 * s + 40 and y > ty - 40:
            return True          # umbrella canopy over the target
        return abs(x - tx) < 30 * s + 26 and ty < y < ty + 150   # body right in front
    people = [p for p in people if not blocks(p)]
    people.sort(key=lambda p: p[1])
    events = [(12.0, 'kapre'), (12.4, 'chime')]
    events += [(9.0 + k, 'pop') for k in range(3)]

    def frame(t):
        img = scene.bg.copy()
        d = ImageDraw.Draw(img)
        scene.draw_ambient(img, d, t)
        for i in (0, 4, 9, 13, 18, 22):
            x, y, sc = scene.candles[i]
            scene.draw_candle(img, d, x, y, sc, 1, t, i)
        drawn = False
        for (x, y, s, sh, sk, hr, um, ph) in people:
            if not drawn and y > ty:
                (nuno if kind == 'nuno' else duwende)(d, tx, ty, ts, 3 * math.sin(t * 3))
                drawn = True
            person(d, x, y, s, sh, sk, hr, 4 * math.sin(t * 2.5 + ph), um)
        if t >= 12.0:  # spotlight reveal
            k = clamp((t - 12.0) / 0.5)
            ov = Image.new('RGBA', img.size, (0, 0, 0, int(160 * k)))
            m = Image.new('L', img.size, 0)
            ImageDraw.Draw(m).ellipse([tx - 110, ty - 170, tx + 110, ty + 50], fill=255)
            ov.putalpha(Image.fromarray(np.minimum(np.asarray(ov.getchannel('A')),
                                                   255 - np.asarray(m)).astype(np.uint8)))
            img.alpha_composite(ov)
            d = ImageDraw.Draw(img)
            ring(d, tx, ty - 60, 120, k)
        U.logo(img)
        who = 'SI NUNO' if kind == 'nuno' else 'ANG DUWENDE'
        if t < 2.8:
            caps.draw(img, 'UNDAS SA GASERA', '', 0.1, t, 300, WHITE, 52)
            caps.draw(img, f'HANAPIN {who}\nSA DAMI NG TAO!',
                      'Find the ' + ('Nuno' if kind == 'nuno' else 'duwende') + ' in the Undas crowd!', 0.4, t, 450)
        elif t < 12.0:
            caps.draw(img, f'HANAPIN {who}!', 'Hint: ' + ('salakot + puting balbas' if kind == 'nuno' else 'pulang sombrero'),
                      2.8, t, 380, YELLOW, 70)
            left = max(0, math.ceil(12.0 - t))
            badge(img, f'{left}', 520, 70)
        else:
            caps.draw(img, 'ANDITO SIYA!', 'Nahanap mo ba? Comment "NAHANAP"!', 12.0, t, 380, YELLOW, 80)
        return img

    encode(f'undas_crowd_{kind}', events, frame)


# ------------------------------------------------------------- spot the difference
def diff(kind):
    scene = U.Scene(seed=3)
    caps = U.Captions()
    base = scene.bg.copy()
    d0 = ImageDraw.Draw(base)
    lit = [i for i in range(len(scene.candles)) if i % 2 == 0]
    for i in sorted(lit, key=lambda j: scene.candles[j][1]):
        x, y, sc = scene.candles[i]
        scene.draw_candle(base, d0, x, y, sc, 1, 0.3, i)
    Y0, PH = 860, 560                       # crop of the scene used for each panel
    A = base.crop((0, Y0, W, Y0 + PH))
    i_off = lit[3 if kind == 'a' else 7]
    base_b = scene.bg.copy()
    d1 = ImageDraw.Draw(base_b)
    for j in sorted(lit, key=lambda q: scene.candles[q][1]):
        x, y, sc = scene.candles[j]
        scene.draw_candle(base_b, d1, x, y, sc, 0 if j == i_off else 1, 0.3, j)
    B = base_b.crop((0, Y0, W, Y0 + PH))
    db = ImageDraw.Draw(B)
    rng = np.random.default_rng(5 if kind == 'a' else 9)
    diffs = []                              # (x, y) in panel coords

    def add(x, y):
        diffs.append((x, y))

    # 1) a lit hut window goes dark
    if kind == 'a':
        db.rectangle([127, 1010 - 120 - Y0 + 30, 157, 1010 - 120 - Y0 + 60], fill=(60, 50, 40)); add(142, 1010 - 120 - Y0 + 45)
    else:
        db.rectangle([355, 985 - 90 - Y0 + 22, 380, 985 - 90 - Y0 + 45], fill=(60, 50, 40)); add(368, 985 - 90 - Y0 + 34)
    # 2) a candle flame missing
    x, y, sc = scene.candles[i_off]
    add(x, y - 100 * sc - Y0)
    # 3) a tomb loses its cross
    tx, ty, ts = scene.tombs[(2 if kind == 'a' else 9)]
    db.rectangle([tx - 30 * ts, ty - 150 * ts * 0.85 - Y0, tx + 30 * ts, ty - 150 * ts * 0.35 - Y0], fill=(238, 226, 200))
    add(tx, ty - 90 * ts - Y0)
    # 4) glowing eyes on the balete trunk (only in B)
    ex, ey = (930, 1000 - Y0) if kind == 'a' else (890, 960 - Y0)
    for dx in (-14, 14):
        db.ellipse([ex + dx - 8, ey - 6, ex + dx + 8, ey + 6], fill=(255, 238, 140))
    add(ex, ey)
    # 5) a flower bouquet appears
    fx, fy, fs = scene.tombs[(7 if kind == 'a' else 5)]
    for dx, dy, c in [(-10, -30, (240, 220, 90)), (10, -30, (240, 120, 150)), (0, -44, (250, 250, 245))]:
        db.ellipse([fx + dx - 9, fy + dy - Y0 - 9, fx + dx + 9, fy + dy - Y0 + 9], fill=c)
    add(fx, fy - 32 - Y0)

    PA, PB = 380, 380 + PH + 24            # panel y positions in the frame
    reveal = [11.0 + 0.6 * k for k in range(5)]
    events = [(tm, 'chime') for tm in reveal]

    def frame(t):
        img = Image.new('RGBA', (W, H), (16, 20, 52, 255))
        img.alpha_composite(A, (0, PA))
        img.alpha_composite(B, (0, PB))
        d = ImageDraw.Draw(img)
        d.line([(0, PB - 12), (W, PB - 12)], fill=(255, 210, 63), width=6)
        for k, (x, y) in enumerate(diffs):
            if t >= reveal[k]:
                ring(d, x, PB + y, 46, (t - reveal[k]) / 0.3)
                ring(d, x, PA + y, 46, (t - reveal[k]) / 0.3)
        U.logo(img)
        if t < 11.0:
            caps.draw(img, '5 PAGKAKAIBA!\nKAYA MO BA?', 'Spot the 5 differences!', 0.2, t, 250, YELLOW, 66)
            left = max(0, math.ceil(11.0 - t))
            d = ImageDraw.Draw(img)
            f = ImageFont.truetype(U.FONT_BLACK, 54)
            d.text((W // 2 - 20, PB + PH + 30), str(left), font=f, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
        else:
            n = sum(1 for tm in reveal if t >= tm)
            caps.draw(img, f'SAGOT: {n}/5', 'Ilan ang nakita mo? Comment!', 11.0, t, 250, YELLOW, 66)
        return img

    encode(f'undas_diff_{kind}', events, frame)


if __name__ == '__main__':
    k = sys.argv[1]
    if k.startswith('sagot_'):
        sagot(k[6:])
    elif k.startswith('crowd_'):
        crowd(k[6:])
    else:
        diff(k[5:])
