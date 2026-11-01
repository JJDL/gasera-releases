"""Two more GASERA Undas reels tied to this week's Philippine trends (drawn in code).

    python make_undas2.py nakalimutan  -> out/undas_nakalimutan.mp4  ("forgotten graves" story)
    python make_undas2.py ulan         -> out/undas_ulan.mp4         (Undas rain; Kapre saves the candles)

Reuses the night-cemetery scene, captions and audio from make_undas.py.
"""
import math
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

import make_undas as U
from make_undas import W, H, FPS, DUR, YELLOW, WHITE, clamp


def make_audio2(path, events, rain=False, dur=DUR, sr=44100):
    U.make_audio(path, events, dur, sr)
    if not rain:
        return
    import wave
    with wave.open(path, 'rb') as w:
        a = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(np.float32) / 32767
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(len(a))
    k = np.ones(6) / 6  # soft low-pass for a rain hiss
    noise = np.convolve(noise, k, 'same')
    t = np.arange(len(a)) / sr
    env = np.minimum(1, t / 1.0) * np.minimum(1, (dur - t) / 0.8)
    a = np.clip(a + 0.09 * noise * env, -1, 1)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((a * 32767).astype('<i2').tobytes())


def bouquet(d, x, y, s, k):
    if k <= 0:
        return
    s = s * U.ease_out_back(clamp(k))
    d.line([(x, y), (x, y - 40 * s)], fill=(60, 120, 60), width=max(2, int(4 * s)))
    for dx, dy, c in [(-12, -46, (240, 220, 90)), (12, -46, (250, 250, 245)), (0, -58, (240, 120, 150)),
                      (-6, -38, (250, 250, 245)), (8, -36, (240, 220, 90))]:
        r = 9 * s
        d.ellipse([x + dx * s - r, y + dy * s - r, x + dx * s + r, y + dy * s + r], fill=c)


def render(kind):
    scene = U.Scene(seed=3)
    caps = U.Captions()
    name = f'undas_{kind}'
    wav, mp4 = f'out/{name}.wav', f'out/{name}.mp4'
    # the "forgotten" tomb: front row, second from left
    front = [t for t in scene.tombs if t[2] == 0.9]
    fx, fy, fs = sorted(front)[1]
    near = [i for i, c in enumerate(scene.candles) if abs(c[0] - fx) < 80 and abs(c[1] - fy) < 40]
    others = [i for i in range(len(scene.candles)) if i not in near]

    if kind == 'nakalimutan':
        events = [(8.0, 'kapre'), (9.0, 'chime'), (10.2, 'chime'), (11.0, 'chime'), (11.6, 'chime')]
        make_audio2(wav, events)
    else:
        rng = np.random.default_rng(11)
        out_t = {i: 2.6 + rng.uniform(0, 3.4) for i in others}
        events = [(tm, 'pop') for tm in sorted(out_t.values())[::3]] + [(6.6, 'kapre')]
        events += [(9.0 + 0.25 * j, 'chime') for j in range(6)]
        make_audio2(wav, events, rain=True)
        drops = np.random.default_rng(2).uniform([0, 0], [W, H], size=(260, 2))

    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-', '-i', wav, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', '-shortest', mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(int(DUR * FPS)):
        t = fi / FPS
        img = scene.bg.copy()
        d = ImageDraw.Draw(img)
        scene.draw_ambient(img, d, t)

        if kind == 'nakalimutan':
            for i in sorted(range(len(scene.candles)), key=lambda j: scene.candles[j][1]):
                x, y, sc = scene.candles[i]
                if i in near:
                    lit = clamp((t - (9.0 if i == near[0] else 10.2 + 0.8 * near.index(i))) / 0.3)
                else:
                    lit = 1
                scene.draw_candle(img, d, x, y, sc, lit, t, i)
            for (x, y, sc) in scene.tombs:  # flowers on every visited tomb from the start
                if (x, y, sc) != (fx, fy, fs):
                    bouquet(d, x, y + 6, sc, 1)
            bouquet(d, fx, fy + 6, fs * 1.2, (t - 11.6) / 0.4)
            # darken the forgotten tomb until it is visited
            dark = 1 - clamp((t - 9.0) / 2.5)
            if dark > 0:
                ov = Image.new('RGBA', img.size, (0, 0, 0, 0))
                ImageDraw.Draw(ov).ellipse([fx - 110, fy - 210, fx + 110, fy + 40], fill=(8, 8, 20, int(120 * dark)))
                img.alpha_composite(ov)
            # fireflies gather at the tomb from 3.5 s
            g = clamp((t - 3.5) / 4.0)
            for j in range(10):
                a0 = j * 0.63 + t * 0.9
                rx, ry = 260 - 170 * g, 300 - 220 * g
                cx = fx + rx * math.cos(a0) + (1 - g) * (j * 70 - 300)
                cy = fy - 90 + ry * math.sin(a0 * 1.3) - (1 - g) * 200
                U.paste_glow(img, scene.fly_glow, cx, cy, 0.5 + 0.5 * math.sin(t * 4 + j))
            scene.draw_kapre(img, d, t, 8.0, cigar=False)
        else:
            # rain darkens the sky a little
            ov = Image.new('RGBA', img.size, (20, 26, 40, 70))
            img.alpha_composite(ov)
            d = ImageDraw.Draw(img)
            leaf = clamp((t - 6.8) / 0.8)
            for i in sorted(range(len(scene.candles)), key=lambda j: scene.candles[j][1]):
                x, y, sc = scene.candles[i]
                if i in out_t:
                    lit = clamp((out_t[i] - t) / 0.2) if t < out_t[i] + 0.01 else 0
                    if t > 9.0:  # relit under the leaf, one row at a time
                        lit = clamp((t - (9.0 + (y - 1100) / 400)) / 0.3)
                else:
                    lit = 1
                scene.draw_candle(img, d, x, y, sc, lit, t, i)
            scene.draw_kapre(img, d, t, 6.4, cigar=False)
            if leaf > 0:  # Kapre's long arm holds a giant banana leaf over the cemetery
                lx, ly = 470, 980 - 40 * (1 - leaf)
                d.line([(860, 640), (lx + 260, ly + 10)], fill=(14, 12, 12), width=26)
                pts = [(lx - 470 * leaf, ly + 40), (lx - 200 * leaf, ly - 60), (lx + 300 * leaf, ly - 50),
                       (lx + 470 * leaf, ly + 40), (lx, ly + 10)]
                d.polygon(pts, fill=(52, 128, 62))
                d.line([(lx - 460 * leaf, ly + 36), (lx + 460 * leaf, ly + 36)], fill=(110, 170, 90), width=8)
                for k in range(-8, 9):
                    d.line([(lx + k * 52 * leaf, ly + 34), (lx + k * 60 * leaf, ly - 40)], fill=(80, 150, 80), width=3)
            for j, (x0, y0) in enumerate(drops):  # rain streaks (stop under the leaf)
                y = (y0 + t * 1900) % H
                x = (x0 - t * 220) % W
                if leaf > 0.9 and 990 < y < 1650 and 40 < x < 960:
                    continue
                d.line([(x, y), (x - 9, y + 46)], fill=(170, 190, 230), width=3)

        z = 1 + 0.05 * (t / DUR)
        cw, ch = W / z, H / z
        x0, y0 = (W - cw) / 2, (H - ch) * 0.7
        img = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((W, H), Image.BILINEAR)

        U.logo(img)
        if kind == 'nakalimutan':
            if t < 3.5:
                caps.draw(img, 'UNDAS SA GASERA', '', 0.1, t, 300, WHITE, 52)
                caps.draw(img, 'WALANG BULAKLAK.\nWALANG DUMALAW...', 'No flowers. No visitors...', 0.4, t, 450, WHITE)
            elif t < 7.8:
                caps.draw(img, 'PERO MAY\nNAKAPANSIN...', 'But someone noticed...', 3.5, t, 400, WHITE)
            elif t < 12.0:
                caps.draw(img, 'HINDI KA\nNAG-IISA.', 'You are not alone.', 7.8, t, 400)
            else:
                caps.draw(img, 'ALALAHANIN DIN\nANG NAKALIMUTAN', 'This Undas, remember the forgotten too.', 12.0, t, 400)
        else:
            if t < 2.8:
                caps.draw(img, 'UNDAS SA GASERA', '', 0.1, t, 300, WHITE, 52)
                caps.draw(img, 'UNDAS NA,\nUMUULAN PA!', "It's Undas... and it's raining!", 0.4, t, 450)
            elif t < 6.4:
                caps.draw(img, 'NAMATAY ANG\nMGA KANDILA!', 'The candles went out!', 2.8, t, 400, WHITE)
            elif t < 10.5:
                caps.draw(img, 'TAWAGIN\nSI KAPRE!', 'Call the Kapre!', 6.4, t, 400)
            else:
                caps.draw(img, 'SALAMAT, KAPRE!\nINGAT SA BIYAHE!', 'Thanks, Kapre! Stay safe this Undas.', 10.5, t, 400)
        ff.stdin.write(img.convert('RGB').tobytes())
    ff.stdin.close()
    ff.wait()
    print(mp4)


if __name__ == '__main__':
    render(sys.argv[1])
