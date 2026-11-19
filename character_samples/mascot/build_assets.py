"""Turn the generated pose illustrations into aligned, background-normalised web assets."""
import json, os
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = '/opt/cursor/artifacts/assets/'
OUT = '/tmp/mascot/out/'
os.makedirs(OUT, exist_ok=True)
W, H = 720, 1280
BG = np.array([2, 38, 65], dtype=np.float32)
# name -> (source file stem, shift x, shift y) ; shift moves the drawing so it lines up with idle
POSES = {
    'idle': ('idle_dark', 0, 0),
    'bow': ('bow', 0, 0),
    'sleeves': ('sleeves', 0, 0),
    'crown': ('crown', 0, 0),
    'present': ('present', 13, 0),
    'spread': ('spread', 0, 0),
    'spin': ('spin', 0, 18),
    'tilt': ('tilt', 0, -3),
    'wave': ('wave_dark', -1, 0),
    'cheer': ('cheer', 0, -8),
    'wink': ('wink', 0, 0),
    'toss': ('toss', 7, -4),
    'giggle': ('giggle', 0, -2),
}

def load(stem):
    return np.asarray(Image.open(f'{SRC}wanwu_pose_{stem}_v1.jpg').convert('RGB')).astype(np.float32)

def border_bg(a):
    b = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3), a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
    return np.median(b, axis=0)

def normalise(a, sx=0, sy=0):
    bg = border_bg(a)
    dist = np.linalg.norm(a - bg, axis=2)
    # 1 on the backdrop, fading to 0 on the drawing; edge pixels get a proportional share of the shift
    m = np.clip(1 - (dist - 8) / 56, 0, 1)[..., None]
    out = a + m * (BG - bg)
    out = np.where(dist[..., None] < 8, BG, out)
    if sx or sy:
        shifted = np.empty_like(out); shifted[:] = BG
        ys0, ys1 = max(0, sy), min(H, H + sy); xs0, xs1 = max(0, sx), min(W, W + sx)
        shifted[ys0:ys1, xs0:xs1] = out[ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx]
        out = shifted
    return np.clip(out, 0, 255)

def lum(a):
    return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

def match(template, image, x0, y0, x1, y1):
    """Top-left corner of the best normalised-correlation match of template inside image[y0:y1, x0:x1]."""
    t = lum(template); t = (t - t.mean()) / (t.std() + 1e-6)
    L = lum(image); th, tw = t.shape; best = (-2, 0, 0)
    for y in range(y0, y1 - th + 1, 1):
        for x in range(x0, x1 - tw + 1, 1):
            i = L[y:y + th, x:x + tw]
            s = float(((i - i.mean()) / (i.std() + 1e-6) * t).mean())
            if s > best[0]: best = (s, x, y)
    return best

images = {name: normalise(load(stem), sx, sy) for name, (stem, sx, sy) in POSES.items()}
idle = images['idle']

# --- globe: where each pose drew it (so the renderer can lift it out and animate one shared globe)
GLOBE = (562, 243)          # sphere centre in idle
T = 58                      # half-size of the matching template (sphere only)
template = idle[GLOBE[1] - T:GLOBE[1] + T, GLOBE[0] - T:GLOBE[0] + T]
meta = {'size': [W, H], 'bg': [int(v) for v in BG], 'poses': {}}
for name, image in images.items():
    # coarse search on a half-resolution copy, then refine at full resolution
    small_t = template[::2, ::2]; small_i = image[::2, ::2]
    s, x, y = match(small_t, small_i, 380 // 2, 40 // 2, W // 2, 440 // 2)
    x, y = x * 2, y * 2
    s, x, y = match(template, image, max(0, x - 4), max(0, y - 4), min(W, x + 2 * T + 5), min(H, y + 2 * T + 5))
    meta['poses'][name] = {'globe': [x + T, y + T], 'globeScore': round(s, 3)}
    print(f'{name:8s} globe centre=({x + T},{y + T}) score={s:.3f}')

# --- eyes-closed patch for blinking (the blink drawing matches idle everywhere except the eyes)
blink = normalise(load('blink'), 0, 1)
EYES = (284, 144, 404, 216)   # x0, y0, x1, y1 in idle
diff = np.linalg.norm(blink - idle, axis=2)
region = np.zeros((H, W), bool); region[EYES[1]:EYES[3], EYES[0]:EYES[2]] = True
mask = (diff > 60) & region
labels, count = ndimage.label(ndimage.binary_closing(mask, iterations=2))
sizes = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
keep = [i + 1 for i in np.argsort(sizes)[-2:]]          # the two eyes; hair-line jitter is dropped
mask = np.isin(labels, keep)
mask = ndimage.binary_dilation(mask, iterations=5)
soft = ndimage.gaussian_filter(mask.astype(np.float32), 2.2)
soft = np.clip(soft * 1.6, 0, 1)
ys, xs = np.nonzero(mask)
print('eye diff bbox', xs.min(), xs.max(), ys.min(), ys.max(), 'pixels', int(mask.sum()))
closed = idle * (1 - soft[..., None]) + blink * soft[..., None]
eyes_patch = closed[EYES[1]:EYES[3], EYES[0]:EYES[2]]

# --- parts atlas: globe sprite (pre-masked to its own shape) + eyes patch
GW, GH = 208, 168            # sprite box around the globe centre
gx0, gy0 = GLOBE[0] - GW // 2, GLOBE[1] - GH // 2
yy, xx = np.mgrid[gy0:gy0 + GH, gx0:gx0 + GW].astype(np.float32)
sphere = np.clip((70 - np.hypot(xx - GLOBE[0], yy - GLOBE[1])) / 8, 0, 1)
ang = np.deg2rad(-12.5); cx, cy = 560.5, 245.5
rx = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
ry = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
ring = np.clip((1.0 - np.hypot(rx / 96, ry / 30)) / 0.12, 0, 1)
sparkle = np.clip((84 - np.hypot(xx - GLOBE[0], (yy - GLOBE[1]) * 1.25)) / 10, 0, 1) * np.clip((GLOBE[1] + 36 - yy) / 12, 0, 1)
shape = np.maximum(np.maximum(sphere, ring), sparkle)
shape *= np.clip((309 - yy) / 7, 0, 1)          # stop above her fingertips
sprite = BG + (idle[gy0:gy0 + GH, gx0:gx0 + GW] - BG) * shape[..., None]
atlas = np.empty((256, 256, 3), np.float32); atlas[:] = BG
atlas[4:4 + GH, 4:4 + GW] = sprite
eh, ew = eyes_patch.shape[:2]
atlas[180:180 + eh, 4:4 + ew] = eyes_patch
meta['globe'] = {'centre': list(GLOBE), 'box': [gx0, gy0, GW, GH], 'atlas': [4, 4], 'erase': {'sphere': 70, 'ring': [cx, cy, 96, 30, -12.5]}}
meta['eyes'] = {'box': [EYES[0], EYES[1], ew, eh], 'atlas': [4, 180]}
Image.fromarray(atlas.round().astype(np.uint8)).save(OUT + 'parts.webp', lossless=True, method=6)

total = 0
for name, image in images.items():
    path = f'{OUT}{name}.webp'
    Image.fromarray(image.round().astype(np.uint8)).save(path, quality=86, method=6)
    total += os.path.getsize(path)
total += os.path.getsize(OUT + 'parts.webp')
json.dump(meta, open(OUT + 'meta.json', 'w'), indent=1)
print('files', sorted((f, os.path.getsize(OUT + f)) for f in os.listdir(OUT)), 'total KB', total // 1024)
