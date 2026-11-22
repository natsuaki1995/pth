# aki模式 定稿脚本：Real-CUGAN SE 4x（8 向 TTA）→ Lab 校色 → σ0.65 平滑 + 眼睛高光保护 → PNG / JPG q100 4:4:4 / ≤499,000 字节高质量 JPG（必要时缩小尺寸）
# 用法：python3 aki_mode.py 原图 x y w h n0|n-1 输出前缀 [眼框1 眼框2]
#   x y w h 为原图上的正方形裁框；眼框为放大 4 倍、裁回后的输出图坐标 x0,y0,x1,y1（按原裁框尺寸）
#   裁框边长超过 AKI_MAX_CROP（默认 1000）时，先把原图缩小到裁框边长 = 1000 再处理，成品约 4000×4000
# 权重默认与本脚本同目录（up4x-latest-no-denoise.pth / up4x-latest-conservative.pth）
import sys, os, io, glob, json, numpy as np, cv2, torch
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cugan import load

src, x, y, w, h, mdl, out = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6], sys.argv[7]
eyes = [tuple(map(int, s.split(","))) for s in sys.argv[8:]]
_nm = {"n0": "no-denoise", "n-1": "conservative"}[mdl]
MP = (glob.glob(os.path.join(HERE, "up4x-latest-" + _nm + ".pth")) + glob.glob("/home/claude/natsuaki1995/pth/up4x-latest-" + _nm + ".pth") + glob.glob("/mnt/user-data/uploads/*up4x-latest-" + _nm + ".pth") + glob.glob("/mnt/data/*up4x-latest-" + _nm + ".pth"))[0]
O = Image.open(src); icc = O.info.get("icc_profile"); O = O.convert("RGB"); W0, H0 = O.size
# 裁框过大时先整体缩小原图，使裁框边长 = AKI_MAX_CROP（默认 1000），成品约 4000 像素；坐标与眼框同比例换算
LIMIT = int(os.environ.get("AKI_MAX_CROP", "1000")); scale = 1.0
if max(w, h) > LIMIT:
    scale = LIMIT / max(w, h); sq = (w == h)
    O = Image.fromarray(cv2.resize(np.asarray(O), (round(W0 * scale), round(H0 * scale)), interpolation=cv2.INTER_AREA))
    W0, H0 = O.size
    x, y, w, h = round(x * scale), round(y * scale), round(w * scale), round(h * scale)
    if sq: h = w
    x, y = min(max(x, 0), W0 - w), min(max(y, 0), H0 - h)
    eyes = [tuple(round(v * scale) for v in e) for e in eyes]
pl, pt = min(16, x), min(16, y); pr, pb = min(16, W0 - x - w), min(16, H0 - y - h)
inp = np.asarray(O.crop((x - pl, y - pt, x + w + pr, y + h + pb))).astype(np.float32) / 255
torch.set_num_threads(2); m, _ = load(MP)
def run(x, T=256, P=24):
    Hh, Ww = x.shape[:2]; R = np.zeros((Hh * 4, Ww * 4, 3), np.float32)
    for y0 in range(0, Hh, T):
        for x0 in range(0, Ww, T):
            y1, x1 = min(y0 + T, Hh), min(x0 + T, Ww); a, b = max(y0 - P, 0), max(x0 - P, 0); c, d = min(y1 + P, Hh), min(x1 + P, Ww)
            with torch.no_grad():
                r = m(torch.from_numpy(np.ascontiguousarray(x[a:c, b:d].transpose(2, 0, 1))).unsqueeze(0)).clamp_(0, 1)[0].numpy().transpose(1, 2, 0)
            R[y0*4:y1*4, x0*4:x1*4] = r[(y0-a)*4:(y1-a)*4, (x0-b)*4:(x1-b)*4]
    return R
Hh, Ww = inp.shape[:2]; M = np.zeros((Hh * 4, Ww * 4, 3), np.float64)
for flip in (False, True):
    for k in range(4):
        x_ = np.rot90(inp[:, ::-1] if flip else inp, k)
        y_ = np.rot90(run(np.ascontiguousarray(x_)), -k)
        M += y_[:, ::-1] if flip else y_
M = (M / 8).astype(np.float32)
B = np.clip(cv2.resize(inp, (Ww * 4, Hh * 4), interpolation=cv2.INTER_CUBIC), 0, 1)
Lb, Lm = cv2.cvtColor(B, cv2.COLOR_RGB2Lab), cv2.cvtColor(M, cv2.COLOR_RGB2Lab)
g = lambda a, s: cv2.GaussianBlur(a, (0, 0), s, borderType=cv2.BORDER_REFLECT_101)
C = Lm.copy()
C[..., 0] += g(Lb[..., 0] - Lm[..., 0], 3.5)
C[..., 1] += g(Lb[..., 1] - Lm[..., 1], 6.0)
C[..., 2] += g(Lb[..., 2] - Lm[..., 2], 6.0)
C = np.clip(cv2.cvtColor(C, cv2.COLOR_Lab2RGB), 0, 1)
Hm = np.zeros(C.shape[:2], np.float32); log = []
Y = (0.2126 * C[..., 0] + 0.7152 * C[..., 1] + 0.0722 * C[..., 2]) * 255
for (ex0, ey0, ex1, ey1) in eyes:
    ex0 += pl*4; ex1 += pl*4; ey0 += pt*4; ey1 += pt*4
    E = ex1 - ex0; reg = Y[ey0:ey1, ex0:ex1]
    n, lab, st, _ = cv2.connectedComponentsWithStats((reg >= 235).astype(np.uint8), connectivity=8)
    rad = max(2, round(0.025 * E)); hits = 0; keep = np.zeros(reg.shape, np.uint8)
    for i in range(1, n):
        bx, by, bw, bh, area = st[i]
        if not (max(2, 1e-4 * E * E) <= area <= 0.04 * E * E): continue
        if bx == 0 or by == 0 or bx + bw >= reg.shape[1] or by + bh >= reg.shape[0]: continue
        comp = (lab == i).astype(np.uint8)
        ring = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*rad+1, 2*rad+1))) - comp
        rv = reg[ring > 0]
        if reg[comp > 0].mean() - np.median(rv) >= 25 and (rv <= 170).mean() >= 0.25: keep |= comp; hits += 1
    if hits:
        core = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 5)
        Hm[ey0:ey1, ex0:ex1] = np.maximum(Hm[ey0:ey1, ex0:ex1], np.clip(1 - dist / 2.0, 0, 1))
    log.append(hits)
S = g(C, 0.65); R = Hm[..., None] * C + (1 - Hm[..., None]) * S
R = R[pt*4:pt*4 + h*4, pl*4:pl*4 + w*4]
im = Image.fromarray(np.rint(R * 255).astype(np.uint8)); kw = {"icc_profile": icc} if icc else {}
im.save(out + ".png", **kw); im.save(out + ".jpg", quality=100, subsampling=0, optimize=True, **kw)
# ≤499,000 字节高质量版：4:4:4、质量不低于 AKI_Q499（默认 90）；同尺寸压不进就把长边按 250 像素一档用 INTER_AREA 缩小再试
Q499 = int(os.environ.get("AKI_Q499", "90")); done = None; L0 = max(im.size)
def enc(r, q, sub):
    bb = io.BytesIO(); r.save(bb, "JPEG", quality=q, subsampling=sub, optimize=True, progressive=True, **kw); return bb.getvalue()
def shrink(L):
    if L == L0: return im
    return Image.fromarray(cv2.resize(np.asarray(im), (round(im.size[0] * L / L0), round(im.size[1] * L / L0)), interpolation=cv2.INTER_AREA))
for L in [L0] + list(range((L0 - 1) // 250 * 250, 249, -250)):
    r = shrink(L)
    for q in range(100, Q499 - 1, -1):
        b = enc(r, q, 0)
        if len(b) <= 499000: done = (0, q, b, r.size); break
    if done: break
if not done:  # 极端情况兜底：最小档继续降质量
    for q in range(Q499 - 1, 0, -1):
        b = enc(r, q, 0)
        if len(b) <= 499000: done = (0, q, b, r.size); break
open(out + "_499kb.jpg", "wb").write(done[2])
ref = np.asarray(O.crop((x, y, x + w, y + h))).astype(float)
d = np.asarray(im.resize((w, h), Image.BOX)).astype(float)
sat = lambda a: cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2HSV)[..., 1].astype(float).mean()
Bc = np.rint(B[pt*4:pt*4+h*4, pl*4:pl*4+w*4] * 255); Rc = np.asarray(im).astype(float)
Yf = lambda a: 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
print(json.dumps({"model": mdl, "scale": round(scale, 4), "box": [x, y, w, h], "pad": [pl, pt, pr, pb], "eye_hits": log,
    "rgb_bias": np.round((d - ref).mean((0, 1)), 2).tolist(), "mae": round(float(np.abs(d - ref).mean()), 2),
    "sat_delta": round(sat(d) - sat(ref), 2),
    "dark_pp_vs_bicubic": round(float(((Yf(Rc) < 64).mean() - (Yf(Bc) < 64).mean()) * 100), 2),
    "jpg499": {"size": list(done[3]), "subsampling": "4:4:4", "quality": done[1], "bytes": len(done[2])}}, ensure_ascii=False))
