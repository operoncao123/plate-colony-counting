#!/usr/bin/env python3
"""Count colonies on agar-plate photos (phone camera).

Requires: numpy, opencv-python (cv2).  No scikit-image needed.

Pipeline
--------
1. Locate the dish: coarse Hough circle -> per-angle radial edge search -> robust
   ellipse fit.  The ellipse absorbs camera tilt (the dish is an ellipse, not a circle).
2. Build the agar mask = that ellipse shrunk to `--shrink`.  `--shrink` is the only
   geometry knob, and its meaning is stable because the ellipse tracks the real dish
   rim.  (An earlier version shrank a sloppy Hough circle, so the boundary wandered
   between plates and cut off outer colonies.)
3. White top-hat:  tophat = gray - background, where background = large-kernel median
   of a down-scaled copy.  Removes uneven illumination / glare without capping bright
   dense patches.
4. Threshold the top-hat (Otsu inside the mask, with an absolute floor).
5. Reject non-colony components:
     a) thin streaks (rim-highlight arcs, meniscus): a component whose maximum
        distance-transform radius is far below a colony radius is not a colony.  This
        is the failure that mattered most: those arcs passed the old size filter and
        the watershed then split them into dozens of phantom colonies;
     b) non-blob shapes (low solidity) -- curved arcs that survive (a).
6. Split genuinely merged blobs (touching colonies) by distance-transform watershed.
7. Emit annotated.jpg, boundary_check.jpg, zoom_*.jpg and count_report.json.

Always eyeball boundary_check.jpg and the zoom_*.jpg crops before trusting a count.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------- dish geometry
def _hough_circle(im):
    sc = 0.35
    s = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(s, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    cs = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT, dp=2.0, minDist=g.shape[0] // 2,
        param1=100, param2=50,
        minRadius=int(g.shape[0] * 0.30), maxRadius=int(g.shape[0] * 0.50))
    if cs is None:
        return None
    c = cs[0][0]
    return c[0] / sc, c[1] / sc, c[2] / sc


def _refine_ellipse(im, cx, cy, R0, nang=720, lo=0.80, hi=1.14, sigma=4.0):
    """Per-angle radial gradient search -> cv2.fitEllipse. Robust to tilt + glare."""
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rs = np.arange(lo * R0, hi * R0, 1.0).astype(np.float32)
    pts = []
    for i in range(nang):
        th = 2 * np.pi * i / nang
        xs = (cx + rs * np.cos(th)).reshape(1, -1).astype(np.float32)
        ys = (cy + rs * np.sin(th)).reshape(1, -1).astype(np.float32)
        v = cv2.remap(gray, xs, ys, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE).ravel()
        v = cv2.GaussianBlur(v.reshape(1, -1), (0, 0), sigma).ravel()
        gr = np.gradient(v)
        k = int(np.argmax(np.abs(gr)))
        if abs(gr[k]) < 3.0:
            continue
        pts.append([cx + rs[k] * np.cos(th), cy + rs[k] * np.sin(th)])
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    if len(pts) < 60:
        return dict(cx=float(cx), cy=float(cy), a=float(R0), b=float(R0), angle=0.0)
    (ex, ey), (d1, d2), ang = cv2.fitEllipse(pts)
    return dict(cx=float(ex), cy=float(ey),
                a=float(max(d1, d2) / 2.0), b=float(min(d1, d2) / 2.0),
                angle=float(ang))


def _speckle_map(gray, sigma=4.0, box=25):
    """Band-pass energy at the colony scale (~10 px), box-averaged.

    Agar carries colony speckle; the smooth plastic rim, the meniscus ring and a
    plain bench do not.  Deliberately NOT a local-std map: std also fires on any
    high-contrast edge, including the bright rim highlight, which is what makes the
    gradient method jump between several curves.
    """
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    return cv2.blur(np.abs(hp), (box, box))


def _refine_ellipse_robust(im, cx, cy, R0, speck=11.0, bright=90.0,
                           gap=40.0, nang=720, verbose=True):
    """Agar ellipse from the bright-AND-speckled region, robustly fitted.

    Walks outward along each ray and takes the first *sustained* drop of
    (speckled AND bright) - i.e. the agar/rim junction, whatever lies beyond it
    (dark bench, lit bench, rim highlight).  Several rays inevitably land badly
    (a clumped colony reads smooth, a partially colonised edge, the dish clipped
    by the frame), so the fit is RANSAC + iteratively reweighted rather than a
    plain least-squares cv2.fitEllipse.
    Returns (ellipse, inlier_fraction) or (None, 0.0) if it cannot be trusted.
    """
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]
    ok = ((_speckle_map(gray) > speck) & (gray > bright)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ok, 8)
    if n > 1:
        i = lab[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))]
        if i == 0:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.nonzero(lab == i)
        if len(xs) > 1000:
            cx, cy = float(xs.mean()), float(ys.mean())

    def edges(ccx, ccy, r_lo, r_hi):
        rs = np.arange(r_lo, r_hi, 1.0).astype(np.float32)
        pts = []
        for k in range(nang):
            t = 2 * np.pi * k / nang
            x = np.clip(ccx + rs * np.cos(t), 0, W - 1)
            y = np.clip(ccy + rs * np.sin(t), 0, H - 1)
            v = cv2.remap(ok, x.reshape(1, -1).astype(np.float32),
                          y.reshape(1, -1).astype(np.float32),
                          cv2.INTER_NEAREST).ravel()
            nz = np.nonzero(v)[0]
            if len(nz) == 0 or rs[nz[0]] > 0.85 * r_hi:
                continue
            above = v.copy()
            above[:nz[0]] = 1
            last, run = 0, 0
            for j in range(len(above)):
                if above[j]:
                    last, run = j, 0
                else:
                    run += 1
                    if run > gap:
                        break
            pts.append([ccx + rs[last] * np.cos(t), ccy + rs[last] * np.sin(t)])
        return np.asarray(pts, np.float64)

    def resid(pts, e):
        t = np.deg2rad(e["angle"])
        x0, y0 = pts[:, 0] - e["cx"], pts[:, 1] - e["cy"]
        xr = x0 * np.cos(t) + y0 * np.sin(t)
        yr = -x0 * np.sin(t) + y0 * np.cos(t)
        rr = np.hypot(xr / e["a"], yr / e["b"])
        g = np.hypot(xr / e["a"] ** 2, yr / e["b"] ** 2)
        return (rr - 1.0) / np.maximum(g, 1e-9)

    def fit(pts):
        (ex, ey), (d1, d2), ang = cv2.fitEllipse(
            pts.astype(np.float32).reshape(-1, 1, 2))
        return dict(cx=float(ex), cy=float(ey), a=float(max(d1, d2) / 2),
                    b=float(min(d1, d2) / 2), angle=float(ang))

    R = R0
    inl = None
    e = None
    for _ in range(4):
        pts = edges(cx, cy, 0.35 * R, 1.25 * R)
        if len(pts) < 200:
            return None, 0.0
        rng = np.random.default_rng(0)
        best = None
        for _ in range(3000):
            idx = rng.choice(len(pts), 5, replace=False)
            try:
                cand = fit(pts[idx])
            except cv2.error:
                continue
            if not (700 < 2 * cand["a"] < 4000 and 700 < 2 * cand["b"] < 4000):
                continue
            k = int((np.abs(resid(pts, cand)) < 15).sum())
            if best is None or k > best[0]:
                best = (k, cand)
        if best is None:
            return None, 0.0
        e = best[1]
        inl = np.abs(resid(pts, e)) < 15
        for _ in range(8):
            e = fit(pts[inl])
            r = np.abs(resid(pts, e))
            s = 1.4826 * np.median(r[inl])
            inl = r <= max(4.0, 2.2 * s)
        cx, cy, R = e["cx"], e["cy"], (e["a"] + e["b"]) / 2
    frac = float(inl.sum()) / float(len(pts))
    if verbose:
        print("  robust boundary: a=%.0f b=%.0f angle=%.1f centre=(%.0f,%.0f) "
              "inliers=%.0f%%" % (e["a"], e["b"], e["angle"], e["cx"], e["cy"],
                                  100 * frac))
    return e, frac


def ellipse_mask(shape, ell, scale):
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, ((ell["cx"], ell["cy"]),
                    (ell["a"] * 2 * scale, ell["b"] * 2 * scale), ell["angle"]),
                255, -1)
    return m


def ellipse_poly(ell, scale, n=720):
    t = np.deg2rad(ell["angle"])
    ax, bx = ell["a"] * scale, ell["b"] * scale
    th = np.linspace(0, 2 * np.pi, n)
    x = ax * np.cos(th)
    y = bx * np.sin(th)
    return np.stack([ell["cx"] + x * np.cos(t) - y * np.sin(t),
                     ell["cy"] + x * np.sin(t) + y * np.cos(t)], 1).astype(np.int32)


# ------------------------------------------------------------------ background
def flat_background(gray, ds=4):
    small = cv2.resize(gray, None, fx=1.0 / ds, fy=1.0 / ds,
                       interpolation=cv2.INTER_AREA)
    k = 31 if min(small.shape) > 60 else 15
    med = cv2.medianBlur(small, k)
    return cv2.resize(med, (gray.shape[1], gray.shape[0]),
                      interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------- counter
def count_plate(path, shrink=0.95, min_diam_mm=0.25, dish_mm=90.0,
                peak_frac=0.45, min_dist_frac=1.15, valid_frac=0.55,
                frag_frac=0.35, solidity_min=0.45, verbose=True,
                robust_boundary=False, boundary_json=None, px_per_mm=None,
                min_inlier_frac=0.55):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(path)
    H, W = im.shape[:2]
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    hc = _hough_circle(im)
    if hc is None:
        raise RuntimeError("could not locate the dish in %s" % path)
    if boundary_json:
        with open(boundary_json) as fh:
            d = json.load(fh)
        ell = dict(cx=float(d["cx"]), cy=float(d["cy"]), a=float(d["a"]),
                   b=float(d["b"]), angle=float(d["angle"]))
        if verbose:
            print("  boundary: supplied from %s (a=%.0f b=%.0f angle=%.1f)"
                  % (boundary_json, ell["a"], ell["b"], ell["angle"]))
    elif robust_boundary:
        ell2, frac = _refine_ellipse_robust(im, *hc, verbose=verbose)
        if ell2 is not None and frac >= min_inlier_frac:
            ell = ell2
        else:
            ell = _refine_ellipse(im, *hc)
            if verbose:
                print("  robust boundary untrustworthy (inliers %.0f%% < %.0f%%) "
                      "-> fell back to the gradient fit"
                      % (100 * frac, 100 * min_inlier_frac))
    else:
        ell = _refine_ellipse(im, *hc)
    mask = ellipse_mask(im.shape, ell, shrink)
    if px_per_mm is None:
        px_per_mm = 2.0 * ((ell["a"] + ell["b"]) / 2.0) * shrink / dish_mm

    # 3-4. top-hat + threshold
    th = cv2.GaussianBlur(cv2.subtract(gray, flat_background(gray)), (3, 3), 0)
    otsu = float(cv2.threshold(th[mask > 0].reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    thr = max(otsu, 8.0)
    binary = ((th >= thr) & (mask > 0)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    min_area = max(np.pi * (min_diam_mm * px_per_mm / 2.0) ** 2, 25.0)
    cand = [i for i in range(1, n) if stats[i, 4] >= min_area]
    areas = np.array([stats[i, 4] for i in cand]) if cand else np.array([1.0])
    small = areas[areas <= np.percentile(areas, 45)]
    a1 = float(np.median(small))
    r_single = float(np.sqrt(max(a1, 1.0) / np.pi))

    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 5. artifact rejection
    kept, dropped_edges, dropped_shape = [], [], []
    for i in cand:
        x, y, w, h, a = stats[i]
        if a <= 1.6 * a1:
            kept.append((i, a))
            continue
        m = (lab[y:y + h, x:x + w] == i)
        dmax = float(dt[y:y + h, x:x + w][m].max())
        if dmax < valid_frac * r_single:            # thin streak / rim arc
            dropped_edges.append(i)
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        sol = a / max(cv2.contourArea(hull), 1.0)   # curved arc -> low solidity
        if sol < solidity_min:
            dropped_shape.append(i)
            continue
        kept.append((i, a))

    # 6. split merged blobs
    min_dist = min_dist_frac * r_single
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * min_dist) + 1,) * 2)
    pts, n_split, n_split_cols, n_frag = [], 0, 0, 0
    for i, a in kept:
        x, y, w, h, _ = stats[i]
        if a <= 1.6 * a1:
            pts.append((cent[i][0], cent[i][1], 1))
            continue
        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        sub = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        d = dt[y0:y1, x0:x1] * sub
        mx = cv2.dilate(d, ker)
        pk = ((d >= mx - 1e-4) & (d >= peak_frac * r_single)).astype(np.uint8)
        nk, pkl = cv2.connectedComponents(pk, 8)
        nk -= 1
        if nk <= 1:
            pts.append((cent[i][0], cent[i][1], 1))
            continue
        mk = pkl.copy()
        mk[sub == 0] = 0
        img3 = cv2.cvtColor((255 - np.clip(d / (r_single * 1.5) * 255, 0, 255)
                             ).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mk = cv2.watershed(img3, mk.astype(np.int32))
        n_split += 1
        got = 0
        for j in range(1, nk + 1):
            ys, xs = np.where((mk == j) & (sub > 0))
            if len(xs) < frag_frac * a1:
                n_frag += 1
                continue
            pts.append((x0 + xs.mean(), y0 + ys.mean(), 1))
            got += 1
        if got == 0:
            pts.append((cent[i][0], cent[i][1], 1))
        n_split_cols += got

    # --- independent cross-check: local maxima of the smoothed top-hat.  This path
    # never uses the watershed, so it fails differently: it *undercounts* dense
    # clumps (peaks merge), which makes it a useful lower bound.
    sm = cv2.GaussianBlur(th.astype(np.float32), (0, 0), 0.7 * r_single)
    kk = int(2 * r_single) + 1
    mx2 = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    pk2 = ((sm >= mx2 - 1e-6) & (sm >= thr) & (mask > 0)).astype(np.uint8)
    crosscheck = int(cv2.connectedComponents(pk2, 8)[0]) - 1

    total = len(pts)
    diag = dict(
        colonies=total, blobs=len(kept), dropped_thin=len(dropped_edges),
        crosscheck_peaks=crosscheck,
        crosscheck_ratio=round(crosscheck / max(total, 1), 3),
        dropped_shape=len(dropped_shape), split_blobs=n_split,
        split_into=n_split_cols, dropped_fragments=n_frag,
        r_single_px=round(r_single, 2), single_area_px=round(a1, 1),
        px_per_mm=round(px_per_mm, 2), min_area_px=round(min_area, 1),
        otsu=otsu, threshold=thr, shrink=shrink, dish_mm=dish_mm,
        min_colony_mm=min_diam_mm,
        ellipse={k: round(v, 2) for k, v in ell.items()},
    )
    return dict(im=im, gray=gray, th=th, binary=binary, mask=mask, ell=ell,
                pts=pts, diag=diag, r_single=r_single, a1=a1,
                px_per_mm=px_per_mm, kept=kept, lab=lab, stats=stats, dt=dt,
                rejected_ids=dropped_edges + dropped_shape)


# -------------------------------------------------------------------- render
LEGEND = [
    ((0, 0, 255), "red outline    = colony COUNTED (outline of its blob)"),
    ((0, 255, 0), "green ring+orange cross = one counted colony"),
    ((255, 120, 0), "blue outline   = artefact detected, NOT counted"),
    ((0, 255, 255), "yellow line    = agar / counting boundary"),
]


def stamp_legend(img, n, shrink, boundary=False):
    """Burn the colour key into the image so it is readable on its own."""
    rows = list(LEGEND)
    if boundary:
        rows = [((255, 255, 0), "cyan line      = dish outer ellipse"),
                ((0, 255, 255), "yellow line    = agar / counting boundary")] + rows[3:]
    else:
        rows = rows + [((255, 255, 255), "shrink = %.2f" % shrink)]
    rows.append(((255, 255, 255), "colonies = %d  (+/-10-15%%)" % n))
    H, W = img.shape[:2]
    fs = max(0.5, W / 1700.0)
    th = max(1, int(round(fs * 2)))
    pad = int(12 * fs)
    lh = int(30 * fs)
    box_w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
                    for _, t in rows) + 3.2 * pad)
    box_h = lh * len(rows) + 2 * pad
    x0, y0 = pad, H - box_h - pad
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(ov, 0.62, img, 0.38, 0)
    for k, (col, txt) in enumerate(rows):
        yy = y0 + pad + lh * k + int(lh * 0.72)
        cv2.putText(img, txt, (x0 + pad, yy), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    col, th, cv2.LINE_AA)
    return img


def render(res, stem, outdir, n_zoom=5):
    os.makedirs(outdir, exist_ok=True)
    im, ell, pts = res["im"], res["ell"], res["pts"]
    H, W = im.shape[:2]

    ann = im.copy()
    # red = counted colony (outline of the blob it came from) -- counted blobs only,
    # so a red outline always means "this one is in the number".
    kmask = np.isin(res["lab"], [i for i, _ in res["kept"]]).astype(np.uint8)
    kc, _ = cv2.findContours(kmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ann, kc, -1, (0, 0, 255), 1)
    # blue = artefact detected but deliberately NOT counted
    rej = set(res.get("rejected_ids", []))
    if rej:
        rmask = np.isin(res["lab"], list(rej)).astype(np.uint8)
        rc, _ = cv2.findContours(rmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ann, rc, -1, (255, 120, 0), 2)
    for x, y, _ in pts:
        cv2.circle(ann, (int(round(x)), int(round(y))), 3, (0, 255, 0), 1)
        cv2.drawMarker(ann, (int(round(x)), int(round(y))), (0, 200, 255),
                       cv2.MARKER_CROSS, 5, 1)
    cv2.polylines(ann, [ellipse_poly(ell, res["diag"]["shrink"])], True,
                  (0, 255, 255), 3)
    ann = stamp_legend(ann, res["diag"]["colonies"], res["diag"]["shrink"])
    sc = 1400.0 / max(H, W)
    cv2.imwrite(os.path.join(outdir, "annotated.jpg"),
                cv2.resize(ann, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))

    bnd = im.copy()
    cv2.polylines(bnd, [ellipse_poly(ell, 1.0)], True, (255, 255, 0), 3)   # cyan
    cv2.polylines(bnd, [ellipse_poly(ell, res["diag"]["shrink"])], True,
                  (0, 255, 255), 3)                                       # yellow
    bnd = stamp_legend(bnd, res["diag"]["colonies"], res["diag"]["shrink"],
                       boundary=True)
    cv2.imwrite(os.path.join(outdir, "boundary_check.jpg"),
                cv2.resize(bnd, None, fx=900.0 / max(H, W), fy=900.0 / max(H, W),
                           interpolation=cv2.INTER_AREA))

    # 1:1 zoom crops for eyeball verification
    cx, cy = ell["cx"], ell["cy"]
    rr = 0.45 * (ell["a"] + ell["b"]) / 2.0
    spots = [(cx, cy), (cx + rr, cy - rr), (cx - rr, cy - rr),
             (cx - rr, cy + rr), (cx + rr, cy + rr)]
    tile = int(min(700, min(H, W) * 0.95))
    for k, (px, py) in enumerate(spots[:n_zoom], 1):
        x0 = int(np.clip(px - tile / 2, 0, W - tile))
        y0 = int(np.clip(py - tile / 2, 0, H - tile))
        cv2.imwrite(os.path.join(outdir, "zoom_%d.jpg" % k),
                    ann[y0:y0 + tile, x0:x0 + tile])

    rep = dict(image=os.path.basename(res.get("path", stem)), **res["diag"])
    rep["note"] = ("Counted colonies = orange cross + green circle; red outline = "
                   "detected blob; BLUE outline = artefact detected but deliberately "
                   "NOT counted (rim/meniscus streaks, low-solidity arcs); yellow = "
                   "agar boundary. crosscheck_peaks is an independent local-maxima "
                   "count: it should be <= colonies and within ~20%; a much lower "
                   "value means heavy clumping, a much higher one means over-splitting. "
                   "VERIFY boundary_check.jpg and zoom_*.jpg before trusting the "
                   "count. Report as '~N (+/-10-15%)'.")
    with open(os.path.join(outdir, "count_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Count colonies on plate photos")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--outdir", default=None,
                    help="output root (default: <image>_count next to the image)")
    ap.add_argument("--shrink", type=float, default=0.95,
                    help="agar boundary as fraction of the fitted dish ellipse")
    ap.add_argument("--dish-mm", type=float, default=90.0,
                    help="outer diameter of the plate in mm (for mm calibration)")
    ap.add_argument("--min-diam-mm", type=float, default=0.25,
                    help="minimum colony diameter counted (mm)")
    ap.add_argument("--solidity-min", type=float, default=0.45)
    ap.add_argument("--robust-boundary", action="store_true",
                    help="find the agar/rim junction from the bright-AND-speckled "
                         "region instead of the single strongest radial gradient "
                         "(use on tilted hand-held photos where the rim highlight, "
                         "meniscus and bench reflections confuse the gradient). "
                         "NOTE: this fits the AGAR edge directly, so pair it with "
                         "--shrink 1.0 (shrink 0.95 would inset it a further 5%%)")
    ap.add_argument("--boundary-json", default=None,
                    help="JSON with cx,cy,a,b,angle to use as the dish ellipse "
                         "instead of fitting one")
    ap.add_argument("--px-per-mm", type=float, default=None,
                    help="override the mm scale (otherwise derived from the ellipse)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    for p in a.images:
        res = count_plate(p, shrink=a.shrink, min_diam_mm=a.min_diam_mm,
                          dish_mm=a.dish_mm, solidity_min=a.solidity_min,
                          verbose=not a.quiet,
                          robust_boundary=a.robust_boundary,
                          boundary_json=a.boundary_json,
                          px_per_mm=a.px_per_mm)
        res["path"] = p
        stem = os.path.splitext(os.path.basename(p))[0]
        outdir = os.path.join(a.outdir, stem) if a.outdir else \
            os.path.join(os.path.dirname(os.path.abspath(p)), "%s_count" % stem)
        rep = render(res, stem, outdir)
        if not a.quiet:
            print("%-28s colonies=%d  (blobs=%d, split %d->%d, dropped thin=%d "
                  "shape=%d, r_single=%.1fpx, %.1f px/mm)"
                  % (stem, rep["colonies"], rep["blobs"], rep["split_blobs"],
                     rep["split_into"], rep["dropped_thin"],
                     rep["dropped_shape"], rep["r_single_px"], rep["px_per_mm"]))
            print("   -> %s" % outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
