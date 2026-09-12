#!/usr/bin/env python3
"""Count colonies on an agar plate photo (phone camera, tilted, uneven light).

Pipeline (calibrated on 10-cm dish photos, ~3000x4000 px):
  1. Locate dish: multi-threshold median blob ellipse (photos are tilted -> ellipse, never a circle)
  2. Agar analysis boundary = ellipse x shrink (default 0.96; VERIFY boundary_check.jpg, override via --ellipse)
  3. Exclude glare: large connected very-bright regions (specular reflection on agar)
  4. Background: 4x-downsampled median filter (grey opening UNDERCOUNTS in dense regions - do not use)
  5. Detect blobs by local contrast, filter to human-visible colonies (size + roundness)
  6. Classify: single (1) / dumbbell merged pair (2) / chain (area/median)
  7. Exclude meniscus texture: elongated blobs near rim whose long axis parallels the boundary
  8. Extrapolate glare-excluded area from neighbouring ring density
  9. Cross-check with size-matched local-maxima counting; output annotated image + JSON report

Dependencies: numpy, scipy, pillow  (no OpenCV needed)
"""
import argparse, json, math, os, sys, time
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', help='plate photo path')
    p.add_argument('-o', '--outdir', default=None, help='output dir (default: <image>_count/)')
    p.add_argument('--dish-mm', type=float, default=90.0, help='dish outer diameter in mm (default 90)')
    p.add_argument('--shrink', type=float, default=0.96,
                   help='agar boundary as fraction of the located ellipse (VERIFY boundary_check.jpg)')
    p.add_argument('--ellipse', default=None,
                   help='manual override "cx,cy,A,B,theta_deg" - use when boundary_check.jpg '
                        'shows the automatic boundary off the agar edge')
    p.add_argument('--min-colony-mm', type=float, default=0.22,
                   help='minimum countable colony diameter in mm (default 0.22; raise for '
                        '"only clearly visible dots" standards)')
    p.add_argument('--diff-thr', type=float, default=9.0, help='local-contrast threshold for blob segmentation')
    p.add_argument('--no-extrapolate', action='store_true', help='do not extrapolate glare-excluded area')
    p.add_argument('--no-texture-filter', action='store_true',
                   help='disable meniscus-texture exclusion (for perfectly top-down photos)')
    return p.parse_args()

def fit_ellipse(x, y):
    D = np.column_stack([x*x, x*y, y*y, x, y, np.ones_like(x)])
    C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
    eigval, eigvec = np.linalg.eig(np.linalg.solve(D.T @ D, C))
    cond = 4*eigvec[0]*eigvec[2] - eigvec[1]**2
    k = np.argmax(np.where(cond > 0, eigval.real, -np.inf))
    a, b, c, d, e, f = eigvec[:, k].real
    M = np.array([[a, b/2], [b/2, c]])
    cen = np.linalg.solve(2*M, [-d, -e])
    val = a*cen[0]**2 + b*cen[0]*cen[1] + c*cen[1]**2 + d*cen[0] + e*cen[1] + f
    ev, evec = np.linalg.eigh(M)
    axes = np.sqrt(-val/ev); o = np.argsort(axes)[::-1]
    return cen[0], cen[1], axes[o][0], axes[o][1], math.atan2(evec[1, o[0]], evec[0, o[0]])

def ellipse_resid(x, y, g):
    cx, cy, A, B, th = g
    xr = (x-cx)*math.cos(th) + (y-cy)*math.sin(th)
    yr = -(x-cx)*math.sin(th) + (y-cy)*math.cos(th)
    return np.abs(np.sqrt((xr/A)**2 + (yr/B)**2) - 1) * min(A, B)

def locate_dish(gray):
    """Multi-threshold median blob ellipse. A single threshold is fragile (glare, bright
    carpet); the median across levels is stable. The blob boundary sits between the agar
    edge and the outer rim -> pair with shrink ~0.96 and VERIFY boundary_check.jpg."""
    H, W = gray.shape
    sm = ndi.gaussian_filter(gray, 8)

    def blob_ellipse(thr):
        mask = sm > thr
        lab, n = ndi.label(ndi.binary_fill_holes(mask))
        if n == 0: return None
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        dish = ndi.binary_opening(ndi.binary_fill_holes(lab == big), np.ones((31, 31)))
        dish = ndi.binary_fill_holes(dish)
        if dish.sum() < 0.02*H*W: return None
        bnd = dish & ~ndi.binary_erosion(dish)
        ys, xs = np.nonzero(bnd)
        if len(xs) < 500: return None
        sel = np.arange(0, len(xs), max(1, len(xs)//4000))
        x, y = xs[sel].astype(float), ys[sel].astype(float)
        try:
            g = fit_ellipse(x, y)
            for _ in range(3):
                r = ellipse_resid(x, y, g)
                k = r < max(10, np.percentile(r, 75))
                if k.sum() < 50: break
                g = fit_ellipse(x[k], y[k])
        except np.linalg.LinAlgError:
            return None
        return g

    fits = []
    for thr in np.percentile(sm, (70, 75, 80, 85)):
        g = blob_ellipse(float(thr))
        if g is not None and max(g[2], g[3])/max(min(g[2], g[3]), 1e-9) <= 1.35:
            fits.append(g)
    if not fits: sys.exit('ERROR: cannot locate dish blob at any threshold')
    fits.sort(key=lambda g: g[2])
    print('boundary basis: median of %d threshold fits (A=%d..%d px)' %
          (len(fits), min(f[2] for f in fits), max(f[2] for f in fits)))
    return fits[len(fits)//2]

def rho_map(shape, g):
    H, W = shape
    cx, cy, A, B, th = g
    yy, xx = np.mgrid[0:H, 0:W]
    xr = (xx-cx)*math.cos(th) + (yy-cy)*math.sin(th)
    yr = -(xx-cx)*math.sin(th) + (yy-cy)*math.cos(th)
    return np.sqrt((xr/A)**2 + (yr/B)**2), xr, yr

def main():
    args = parse_args()
    t0 = time.time()
    outdir = args.outdir or (os.path.splitext(args.image)[0] + '_count')
    os.makedirs(outdir, exist_ok=True)

    im = Image.open(args.image).convert('RGB')
    gray = np.asarray(im).astype(np.float32).mean(axis=2)
    H, W = gray.shape

    if args.ellipse:
        vals = [float(v) for v in args.ellipse.split(',')]
        assert len(vals) == 5, '--ellipse expects "cx,cy,A,B,theta_deg"'
        g = (vals[0], vals[1], vals[2], vals[3], math.radians(vals[4]))
        print('using manual ellipse override')
    else:
        g = locate_dish(gray)
    cx, cy, A, B, th = g
    px_per_mm = 2*A/args.dish_mm
    min_area = math.pi*(px_per_mm*args.min_colony_mm/2)**2
    print(f'ellipse: center=({cx:.0f},{cy:.0f}) semi=({A:.0f},{B:.0f})px tilt={math.degrees(th):.1f}deg '
          f'| {px_per_mm:.1f} px/mm, min colony area {min_area:.0f}px2')

    rho, xr, yr = rho_map((H, W), g)
    sm6 = ndi.gaussian_filter(gray, 6)

    # boundary overlay BEFORE counting: verify the agar boundary per sector
    prev = im.copy(); prev.thumbnail((760, 760)); sx = prev.size[0]/W
    d = ImageDraw.Draw(prev)
    for s, col in ((1.0, (255, 0, 0)), (args.shrink, (255, 230, 0))):
        pts = []
        for a in np.linspace(0, 2*math.pi, 200):
            px, py = A*s*math.cos(a), B*s*math.sin(a)
            pts.append(((cx+px*math.cos(th)-py*math.sin(th))*sx, (cy+px*math.sin(th)+py*math.cos(th))*sx))
        d.line(pts+[pts[0]], fill=col, width=3)
    prev.save(f'{outdir}/boundary_check.jpg', quality=85)

    # glare: large connected very-bright regions inside agar (specular reflection)
    inside = rho < args.shrink
    bright = inside & (ndi.gaussian_filter(gray, 6) > 205)
    lb, nb = ndi.label(bright)
    bsz = np.bincount(lb.ravel(), minlength=nb+1)
    glare = np.isin(lb, [i for i in range(1, nb+1) if bsz[i] >= 20000])
    glare = ndi.binary_dilation(glare, np.ones((21, 21))) & inside
    region = inside & ~glare

    # background: median filter on 4x downsample (grey opening biases dense regions low)
    coarse = gray[::4, ::4]
    bg = ndi.gaussian_filter(ndi.median_filter(coarse, size=13), 4)
    bg_full = np.asarray(Image.fromarray(bg.astype(np.float32), 'F').resize((W, H), Image.BILINEAR))
    diff = gray - bg_full

    bw = (diff > args.diff_thr) & region
    bw = ndi.binary_opening(bw, np.ones((2, 2)))
    lab, n = ndi.label(bw)
    sizes = np.bincount(lab.ravel(), minlength=n+1)
    objs = ndi.find_objects(lab)

    MED = float(np.median([sizes[i-1] for i, sl in enumerate(objs, 1)
                           if sl and min_area <= sizes[i-1] <= 8*min_area])) or 1.0
    core_r = max(2.0, 0.45*math.sqrt(MED/math.pi))   # min distance-transform radius of a colony core
    maxwin = int(max(5, 1.5*math.sqrt(MED/math.pi))) # peak suppression window

    def split_blob(m):
        """Watershed-style split of a merged blob: distance-transform peaks = colony centers.
        Returns (n_colonies, list_of_submasks). Falls back to (1, [m]) if no split found."""
        pad = 4
        if m.shape[0] < 3*pad or m.shape[1] < 3*pad:
            return 1, [m]
        mp = np.pad(m, pad)
        dt = ndi.gaussian_filter(ndi.distance_transform_edt(mp).astype(float), 0.8)
        pk = (dt == ndi.maximum_filter(dt, size=maxwin)) & (dt > core_r) & mp
        pl, npk = ndi.label(pk)
        if npk < 2:
            return 1, [m]
        pc = np.array(ndi.center_of_mass(pk, pl, range(1, npk+1)))
        keep = np.ones(npk, bool)
        for i, j in cKDTree(pc).query_pairs(r=maxwin*0.7):
            keep[j if dt[tuple(pc[i].astype(int))] >= dt[tuple(pc[j].astype(int))] else i] = False
        centers = pc[keep]
        if len(centers) < 2:
            return 1, [m]
        ys, xs = np.nonzero(mp)
        _, idx = cKDTree(centers).query(np.column_stack([ys, xs]), k=1)
        subs = []
        for k in range(len(centers)):
            sm_k = np.zeros_like(mp)
            sm_k[ys[idx == k], xs[idx == k]] = True
            subs.append(sm_k[pad:-pad, pad:-pad])
        return len(centers), subs

    n_single = n_pair = n_chain = n_tex = 0
    total = 0
    mark_list = []    # (contour_points, color)
    centroids = []    # (cy, cx, count) for density/extrapolation stats
    for i, sl in enumerate(objs, 1):
        if sl is None: continue
        sz = sizes[i-1]
        if sz < min_area or sz > 60*min_area: continue
        m = lab[sl] == i
        sub = np.argwhere(m)
        r_mean = rho[sl][m].mean()
        dx = sub[:, 1]-sub[:, 1].mean(); dy = sub[:, 0]-sub[:, 0].mean()
        cxx, cyy, cxy = (dx**2).mean(), (dy**2).mean(), (dx*dy).mean()
        tr = cxx+cyy; det = cxx*cyy-cxy**2
        ev1 = (tr+math.sqrt(max(tr**2-4*det, 0)))/2
        ev2 = (tr-math.sqrt(max(tr**2-4*det, 0)))/2
        elong = math.sqrt(ev1/max(ev2, 1e-6))
        v = np.array([cxy, ev1-cxx])
        if np.linalg.norm(v) < 1e-9: v = np.array([ev1-cyy, cxy])
        theta_major = math.atan2(v[1], v[0])
        cy0 = sub[:, 0].mean()+sl[0].start; cx0 = sub[:, 1].mean()+sl[1].start
        t_par = math.atan2(yr[int(cy0), int(cx0)]/B, xr[int(cy0), int(cx0)]/A)
        txi = -A*math.sin(t_par)*math.cos(th)-B*math.cos(t_par)*math.sin(th)
        tyi = -A*math.sin(t_par)*math.sin(th)+B*math.cos(t_par)*math.cos(th)
        theta_tan = math.atan2(tyi, txi)
        dphi = math.degrees(abs((theta_major-theta_tan+math.pi/2) % math.pi-math.pi/2))
        # meniscus texture: elongated streak near rim, long axis parallel to boundary
        if not args.no_texture_filter and r_mean > 0.86 and elong > 2.2 and dphi < 20:
            n_tex += 1
            continue
        y_off, x_off = sl[0].start, sl[1].start
        # candidate merged blob -> watershed split; round compact blob -> single
        if elong > 2.5 or sz > 2.2*MED:
            n_col, subs = split_blob(m)
        else:
            n_col, subs = 1, [m]
        if n_col == 1:
            n_single += 1
        elif n_col == 2:
            n_pair += 1
        else:
            n_chain += 1
        total += n_col
        centroids.append((cy0, cx0, n_col))
        for sm_k in subs:
            bnd = sm_k & ~ndi.binary_erosion(sm_k)
            ys, xs = np.nonzero(bnd)
            col = (255, 40, 40) if len(subs) == 1 else (0, 90, 255)
            if len(xs) < 4:
                continue
            cy0, cx0 = ys.mean(), xs.mean()
            order = np.argsort(np.arctan2(ys-cy0, xs-cx0))
            mark_list.append(([(xs[o]+x_off, ys[o]+y_off) for o in order], col))

    # glare extrapolation from neighbouring ring (0.75..shrink, uncapped)
    extrap = 0.0
    if not args.no_extrapolate and glare.sum() > 0:
        ring_cs = [c for (cy0, cx0, c) in centroids if 0.75 <= rho[int(cy0), int(cx0)] < args.shrink]
        ring_area = ((rho >= 0.75) & (rho < args.shrink) & ~glare).sum()/1e6
        d_ring = sum(ring_cs)/max(ring_area, 1e-9)
        extrap = d_ring*glare.sum()/1e6

    # independent cross-check: local maxima, size-matched to the blob filter
    smd = ndi.gaussian_filter(diff, 1.5)
    peaks = (smd == ndi.maximum_filter(smd, 13)) & (smd > args.diff_thr) & region
    pl, pn = ndi.label(peaks)
    peak_count = 0
    if pn:
        blobs = ndi.binary_dilation(peaks, np.ones((9, 9))) & (diff > args.diff_thr*0.8) & region
        bl, bn = ndi.label(blobs)
        bsz2 = np.bincount(bl.ravel(), minlength=bn+1)
        blob_of_peak = ndi.maximum(bl, pl, np.arange(1, pn+1)).astype(int)
        ok = np.where(bsz2[blob_of_peak] >= min_area)[0] + 1
        if len(ok):
            pc = np.array(ndi.center_of_mass(peaks, pl, ok))
            pv = smd[pc[:, 0].astype(int), pc[:, 1].astype(int)]
            keep = np.ones(len(pc), bool)
            for i, j in cKDTree(pc).query_pairs(r=7.0):
                keep[j if pv[i] >= pv[j] else i] = False
            peak_count = int(keep.sum())

    # annotation: trace actual blob contours (tilted photo -> colonies are not circles)
    dr = ImageDraw.Draw(im)
    bpts = []
    for a in np.linspace(0, 2*math.pi, 360):
        px, py = A*args.shrink*math.cos(a), B*args.shrink*math.sin(a)
        bpts.append((cx+px*math.cos(th)-py*math.sin(th), cy+px*math.sin(th)+py*math.cos(th)))
    dr.line(bpts+[bpts[0]], fill=(255, 230, 0), width=6)
    for pts, col in mark_list:
        if len(pts) < 6:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            dr.ellipse([np.mean(xs)-9, np.mean(ys)-9, np.mean(xs)+9, np.mean(ys)+9],
                       outline=col, width=3)
            continue
        dr.line(pts+[pts[0]], fill=col, width=4, joint='curve')
    ann_path = f'{outdir}/annotated.jpg'
    im.save(ann_path, quality=88)

    final = total+extrap
    report = dict(image=args.image, final_count=round(final), direct_count=total,
                  extrapolated=round(extrap), crosscheck_peak_method=peak_count,
                  single=n_single, merged_pairs=n_pair, chains=n_chain,
                  texture_excluded=n_tex, px_per_mm=round(px_per_mm, 2),
                  min_colony_mm=args.min_colony_mm, agar_boundary_frac=args.shrink,
                  dish_ellipse=dict(cx=cx, cy=cy, A=A, B=B, theta=th),
                  note='red contour=1 colony, blue=merged (2+), yellow=agar boundary. '
                       'Uncertainty +/-10-15%. VERIFY boundary_check.jpg (yellow on agar/rim '
                       'edge in all sectors) and annotated.jpg (contours on visible colonies) '
                       'before reporting; if boundary is off, rerun with --shrink or --ellipse.')
    json.dump(report, open(f'{outdir}/count_report.json', 'w'), indent=1, ensure_ascii=False)
    print(f'single={n_single} pairs={n_pair}(x2) chains={n_chain} texture_excluded={n_tex}')
    print(f'direct={total} + extrapolated={extrap:.0f} -> FINAL ~{final:.0f} '
          f'(peak cross-check: {peak_count})')
    print(f'outputs: {ann_path} | {outdir}/count_report.json | {outdir}/boundary_check.jpg')
    print('NEXT: view boundary_check.jpg (yellow must sit on the agar/rim edge in all '
          'sectors) and annotated.jpg (contours must sit on visible colonies) before '
          'reporting the number. If the boundary is off: rerun with --shrink or --ellipse.')

if __name__ == '__main__':
    main()
