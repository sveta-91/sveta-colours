"""
Geometry detectors used by both the offline pipeline and the photo_manager HTTP backend.

Public surface (everything else is implementation detail):
    analyze(img_bgr) -> dict with the shape documented in
    docs/superpowers/plans/2026-06-05-python-backend-integration.md.

The detectors do NOT mutate the input image. They return coordinates/decisions only;
the actual warp/crop is the caller's job (the browser uses CPU homography from
photo_manager.html commit 69cae7c; the CLI uses run_pipeline.py's warpers).

Imports: only numpy and cv2 — intentionally decoupled from run_pipeline.py
(which is CLI-shaped: reads paths, writes paths, prints). The detector
function bodies below are pasted verbatim from run_pipeline.py; keep them in
sync if that file is ever refactored.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np


# ── Verbatim copies from run_pipeline.py ────────────────────────────────────


def _lab_histogram(region: np.ndarray) -> np.ndarray:
    """Compute a normalised LAB colour histogram for a region."""
    channels = [0, 1, 2]
    hist_size = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]
    hist = cv2.calcHist([region.astype("uint8")], channels, None, hist_size, ranges)
    cv2.normalize(hist, hist)
    return hist


def _check_hough_lines(img: np.ndarray) -> list[str]:
    """Return list of sides with strong straight border lines."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    margin = max(int(min(h, w) * 0.15), 10)
    mask = np.ones_like(edges) * 255
    mask[margin:h - margin, margin:w - margin] = 0
    border_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        border_edges, rho=1, theta=np.pi / 180, threshold=80,
        minLineLength=int(min(h, w) * 0.30), maxLineGap=10,
    )

    if lines is None:
        return []

    sides = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 5 or angle > 175:
            mid_y = (y1 + y2) / 2
            if mid_y < margin:
                sides.append("top")
            elif mid_y > h - margin:
                sides.append("bottom")
        elif 85 < angle < 95:
            mid_x = (x1 + x2) / 2
            if mid_x < margin:
                sides.append("left")
            elif mid_x > w - margin:
                sides.append("right")

    return sorted(set(sides))


def _check_corner_vs_center(img: np.ndarray):
    """Return list of corners whose colour differs strongly from the centre."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")

    patch = max(int(min(h, w) * 0.10), 10)
    ch, cw = h // 2, w // 2
    half = patch // 2
    center = lab[ch - half:ch + half, cw - half:cw + half]
    center_hist = _lab_histogram(center)

    corners = {
        "TL": lab[:patch, :patch],
        "TR": lab[:patch, w - patch:],
        "BL": lab[h - patch:, :patch],
        "BR": lab[h - patch:, w - patch:],
    }

    bg_corners_strict = []  # < 0.05
    bg_corners_soft = []    # < 0.15
    similarities = {}
    for name, region in corners.items():
        corner_hist = _lab_histogram(region)
        similarity = cv2.compareHist(corner_hist, center_hist, cv2.HISTCMP_CORREL)
        similarities[name] = similarity
        if similarity < 0.05:
            bg_corners_strict.append(name)
        if similarity < 0.15:
            bg_corners_soft.append(name)

    detail = " | ".join(f"{k}={v:.3f}" for k, v in similarities.items())
    logging.info("Corner-vs-center similarity: %s (%d strict, %d soft)", detail, len(bg_corners_strict), len(bg_corners_soft))

    return bg_corners_strict, bg_corners_soft, similarities


def _check_content_coverage(img: np.ndarray) -> tuple[float, str]:
    """Check what fraction of the image the main content fills.

    Returns (coverage, detail_string).  Coverage < 0.85 means significant
    background margins exist.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=3)

    coords = cv2.findNonZero(edges)
    if coords is None:
        return 1.0, "no edges"

    x, y, bw, bh = cv2.boundingRect(coords)
    coverage = (bw * bh) / (w * h)
    return coverage, f"bbox {bw}x{bh} at ({x},{y}) = {coverage:.0%}"


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _find_best_quad(img: np.ndarray) -> Optional[np.ndarray]:
    """Find the largest 4-sided contour that looks like a canvas/frame."""
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_area = 0

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > img_area * 0.20 and area > best_area:
                best = approx
                best_area = area

    return best


def _needs_straightening(img: np.ndarray):
    """Detect if the painting is photographed at an angle and find the quad to straighten.

    Looks for the largest 4-point contour (canvas/frame). If its corners
    deviate significantly from a perfect rectangle, the image needs
    perspective correction.

    Returns (needs_straightening, quad_or_None).
    """
    quad = _find_best_quad(img)
    if quad is None:
        return False, None

    h, w = img.shape[:2]
    img_area = h * w
    quad_area = cv2.contourArea(quad)
    coverage = quad_area / img_area

    # Only consider quads between 20-90% coverage. High-coverage quads
    # are likely painted content (architectural lines), not canvas edges.
    if coverage < 0.20 or coverage > 0.90:
        return False, None

    corners = _order_corners(quad.reshape(4, 2).astype("float32"))
    tl, tr, br, bl = corners

    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)

    w_ratio = min(top_w, bot_w) / max(top_w, bot_w) if max(top_w, bot_w) > 0 else 1.0
    h_ratio = min(left_h, right_h) / max(left_h, right_h) if max(left_h, right_h) > 0 else 1.0

    def _angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    angles = [
        _angle(bl, tl, tr),
        _angle(tl, tr, br),
        _angle(tr, br, bl),
        _angle(br, bl, tl),
    ]
    max_angle_dev = max(abs(a - 90) for a in angles)

    logging.info(
        "Straighten check: coverage=%.0f%% w_ratio=%.3f h_ratio=%.3f max_angle_dev=%.1f",
        coverage * 100, w_ratio, h_ratio, max_angle_dev,
    )

    if w_ratio < 0.70 or h_ratio < 0.70:
        return False, None

    if w_ratio < 0.95 or h_ratio < 0.95 or max_angle_dev > 3.0:
        return True, quad

    return False, None


def _line_intersection(p1, p2, p3, p4):
    """Find intersection of line (p1-p2) and line (p3-p4)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    ix = x1 + t * (x2 - x1)
    iy = y1 + t * (y2 - y1)
    return np.array([ix, iy], dtype="float32")


def _find_frame_corners_from_hough(img: np.ndarray) -> Optional[np.ndarray]:
    """Use Hough lines to find precise frame/canvas corners.

    More accurate than contour approximation because Hough gives exact
    line equations, and intersecting them gives exact corner positions.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    margin = int(min(h, w) * 0.15)
    mask = np.ones_like(edges) * 255
    mask[margin:h - margin, margin:w - margin] = 0
    border_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        border_edges, 1, np.pi / 180, 40,
        minLineLength=int(min(h, w) * 0.20), maxLineGap=15,
    )
    if lines is None:
        return None

    sides: dict[str, list] = {"top": [], "bottom": [], "left": [], "right": []}
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if angle < 10 or angle > 170:
            mid_y = (y1 + y2) / 2
            if mid_y < margin:
                sides["top"].append((x1, y1, x2, y2, length))
            elif mid_y > h - margin:
                sides["bottom"].append((x1, y1, x2, y2, length))
        elif 80 < angle < 100:
            mid_x = (x1 + x2) / 2
            if mid_x < margin:
                sides["left"].append((x1, y1, x2, y2, length))
            elif mid_x > w - margin:
                sides["right"].append((x1, y1, x2, y2, length))

    found = {k: v for k, v in sides.items() if v}
    if len(found) < 4:
        return None

    best = {}
    for name, line_list in found.items():
        longest = max(line_list, key=lambda l: l[4])
        best[name] = ((longest[0], longest[1]), (longest[2], longest[3]))

    tl = _line_intersection(*best["top"], *best["left"])
    tr = _line_intersection(*best["top"], *best["right"])
    br = _line_intersection(*best["bottom"], *best["right"])
    bl = _line_intersection(*best["bottom"], *best["left"])

    if any(p is None for p in [tl, tr, br, bl]):
        return None

    corners = np.array([tl, tr, br, bl], dtype="float32")
    for pt in corners:
        if pt[0] < -50 or pt[0] > w + 50 or pt[1] < -50 or pt[1] > h + 50:
            return None

    # Verify the lines represent real canvas edges (not painted content)
    # by checking the colour contrast across each edge.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype("float32")
    contrasts = []
    for side_name in ["top", "bottom", "left", "right"]:
        line = best[side_name]
        x1, y1 = line[0]
        x2, y2 = line[1]
        angle = np.arctan2(y2 - y1, x2 - x1)
        nx, ny = -np.sin(angle), np.cos(angle)
        strip = 15
        dists = []
        for t_val in np.linspace(0, 1, 15):
            px = int(x1 + t_val * (x2 - x1))
            py = int(y1 + t_val * (y2 - y1))
            ix1 = int(np.clip(px + nx * strip, 0, w - 1))
            iy1 = int(np.clip(py + ny * strip, 0, h - 1))
            ix2 = int(np.clip(px - nx * strip, 0, w - 1))
            iy2 = int(np.clip(py - ny * strip, 0, h - 1))
            dists.append(float(np.linalg.norm(lab[iy1, ix1] - lab[iy2, ix2])))
        contrasts.append(np.mean(dists))

    avg_contrast = np.mean(contrasts)
    logging.info("Hough edge contrast: %s avg=%.1f",
                 " | ".join(f"{c:.0f}" for c in contrasts), avg_contrast)

    if avg_contrast < 65:
        logging.info("Low cross-edge contrast — lines are painted content, not canvas edges")
        return None

    return corners


# ── Array-based replacement for run_pipeline._needs_crop ───────────────────


def _needs_crop_array(
    img: np.ndarray,
) -> tuple[bool, str, Optional[tuple[int, int, int, int]]]:
    """Three-signal crop detection — same logic as run_pipeline._needs_crop
    (line 384) but takes an np.ndarray directly and also returns the bbox.

    Returns (needs_crop, reason, bbox).
    bbox is (x, y, w, h) in pixels when content edges were detected, else None.
    The bbox is computed via cv2.boundingRect over the SAME dilated edge mask
    that _check_content_coverage uses — we recompute the mask here once to
    avoid changing _check_content_coverage's existing signature.
    """
    h, w = img.shape[:2]

    line_sides = _check_hough_lines(img)
    bg_strict, bg_soft, similarities = _check_corner_vs_center(img)
    coverage, cov_detail = _check_content_coverage(img)

    # Recompute the same edge mask _check_content_coverage uses so we can
    # extract the bounding rect. Mirrors that function's pipeline exactly.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=3)
    coords = cv2.findNonZero(edges)
    bbox: Optional[tuple[int, int, int, int]] = None
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        bbox = (int(x), int(y), int(bw), int(bh))

    logging.info(
        "Hough sides: %s | BG corners: %s (soft: %s) | Coverage: %s",
        line_sides or "none", bg_strict or "none", bg_soft or "none", cov_detail,
    )

    if len(line_sides) >= 3:
        return True, f"Needs crop — canvas edges on: {', '.join(line_sides)}", bbox
    if len(line_sides) >= 2 and coverage < 0.95:
        return True, f"Needs crop — canvas edges on: {', '.join(line_sides)}", bbox

    if len(bg_strict) >= 3:
        return True, f"Needs crop — background corners: {', '.join(bg_strict)}", bbox

    if len(bg_strict) >= 2 and len(line_sides) >= 1 and coverage < 0.95:
        return (
            True,
            f"Needs crop — background corners ({', '.join(bg_strict)}) + edge on {', '.join(line_sides)}",
            bbox,
        )

    if line_sides and bg_soft:
        side_to_adj_corners = {
            "top": ["TL", "TR"], "bottom": ["BL", "BR"],
            "left": ["TL", "BL"], "right": ["TR", "BR"],
        }
        for side in line_sides:
            adj = side_to_adj_corners.get(side, [])
            for c in adj:
                sim = similarities.get(c, 1.0)
                if 0.0 <= sim < 0.12:
                    return (
                        True,
                        f"Needs crop — edge on {side} confirmed by {c} ({sim:.3f})",
                        bbox,
                    )

    if coverage < 0.85:
        return True, f"Needs crop — content covers only {coverage:.0%} of image", bbox

    if len(bg_strict) == 2 and set(bg_strict) in [{"TL", "BR"}, {"TR", "BL"}]:
        return False, "Needs manual review — diagonal background", bbox

    return False, "No crop needed", bbox


# ── Public entry point ─────────────────────────────────────────────────────


def _compute_post_warp_crop(img_bgr: np.ndarray, corners: np.ndarray) -> Optional[dict]:
    """Apply the perspective warp internally, then run crop detection on the
    warped result. Returns crop as fractions of the warped output dimensions,
    or None when the warped image is already tight to the painting.

    Used to recover from Hough overshoot: even when corners include some
    background, this second pass catches and trims it.
    """
    xs = corners[:, 0]
    ys = corners[:, 1]
    out_w = int(round(max(xs) - min(xs)))
    out_h = int(round(max(ys) - min(ys)))
    if out_w < 50 or out_h < 50:
        return None

    src = corners.astype(np.float32)
    dst = np.array(
        [[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32
    )
    try:
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img_bgr, H, (out_w, out_h))
    except cv2.error:
        return None

    needs, _reason, bbox = _needs_crop_array(warped)
    if not needs or bbox is None:
        return None
    x, y, bw, bh = bbox
    # Skip near-identity crops — saves a redundant render in the browser.
    if bw / out_w > 0.97 and bh / out_h > 0.97:
        return None
    return {
        "x": x / out_w,
        "y": y / out_h,
        "w": bw / out_w,
        "h": bh / out_h,
    }


def analyze(img_bgr: np.ndarray) -> dict:
    """Top-level entry. Returns the dict shape the photo_manager expects.

    Coordinates are fractions [0,1] of original image dimensions.

    NOTE on corner ordering: _order_corners returns rows in the order
    [TL, TR, BR, BL] (confirmed against run_pipeline.py:445-454, which
    documents the same order). The dict-building loop below relies on
    that ordering — if _order_corners ever changes, update this too.
    """
    h, w = img_bgr.shape[:2]
    if h < 100 or w < 100:
        return {
            "needs_perspective_correction": False,
            "perspective_corners": None,
            "needs_crop": False,
            "crop": None,
            "source": "python",
            "reason": "image_too_small",
        }

    # 1) Try Hough-based 4-corner detection (covers both straighten + crop)
    corners = _find_frame_corners_from_hough(img_bgr)
    if corners is not None and len(corners) == 4:
        ordered = _order_corners(corners)  # TL, TR, BR, BL
        # Clip to image bounds — Hough can overshoot by ~50px (run_pipeline.py:645)
        clipped = np.clip(ordered, [[0, 0]], [[w - 1, h - 1]]).astype(np.float32)
        frac = clipped / np.array([w, h], dtype=np.float32)

        # Internally apply the warp and check whether the result still has
        # background bleed (Hough corners can sit on wall edges rather than
        # the painting's actual frame). Return the post-warp crop in fractions
        # of the WARPED image so the browser can apply it after its own warp.
        post_crop = _compute_post_warp_crop(img_bgr, clipped)

        result = {
            "needs_perspective_correction": True,
            "perspective_corners": {
                "tl": {"x": float(frac[0][0]), "y": float(frac[0][1])},
                "tr": {"x": float(frac[1][0]), "y": float(frac[1][1])},
                "br": {"x": float(frac[2][0]), "y": float(frac[2][1])},
                "bl": {"x": float(frac[3][0]), "y": float(frac[3][1])},
            },
            "source": "python",
            "reason": None,
        }
        if post_crop is not None:
            result["needs_crop"] = True
            result["crop"] = post_crop  # fractions in WARPED image space
        else:
            result["needs_crop"] = False
            result["crop"] = None
        return result

    # 2) No perspective needed — try a pure crop via the same _needs_crop signals
    needs, reason, bbox = _needs_crop_array(img_bgr)
    if needs and bbox is not None:
        x, y, bw, bh = bbox
        return {
            "needs_perspective_correction": False,
            "perspective_corners": None,
            "needs_crop": True,
            "crop": {
                "x": x / w,
                "y": y / h,
                "w": bw / w,
                "h": bh / h,
            },
            "source": "python",
            "reason": reason,
        }

    # 3) Neither — nothing to do, let browser fall back to Gemini coords if present
    return {
        "needs_perspective_correction": False,
        "perspective_corners": None,
        "needs_crop": False,
        "crop": None,
        "source": "python",
        "reason": "no_frame_detected",
    }
