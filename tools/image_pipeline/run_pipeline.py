from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from googleapiclient.http import MediaIoBaseDownload
from PIL import Image

from drive_auth import get_drive_service


# ------------------------------------------------------------
# Sveta Colours — image pipeline v1 scaffold
# Current scope:
# - local Python pipeline living inside the site repo
# - creates required local directories on start
# - loads config from JSON
# - prepares clear stages for future Drive download/upload
# - prepares clear stages for future image processing
# - no Google Sheet updates in v1
# ------------------------------------------------------------


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"


@dataclass
class PipelineConfig:
    google_drive_inbox_folder_id: str
    google_drive_processed_folder_id: str
    google_drive_needs_review_folder_id: str
    google_service_account_key_path: Path
    local_tmp_dir: Path
    local_output_dir: Path
    local_logs_dir: Path
    jpg_quality: int = 92
    overwrite_existing: bool = False
    dry_run: bool = True

    @property
    def processed_dir(self) -> Path:
        return self.local_output_dir / "processed"

    @property
    def portrait_dir(self) -> Path:
        return self.processed_dir / "portrait"

    @property
    def landscape_dir(self) -> Path:
        return self.processed_dir / "landscape"

    @property
    def needs_review_dir(self) -> Path:
        return self.local_output_dir / "needs-review"


@dataclass
class ImageJob:
    source_id: str
    source_name: str
    local_source_path: Path


@dataclass
class ProcessResult:
    status: str  # processed | needs-review | skipped | failed
    output_path: Path | None
    orientation: str | None  # portrait | landscape | None
    reason: str | None = None


class PipelineError(Exception):
    pass


def load_config(config_path: Path) -> PipelineConfig:
    if not config_path.exists():
        raise PipelineError(
            f"Missing config file: {config_path}\n"
            "Copy config.example.json to config.json and fill in the folder IDs."
        )

    raw = json.loads(config_path.read_text(encoding="utf-8"))

    return PipelineConfig(
        google_drive_inbox_folder_id=raw["google_drive_inbox_folder_id"],
        google_drive_processed_folder_id=raw["google_drive_processed_folder_id"],
        google_drive_needs_review_folder_id=raw["google_drive_needs_review_folder_id"],
        google_service_account_key_path=(ROOT / raw.get("google_service_account_key_path", "service-account-key.json")).resolve(),
        local_tmp_dir=(ROOT / raw.get("local_tmp_dir", "tmp")).resolve(),
        local_output_dir=(ROOT / raw.get("local_output_dir", "output")).resolve(),
        local_logs_dir=(ROOT / raw.get("local_logs_dir", "logs")).resolve(),
        jpg_quality=int(raw.get("jpg_quality", 92)),
        overwrite_existing=bool(raw.get("overwrite_existing", False)),
        dry_run=bool(raw.get("dry_run", True)),
    )


def ensure_directories(config: PipelineConfig) -> None:
    required_dirs = [
        config.local_tmp_dir,
        config.local_output_dir,
        config.local_logs_dir,
        config.processed_dir,
        config.portrait_dir,
        config.landscape_dir,
        config.needs_review_dir,
    ]

    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(config: PipelineConfig) -> None:
    log_file = config.local_logs_dir / "pipeline.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def list_drive_source_images(config: PipelineConfig) -> list[dict]:
    """List image files in the Drive inbox folder."""
    logging.info("Listing source images from Drive inbox folder: %s", config.google_drive_inbox_folder_id)

    service = get_drive_service(config.google_service_account_key_path)
    query = (
        f"'{config.google_drive_inbox_folder_id}' in parents"
        " and trashed = false"
        " and ("
        "mimeType = 'image/jpeg'"
        " or mimeType = 'image/heic'"
        " or mimeType = 'image/png'"
        ")"
    )

    results: list[dict] = []
    page_token: str | None = None

    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType)",
            pageSize=100,
            pageToken=page_token,
        ).execute()

        results.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def download_drive_file(file_meta: dict, config: PipelineConfig) -> Path:
    """Download a file from Drive into the local tmp directory."""
    target_path = config.local_tmp_dir / file_meta["name"]
    logging.info("Downloading %s -> %s", file_meta["name"], target_path)

    if config.dry_run:
        logging.info("[dry-run] Would download %s", file_meta["name"])
        return target_path

    service = get_drive_service(config.google_service_account_key_path)
    request = service.files().get_media(fileId=file_meta["id"])

    with open(target_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    logging.info("Downloaded %s (%d bytes)", file_meta["name"], target_path.stat().st_size)
    return target_path


def normalize_input_to_jpeg(local_source_path: Path, config: PipelineConfig) -> Path:
    """Convert HEIC/PNG/other formats to JPEG. Pass-through if already JPEG."""
    logging.info("Normalizing input to JPEG: %s", local_source_path.name)

    if not local_source_path.exists():
        logging.info("[dry-run] Source file not on disk, skipping normalization")
        return local_source_path

    suffix = local_source_path.suffix.lower()

    # Register HEIC support
    if suffix in (".heic", ".heif"):
        import pillow_heif
        pillow_heif.register_heif_opener()

    img = Image.open(local_source_path)
    img = img.convert("RGB")

    # If already JPEG, just return — no re-encode needed
    if suffix in (".jpg", ".jpeg"):
        img.close()
        return local_source_path

    # Convert to JPEG
    jpeg_path = local_source_path.with_suffix(".jpg")
    img.save(jpeg_path, "JPEG", quality=config.jpg_quality)
    img.close()
    logging.info("Converted %s -> %s", local_source_path.name, jpeg_path.name)
    return jpeg_path


def _detect_content_orientation(img_path: Path) -> str:
    """Detect orientation by analysing the actual artwork content area.

    For clearly non-square images, pixel dimensions are enough.
    For square or near-square images, find the bounding box of the main
    content (edges/detail) vs uniform background to determine the real
    artwork aspect ratio.
    """
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    aspect = w / h

    # Clearly not square — dimensions are reliable
    if aspect > 1.08:
        return "landscape"
    if aspect < 0.92:
        return "portrait"

    # Near-square: analyse content to find the artwork bounds
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate to connect nearby edges into solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=3)

    # Find the bounding rect of all edge pixels
    coords = cv2.findNonZero(edges)
    if coords is None:
        return "landscape"  # fallback

    x, y, bw, bh = cv2.boundingRect(coords)
    content_aspect = bw / bh if bh > 0 else 1.0

    logging.info(
        "Content bounds: %dx%d at (%d,%d) in %dx%d image — content aspect %.2f",
        bw, bh, x, y, w, h, content_aspect,
    )

    if content_aspect < 0.95:
        return "portrait"
    if content_aspect > 1.05:
        return "landscape"

    # Still ambiguous — check which margins are larger (more background)
    margin_left_right = x + (w - (x + bw))
    margin_top_bottom = y + (h - (y + bh))

    if margin_left_right > margin_top_bottom:
        return "portrait"  # more side padding = tall content
    return "landscape"


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


def _check_corner_vs_center(img: np.ndarray) -> list[str]:
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


def _lab_histogram(region: np.ndarray) -> np.ndarray:
    """Compute a normalised LAB colour histogram for a region."""
    channels = [0, 1, 2]
    hist_size = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]
    hist = cv2.calcHist([region.astype("uint8")], channels, None, hist_size, ranges)
    cv2.normalize(hist, hist)
    return hist


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


def _needs_crop(img_path: Path) -> tuple[bool, str]:
    """Determine whether the image has background that should be cropped.

    Uses three complementary signals:
    1. Hough line detection — catches clear canvas/frame edges
    2. Corner-vs-centre colour histogram — catches backgrounds without
       straight edges (dark edges, wooden tables)
    3. Content coverage — catches cluttered backgrounds (easels, other art)
    """
    img = cv2.imread(str(img_path))

    line_sides = _check_hough_lines(img)
    bg_strict, bg_soft, similarities = _check_corner_vs_center(img)
    coverage, cov_detail = _check_content_coverage(img)

    logging.info(
        "Hough sides: %s | BG corners: %s (soft: %s) | Coverage: %s",
        line_sides or "none", bg_strict or "none", bg_soft or "none", cov_detail,
    )

    # Hough lines on 3-4 sides are a very strong signal regardless of coverage.
    # On just 2 sides, only trust them when content doesn't fill the whole
    # frame — paintings with architectural content create false-positive lines.
    if len(line_sides) >= 3:
        return True, f"Needs crop — canvas edges on: {', '.join(line_sides)}"
    if len(line_sides) >= 2 and coverage < 0.95:
        return True, f"Needs crop — canvas edges on: {', '.join(line_sides)}"

    if len(bg_strict) >= 3:
        return True, f"Needs crop — background corners: {', '.join(bg_strict)}"

    # Weaker signals reinforce each other
    if len(bg_strict) >= 2 and len(line_sides) >= 1 and coverage < 0.95:
        return True, f"Needs crop — background corners ({', '.join(bg_strict)}) + edge on {', '.join(line_sides)}"

    # Hough line confirmed by a corner with moderately low similarity
    # (0.0 to 0.12).  Very negative values (< 0) indicate painted content
    # with different colours, not background — exclude those.
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
                    return True, f"Needs crop — edge on {side} confirmed by {c} ({sim:.3f})"

    if coverage < 0.85:
        return True, f"Needs crop — content covers only {coverage:.0%} of image"

    return False, "No crop needed"


def detect_artwork_and_process(local_jpeg_path: Path, config: PipelineConfig) -> ProcessResult:
    """Detect orientation and whether crop is needed. Route to correct folder."""
    logging.info("Processing image: %s", local_jpeg_path.name)

    if not local_jpeg_path.exists():
        logging.info("[dry-run] Source file not on disk, skipping processing")
        placeholder_output = config.needs_review_dir / local_jpeg_path.name
        return ProcessResult(status="processed", output_path=placeholder_output, orientation=None, reason="dry-run")

    orientation = _detect_content_orientation(local_jpeg_path)

    needs_crop, crop_reason = _needs_crop(local_jpeg_path)
    logging.info("Crop check: %s", crop_reason)

    output_dir = config.portrait_dir if orientation == "portrait" else config.landscape_dir
    output_path = output_dir / local_jpeg_path.name

    img = Image.open(local_jpeg_path)
    w, h = img.size
    img.close()
    logging.info("Orientation: %s (%dx%d)", orientation, w, h)

    return ProcessResult(
        status="processed",
        output_path=output_path,
        orientation=orientation,
        reason=crop_reason,
    )


def save_processed_output(result: ProcessResult, normalized_path: Path, config: PipelineConfig) -> ProcessResult:
    """Save the processed JPEG to the appropriate output folder."""
    if result.output_path is None:
        return result

    if not normalized_path.exists():
        logging.info("[dry-run] Source file not on disk, skipping save")
        return result

    img = Image.open(normalized_path)
    img = img.convert("RGB")
    result.output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(result.output_path, "JPEG", quality=config.jpg_quality)
    img.close()
    logging.info("Saved local output: %s (%d bytes)", result.output_path, result.output_path.stat().st_size)
    return result


def upload_result_to_drive(result: ProcessResult, config: PipelineConfig) -> None:
    """
    TODO:
    Replace this stub with Google Drive upload logic.

    Rules for v1:
    - processed files go to processed Drive folder
    - needs-review files go to needs-review Drive folder
    - no Google Sheet updates yet
    """
    if result.output_path is None:
        logging.info("No output to upload.")
        return

    if config.dry_run:
        logging.info("[dry-run] Would upload %s (status=%s)", result.output_path.name, result.status)
        return

    logging.info("Uploading result to Drive: %s", result.output_path.name)


def cleanup_temp_files(paths: Iterable[Path]) -> None:
    for path in paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except Exception as exc:
            logging.warning("Failed to delete temp file %s: %s", path, exc)


def process_one_file(file_meta: dict, config: PipelineConfig) -> None:
    local_source_path: Path | None = None
    normalized_path: Path | None = None

    try:
        logging.info("---")
        logging.info("Starting file: %s", file_meta.get("name", "<unknown>"))

        local_source_path = download_drive_file(file_meta, config)
        normalized_path = normalize_input_to_jpeg(local_source_path, config)
        result = detect_artwork_and_process(normalized_path, config)
        result = save_processed_output(result, normalized_path, config)
        upload_result_to_drive(result, config)

        logging.info(
            "Finished file: %s | status=%s | reason=%s",
            file_meta.get("name", "<unknown>"),
            result.status,
            result.reason,
        )

    except Exception as exc:
        logging.exception("Failed file %s: %s", file_meta.get("name", "<unknown>"), exc)

    finally:
        cleanup_temp_files([p for p in [local_source_path, normalized_path] if p is not None])


def run_pipeline(config: PipelineConfig) -> None:
    ensure_directories(config)
    configure_logging(config)

    logging.info("Sveta Colours image pipeline started")
    logging.info("Dry run: %s", config.dry_run)

    source_files = list_drive_source_images(config)
    logging.info("Found %d source files", len(source_files))

    for file_meta in source_files:
        process_one_file(file_meta, config)

    logging.info("Pipeline finished")


def main() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    run_pipeline(config)


if __name__ == "__main__":
    main()
