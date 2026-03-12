from math import hypot
import math
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import cv2
from skimage.filters import frangi
import utils.projection_utils as projection_utils
# --- FAST direction + speed from last 10s (endpoints only) ---


def dir_speed_last10(hist_ts, hist_x, hist_y, min_span_s=3.0, min_disp_px=2.0):
    if len(hist_ts) < 2:
        return None
    t0, t1 = hist_ts.iloc[0], hist_ts.iloc[-1]
    span = (t1 - t0).total_seconds()
    x0, y0 = float(hist_x.iloc[0]), float(hist_y.iloc[0])
    x1, y1 = float(hist_x.iloc[-1]), float(hist_y.iloc[-1])
    dx, dy = x1 - x0, y1 - y0
    disp = hypot(dx, dy)
    if disp < min_disp_px:
        return None
    ux, uy = dx/disp, dy/disp
    speed_px_s = disp / span
    return ux, uy, speed_px_s


# --- Draw a trailing rectangle BEHIND (cx, cy) ---
def draw_trailing_rect(img, cx, cy, ux, uy, length_px, width_px=200):
    px, py = -uy, ux
    halfW = width_px / 2.0
    head_left = (cx - px*halfW, cy - py*halfW)
    head_right = (cx + px*halfW, cy + py*halfW)
    tail_cx, tail_cy = (cx - ux*length_px, cy - uy*length_px)
    tail_left = (tail_cx - px*halfW, tail_cy - py*halfW)
    tail_right = (tail_cx + px*halfW, tail_cy + py*halfW)
    p1 = (max(0, min(head_left[0], img.shape[1])),
          max(0, min(head_left[1], img.shape[0])))
    p2 = (max(0, min(head_right[0], img.shape[1])),
          max(0, min(head_right[1], img.shape[0])))
    p3 = (max(0, min(tail_right[0], img.shape[1])),
          max(0, min(tail_right[1], img.shape[0])))
    p4 = (max(0, min(tail_left[0], img.shape[1])),
          max(0, min(tail_left[1], img.shape[0])))
    poly = np.array([p1, p2, p3, p4], dtype=np.int32)

    # Draw arrow
    # tip = (int(cx), int(cy))
    # base = (int(cx - ux*max(20, length_px*0.25)),
    #         int(cy - uy*max(20, length_px*0.25)))

    return poly, (0, 0)


def get_directional_rectangle(img, df_filtered, timestamp, df_upsampled,
                              length_px=200, width_px=70,
                              color=(255, 0, 0), thickness=1,
                              draw_arrow=False, fill=False):

    trail_seconds = 30
    min_len, max_len = 50, 500
    """
    For each aircraft present at `timestamp`, draw a 100x20 px rectangle
    oriented along its motion estimated from the previous 10 seconds.

    df_filtered columns used: ['ident', 'timestamp', 'image_x', 'image_y']
    - `timestamp` should be pandas.Timestamp (UTC ok).
    """
    out = img.copy()

    if not isinstance(timestamp, pd.Timestamp):
        timestamp = pd.to_datetime(timestamp, utc=True, errors='coerce')

    # aircraft visible at this frame
    cur = df_filtered
    if cur.empty:
        return {}

    # previous-10s window for only those aircraft
    prev_start = timestamp - timedelta(seconds=10)
    prev = df_upsampled[
        (df_upsampled['time'] >= prev_start) &
        (df_upsampled['time'] < timestamp) &
        (df_upsampled['ident'].isin(cur['ident']))
    ]

    # group prev by ident for quick lookup
    prev_groups = {k: g for k, g in prev.groupby('ident')}
    result = {}
    for _, row in cur.iterrows():
        ident = row['ident']
        cx, cy = float(row['image_x']), float(row['image_y'])
        if ident not in prev_groups:
            continue

        g = prev_groups[ident]
        res = dir_speed_last10(g['time'], g['image_x'], g['image_y'])

        if res is None:
            continue
        else:
            ux, uy, speed_px_s = res

        # --- scale length by apparent speed ---
        length = float(np.clip(speed_px_s * trail_seconds, min_len, max_len))
        if np.any(np.isnan([ux, uy, cx, cy])):
            continue
        rect, arrow = draw_trailing_rect(out, cx, cy, ux, uy,
                                         length_px=length, width_px=width_px)
        result[ident] = (rect, arrow, res)
    return result


def normalize(vx, vy, eps=1e-9):
    n = math.hypot(vx, vy)
    if n < eps:
        return 1.0, 0.0  # default to +x if degenerate
    return vx / n, vy / n


def angle180_from_vec(vx, vy):
    # Orientation modulo 180° (lines: θ == θ+180)
    a = math.degrees(math.atan2(vy, vx))  # [-180, 180)
    return a % 180.0


def compute_dominant_line_angle(rect_dir_vec, edges_final):
    """
    Compute the dominant line angle from Hough lines within the given edge map,
    filtering lines that align with the rectangle direction vector.

    Args:
        rect_dir_vec: Tuple (ux, uy) representing the rectangle direction vector components.
        edges_final: Binary edge map (numpy array) for Hough transform.

    Returns:
        best_angle: Dominant line angle in degrees (None if not found).
        angle_offset_from_rect: Signed offset from rectangle direction in degrees (None if not found).
    """
    tolerance_deg = 8.0  # degrees
    # ---- Inputs you already have ----
    # ux, uy: rectangle direction vector components (image coords: +x right, +y down)
    ux, uy = rect_dir_vec  # e.g., (ux, uy) from your data
    edges = edges_final  # or edges_big

    # Normalize the rectangle direction
    rx, ry = normalize(ux, uy)
    rect_angle = angle180_from_vec(rx, ry)  # for reporting/offset
    def norm(a): return (a + 180) % 180 - 90
    # Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.deg2rad(1.0),
        threshold=20,
        minLineLength=8,
        maxLineGap=2
    )
    if lines is None:
        return 0.0, []
    good_lines = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2(y2-y1, x2-x1))
        if abs(norm(ang) - norm(rect_angle)) <= tolerance_deg:
            good_lines.append((x1, y1, x2, y2))
    if not good_lines:
        return 0.0, []
    lines_with_lengths = [(x1, y1, x2, y2, np.hypot(x2-x1, y2-y1))
                          for (x1, y1, x2, y2) in good_lines]
    score = len(good_lines)
    return score, lines_with_lengths


def resize_rect_polygon(rect_poly: np.ndarray, delta_px: int) -> np.ndarray:
    """
    Grow/shrink a (possibly rotated) rectangle polygon by delta_px on all sides
    without using morphology. Works by adjusting the min-area rectangle's width/height.
    rect_poly: (N,2) int/float array with the rectangle's 4 vertices (or more along edges).
    delta_px:  positive to expand, negative to shrink.
    Returns: (4,2) int32 polygon in image coords.
    """
    rect_poly = rect_poly.astype(np.float32)
    rect = cv2.minAreaRect(rect_poly)  # ((cx,cy), (w,h), angle)
    (cx, cy), (w, h), angle = rect

    new_w = max(1.0, w + 2.0 * delta_px)
    new_h = max(1.0, h + 2.0 * delta_px)

    box = cv2.boxPoints(((cx, cy), (new_w, new_h), angle))  # (4,2) float32
    return np.round(box).astype(np.int32)


def _compute_edges_for_rectangles(gray_blurred, rectangles_dict, border_px=7):
    """
    Helper function to compute Canny edges for each rectangle.
    Returns a dictionary: {ident: edges_masked}
    """
    H, W = gray_blurred.shape[:2]
    edges_dict = {}

    for ident, (rect_poly, arrow, direction_info) in rectangles_dict.items():
        # 1) Expand mask geometry by +border_px
        big_poly = resize_rect_polygon(rect_poly, +border_px)

        # 2) Build expanded ROI mask and run Canny only in that ROI
        mask_big = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_big, [big_poly], 255)

        vals = gray_blurred[mask_big > 0]
        if vals.size == 0:
            t = 0
        else:
            # try p85–p95; or replace with your masked_otsu
            t = int(np.percentile(vals, 96))
        p95 = np.percentile(vals, 99.5)
        high = max(int(p95), 60)
        low = int(0.25 * high)
        roi_big = cv2.bitwise_and(gray_blurred, gray_blurred, mask=mask_big)
        _, roi_big = cv2.threshold(roi_big, t, 255, cv2.THRESH_TOZERO)
        edges_big = cv2.Canny(roi_big, low, high)

        # 3) Keep only edges inside the original rectangle
        mask_small = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask_small, [rect_poly], 255)
        edges_final = cv2.bitwise_and(edges_big, edges_big, mask=mask_small)

        edges_dict[ident] = (edges_final, rect_poly)

    return edges_dict


def apply_canny_to_rectangles(img, prev_img, rectangles_dict,
                              blur_kernel=(3, 3),
                              border_px=7,
                              min_line_length=40.0,
                              ):
    """
    Apply Canny edge detection to rectangular regions returned by get_directional_rectangle.

    Args:
        img: Input image (BGR format)
        rectangles_dict: Dictionary from get_directional_rectangle with format:
                        {ident: (rect_poly, arrow, direction_info)}
        lower_threshold: Lower threshold for Canny edge detection (default: 50)
        upper_threshold: Upper threshold for Canny edge detection (default: 150)
        blur_kernel: Kernel size for Gaussian blur preprocessing (default: (5, 5))
        overlay_edges: If True, overlay edges on original image; if False, return edge images
        edge_color: Color for edge visualization when overlay_edges=True (BGR format)
        border_px: Pixels to expand rectangle for edge detection to avoid edge artifacts (default: 7)

    Returns:
        If overlay_edges=True: Image with edges overlaid
        If overlay_edges=False: Dictionary {ident: edge_data} for each aircraft
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray, prev_gray)
    # Apply Gaussian blur to reduce noise
    if blur_kernel is not None and blur_kernel[0] > 0:
        gray_blurred = cv2.GaussianBlur(diff, blur_kernel, 0)
        # gray_blurred = cv2.threshold(gray_blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    else:
        gray_blurred = gray

    # Compute edges for all rectangles (shared logic)
    edges_dict = _compute_edges_for_rectangles(
        gray_blurred, rectangles_dict, border_px)

    # Create output image and overlay all edges
    output = img.copy()

    # for ident, (edges_final, rect_poly) in edges_dict.items():
    #     ys, xs = np.where(edges_final > 0)
    #     output[ys, xs] = edge_color

    # Return individual edge images per aircraft with statistics
    edge_data = {}

    for ident, (edges_final, rect_poly) in edges_dict.items():
        # Get bounding box of rectangle to crop ROI
        x, y, w, h = cv2.boundingRect(rect_poly)
        direction_info = rectangles_dict[ident][2]
        ux, uy, speed_px_s = direction_info
        # Crop the ROI and edges
        roi_cropped = gray_blurred[y:y+h, x:x+w]
        edges_cropped = edges_final[y:y+h, x:x+w]

        is_contrail = False
        best_angle = None
        score, lines_with_length = 0, []
        if np.count_nonzero(edges_cropped) > 0:
            score, lines_with_length = compute_dominant_line_angle(
                (ux, uy), edges_final)
            # atleast 2 lines are greater than min_line_length
            good_lines_with_min_length = [
                length for *_, length in lines_with_length if length >= min_line_length]

            is_contrail = (score >= 2) and (
                len(good_lines_with_min_length) >= 2)

        edge_data[ident] = {
            'edges': edges_cropped,
            'roi': roi_cropped,
            'bbox': (x, y, w, h),
            'edge_pixel_count': np.count_nonzero(edges_cropped),
            'is_making_contrails': is_contrail,
            'best_line_angle': best_angle,
            "score": score,
            "lines": lines_with_length,
            'edge_density': np.count_nonzero(edges_cropped) / (w * h) if w * h > 0 else 0
        }
    return output, edge_data, edges_dict


def calculate_edge_statistics(edge_data_dict):
    """
    Calculate statistics for edge detection results.

    Args:
        edge_images_dict: Dictionary from apply_canny_to_rectangles (with overlay_edges=False)

    Returns:
        DataFrame with edge statistics per aircraft
    """
    stats = []
    print("Edge Detection Statistics:")
    print()
    for ident, data in edge_data_dict.items():
        stats.append({
            'ident': ident,
            'edge_pixel_count': data['edge_pixel_count'],
            'edge_density': data['edge_density'],
            'bbox_width': data['bbox'][2],
            'bbox_height': data['bbox'][3],
            'bbox_area': data['bbox'][2] * data['bbox'][3]
        })

    return pd.DataFrame(stats)


def process_image_with_canny_edges(img_path, prev_img_path, timestamp, df_filtered, df_upsampled, min_line_length=40.0,):
    """
    Process a single image: load, detect rectangles, apply Canny edge detection.

    Args:
        img_path: Path to image file
        timestamp: Timestamp for this image (pandas.Timestamp)
        df_filtered: DataFrame with aircraft tracking data
        draw_rectangles: Whether to draw rectangle outlines
        draw_edges: Whether to draw detected edges
        lower_threshold: Canny lower threshold
        upper_threshold: Canny upper threshold

    Returns:
        Processed image, rectangles dictionary, edge statistics
    """
    # Load image
    img = cv2.imread(img_path)
    prev_img = cv2.imread(prev_img_path)
    if img is None or prev_img is None:
        return None, None, None, None

    # Get directional rectangles
    rectangles = get_directional_rectangle(
        img, df_filtered, timestamp, df_upsampled, length_px=200, width_px=100)
    if len(rectangles) == 0:
        return img, {}, pd.DataFrame(), {}

    # Apply edge detection
    img_output, edge_data, edges_dict = apply_canny_to_rectangles(
        img, prev_img, rectangles,
        blur_kernel=(3, 3),
        min_line_length=min_line_length,
    )

    return img_output, rectangles, edge_data, edges_dict


def get_flight_distance(gps_flight, gps_origin):
    flight_ecef = projection_utils.gps_to_ecef(
       gps_flight)
    origin_ecef = projection_utils.gps_to_ecef(
        gps_origin)
    # Convert ECEF to ENU
    distance = math.dist(flight_ecef, origin_ecef)
    return distance


def convert_texture_to_gps_points(texture, flight_gps, gps_origin, k_matrix, r_matrix, tvec, dist_coeffs=None):
    ys, xs = np.where(texture > 0)
    flight_distance = get_flight_distance(flight_gps, gps_origin)
    # create an array [] of pixel points
    pixel_points = np.array([[x, y] for y, x in zip(ys, xs)], dtype=np.float32)
    gps_coords = projection_utils.image_to_gps(
        image_points=pixel_points,
        k_matrix=k_matrix,
        r_matrix=r_matrix,
        t_vector=tvec,
        dist_coeffs=dist_coeffs,
        camera_gps=gps_origin,
        distance_m=flight_distance
    )

    return gps_coords
