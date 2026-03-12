from pyproj import Transformer, CRS
import numpy as np
import cv2
import pymap3d as pm
import json

def ecef_to_enu(origin_ecef, points_ecef):

    # use pymap3d to convert ecef to enu
    enu_points = pm.ecef2enu(points_ecef[:, 0], points_ecef[:, 1], points_ecef[:, 2],
                              origin_ecef[0], origin_ecef[1], origin_ecef[2])
    # make numpy array
    return np.array(enu_points).T


def gps_to_camxy_vasha_fixed(lats, lons, alts, cam_k, cam_r, cam_t, camera_gps, distortion=None):
    """
    Fixed version of GPS to camera coordinates conversion.
    Properly handles objects behind camera and outside frame.
    """
    # Convert GPS to ECEF
    transformer_geodetic_to_ecef = Transformer.from_crs(
        "epsg:4979", "epsg:4978", always_xy=True)
    eX, eY, eZ = transformer_geodetic_to_ecef.transform(lons, lats, alts)
    ecef_points = np.column_stack((eX, eY, eZ))

    # Convert ECEF to ENU relative to camera
    enu_points = ecef_to_enu(
        camera_gps, ecef_points)  # Shape: (N, 3)

    # CRITICAL FIX 1: Transform to camera coordinate system properly
    # Camera coordinates: X=right, Y=down, Z=forward (into scene)
    points_cam = (cam_r @ enu_points.T + cam_t).T  # Shape: (N, 3)
# # 2. Calculate normalized coordinates (u, v)
# These represent the position on a flat plane 1 unit in front of the lens
    z_depth = points_cam[:, 2]
    u_norm = points_cam[:, 0] / z_depth
    v_norm = points_cam[:, 1] / z_depth

    # 3. Apply a FOV (Field of View) Guard Rail
    # For a standard lens, u_norm and v_norm rarely exceed 2.0 at the very edges.
    # If the value is > 3.0, it's way outside the FOV and will cause distortion 'wrap-around'.
    fov_mask = (z_depth > 1.0) & (np.abs(u_norm) < 1.2) & (np.abs(v_norm) < 1.2)

    num_points = len(lats)
    image_x = np.full(num_points, np.nan)
    image_y = np.full(num_points, np.nan)
    
    if np.any(fov_mask):
        # Filter the points
        valid_enu = enu_points[fov_mask]
        
        # Prepare transformation vectors for OpenCV
        rvec, _ = cv2.Rodrigues(cam_r)
        tvec = cam_t.reshape(3, 1) # Ensure correct shape

        # 3. Project only the valid points
        # cv2.projectPoints handles the R and T transformation internally
        img_pts, _ = cv2.projectPoints(
            valid_enu.astype(np.float32), 
            rvec, 
            tvec, 
            cam_k.astype(np.float32), 
            distortion.astype(np.float32) if distortion is not None else None
        )
        
        valid_indices = np.where(fov_mask)[0]
        image_x[valid_indices] = img_pts[:, 0, 0]
        image_y[valid_indices] = img_pts[:, 0, 1]

    return image_x, image_y, points_cam[:, 2]

def gps_to_ecef(points_gps):
    transformer_geodetic_to_ecef = Transformer.from_crs(
        "epsg:4979", "epsg:4978", always_xy=True)
    eX, eY, eZ = transformer_geodetic_to_ecef.transform(
        points_gps[:, 1], points_gps[:, 0], points_gps[:, 2])
    ecef_points = np.vstack((eX, eY, eZ)).T  # Shape: (N, 3)
    return ecef_points

def estimate_camera_params(origin_gps, poi_ecef, poi_xy, frame_size, intrinsics_estimate=None, distortion_estimate=None, rvecs=None, tvecs=None):
    # Convert camera GPS coordinates to ECEF
    transformer_geodetic_to_ecef = Transformer.from_crs(
        "epsg:4979", "epsg:4978", always_xy=True)
    # Note: pyproj expects lon,lat,alt order
    eX, eY, eZ = transformer_geodetic_to_ecef.transform(
        origin_gps[1], origin_gps[0], origin_gps[2])
    cam_ecef = np.array([eX, eY, eZ]).T  # /1000  # Shape: (3,)
    # eX, eY, eZ = transformer_geodetic_to_ecef.transform(
    #     poi_gps[:, 1], poi_gps[:, 0], poi_gps[:, 2])
    # poi_ecef = np.vstack((eX, eY, eZ)).T  # /1000  # Shape: (3, N)
    poi_enu = ecef_to_enu(
        origin_gps, poi_ecef)  # Shape: (N,3)

    # Put everything in contiguous arrays with (N, 3), not (3, N)
    object_points = np.ascontiguousarray(poi_enu.astype(np.float32))  # (N, 3)
    image_points = np.ascontiguousarray(
        poi_xy.astype(np.float32))      # (N, 2)

    if intrinsics_estimate is None:
        estimated_focal_dist = 1e5
        intrinsics_estimate = np.array([[estimated_focal_dist, 0, frame_size[1] / 2],
                                        [0, estimated_focal_dist,
                                            frame_size[0] / 2],
                                        [0, 0, 1]], dtype=np.float32)  # (3, 3)
    if distortion_estimate is None:
        # Use a zero distortion model if not provided
        distortion_estimate = np.zeros((5, 1), dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10000,    # Maximum 10,000 iterations (vs default ~30)
        1e-12     # Extremely tight convergence (vs default 1e-6)
    )
    print("Initial intrinsics:\n", intrinsics_estimate)
    print("Initial distortion:\n", distortion_estimate.ravel())
    calibrate_flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS |         # Use your good initial guess
        # cv2.CALIB_USE_LU |                    # Use your good initial guess
        cv2.CALIB_FIX_PRINCIPAL_POINT  |        # Keep principal point fixed
        # cv2.CALIB_FIX_FOCAL_LENGTH |          # Keep focal lengths fixed
        cv2.CALIB_FIX_ASPECT_RATIO |            # Keep fx/fy ratio fixed
        # cv2.CALIB_FIX_K1 |
        # cv2.CALIB_FIX_K2 | 
        # cv2.CALIB_FIX_K3 |      # keep all c
        # cv2.CALIB_FIX_P1 | 
        # CALIB_FIX_P2 |  #fix tangential distortion
        # fix higher order radial distortions
        cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6 |
        cv2.CALIB_ZERO_TANGENT_DIST           # Only estimate radial distortion
        | cv2.CALIB_USE_EXTRINSIC_GUESS
    )
    # Let OpenCV estimate all distortion parameters
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [object_points], [image_points], frame_size,
        intrinsics_estimate,
        distortion_estimate,
        rvecs,
        tvecs,
        flags=calibrate_flags,
        criteria=criteria
    )

    T = tvecs[0]  # .reshape(3, 1)  # Reshape to (3, 1)
    R, _ = cv2.Rodrigues(rvecs[0])

    return camera_matrix, dist_coeffs, R, T, cam_ecef


def calculate_fov_from_intrinsics(intrinsics, image_width, image_height, distortion=None):
    """
    Calculate horizontal and vertical field of view from camera intrinsics matrix.

    *** NOTE: This method assumes no distortion is applied.
    Without distortion -  ***

    Args:
        intrinsics: 3x3 camera intrinsics matrix (K matrix)
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        distortion: Optional distortion coefficients (not used in this calculation)

    Returns:
        hfov: Horizontal field of view in radians
        vfov: Vertical field of view in radians
        hfov_deg: Horizontal field of view in degrees
        vfov_deg: Vertical field of view in degrees
    """
    # Extract focal lengths and principal point
    fx = intrinsics[0, 0]  # focal length in x (pixels)
    fy = intrinsics[1, 1]  # focal length in y (pixels)
    cx = intrinsics[0, 2]  # principal point x
    cy = intrinsics[1, 2]  # principal point y

    # Left and right angles from principal point
    angle_left = np.arctan(cx / fx)
    angle_right = np.arctan((image_width - cx) / fx)
    hfov = angle_left + angle_right

    # Top and bottom angles from principal point
    angle_top = np.arctan(cy / fy)
    angle_bottom = np.arctan((image_height - cy) / fy)
    vfov = angle_top + angle_bottom

    # Method 2: Simplified calculation (assumes centered principal point)
    # hfov_simple = 2 * np.arctan(image_width / (2 * fx))
    # vfov_simple = 2 * np.arctan(image_height / (2 * fy))

    # Convert to degrees for display
    hfov_deg = np.degrees(hfov)
    vfov_deg = np.degrees(vfov)

    # return hfov, vfov, hfov_deg, vfov_deg
    return (hfov_deg.item(), vfov_deg.item())

def image_to_gps(image_points, k_matrix, r_matrix, t_vector, dist_coeffs, camera_gps, distance_m=1000):
    """
    Convert one or multiple image pixel coordinates to GPS coordinates at a fixed distance from camera.

    Args:
        image_points: array-like of shape (N,2) or (2,) with pixel coordinates [x, y]
        k_matrix: Camera intrinsics matrix (3x3)
        r_matrix: Camera rotation matrix (3x3)
        t_vector: Camera translation vector (3,) or (3,1)
        dist_coeffs: Distortion coefficients (or None)
        camera_gps: Camera GPS location [lat, lon, alt]
        distance_m: Distance from camera in meters (scalar)

    Returns:
        If input was a single point, returns [lat, lon, alt].
        If input was multiple points, returns an (N, 3) NumPy array of [lat, lon, alt].
    """
    pts = np.asarray(image_points, dtype=np.float32)
    single_input = False
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
        single_input = True
    elif pts.ndim == 2 and pts.shape[1] != 2:
        raise ValueError("image_points must have shape (N,2) or (2,)")

    # Ensure distortion is a valid array for OpenCV
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    else:
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float32)

    # Undistort points: input shape should be (N,1,2)
    undistorted = cv2.undistortPoints(pts.reshape(-1, 1, 2), k_matrix, dist_coeffs)
    # undistorted has shape (N,1,2)
    x_norm = undistorted[:, 0, 0]
    y_norm = undistorted[:, 0, 1]

    # Rays in camera coordinates (X=right, Y=down, Z=forward)
    rays_cam = np.column_stack((x_norm, y_norm, np.ones_like(x_norm)))  # (N,3)
    # Normalize direction vectors
    norms = np.linalg.norm(rays_cam, axis=1, keepdims=True)
    rays_cam_unit = rays_cam / norms

    # Transform rays to world (ENU) coordinates: R transforms ENU->cam, so R.T transforms cam->ENU
    rays_world_unit = (r_matrix.T @ rays_cam_unit.T).T  # (N,3)

    # Camera position in ENU (origin at camera_gps)
    t_vec = np.asarray(t_vector).reshape(-1)
    camera_enu = (-r_matrix.T @ t_vec)  # (3,)

    # Compute points at fixed distance along each ray
    point_enu = camera_enu[None, :] + distance_m * rays_world_unit  # (N,3)

    # Convert ENU to geodetic (lat, lon, alt). pymap3d supports array inputs.
    lats, lons, alts = pm.enu2geodetic(
        point_enu[:, 0], point_enu[:, 1], point_enu[:, 2],
        camera_gps[0], camera_gps[1], camera_gps[2]
    )

    result = np.column_stack((lats, lons, alts))
    if single_input:
        return result[0].tolist()
    return result


def gps_to_ecef(point_gps):

    # Define the transformation from Geodetic (EPSG:4979) to ECEF (EPSG:4978)
    # always_xy=True ensures the transformer expects (longitude, latitude) order
    transformer_geodetic_to_ecef = Transformer.from_crs(
        "epsg:4979", "epsg:4978", always_xy=True)

    # Perform the transformation
    # Note: pyproj expects longitude, latitude, altitude order for transform
    eX, eY, eZ = transformer_geodetic_to_ecef.transform(
        point_gps[1], point_gps[0], point_gps[2])

    # Stack the results into an Nx3 NumPy array
    ecef_point = (eX, eY, eZ)
    return ecef_point

def load_camera_parameters(path):
    with open(path, 'r') as f:
        camera_params = json.load(f)
        intrinsics = np.array(camera_params['intrinsics'], dtype=np.float32)
        distortion = np.array(camera_params['distortion'], dtype=np.float32).reshape(-1, 1)
        rvec = np.array(camera_params['rotation'], dtype=np.float32)
        tvec = np.array(camera_params['translation'], dtype=np.float32).reshape(-1, 1) #shape should be (3,1)
        origin_gps = [camera_params['origin_gps']['lat'], camera_params['origin_gps']['lon'], camera_params['origin_gps']['alt']]  # [lat, lon, alt]
    return intrinsics, distortion, rvec, tvec, origin_gps