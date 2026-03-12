import pandas as pd
import numpy as np
import cv2
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import utils.adsb_utils as adsb_utils
from utils.db_utils import ContrailDatabase
import utils.projection_utils as proj_utils
from utils.image_data_utils import get_image_data_uwisc
import  utils.detection_utils as detection_utils
from tqdm import tqdm
import math

def make_batch(ident, timestamp, gps_points, camera_name):
    """
    Create a batch of contrail detection data.

    Args:
        ident: Aircraft identifier
        timestamp: Timestamp of the observation
        gps_points: List of GPS points (latitude, longitude, altitude)
        camera_name: Name of the camera

    Returns:
        DataFrame containing the batch of contrail detection data
    """
    # Create a DataFrame from the GPS points
    df = pd.DataFrame(gps_points, columns=['lat', 'lon', 'altitude'])
    df['timestamp'] = timestamp
    df['ident'] = ident
    df['camera_name'] = camera_name
    return df

def run_contrail_pipeline_uwisc(date_str, camera_side='east'):
    year, month, day = date_str.split('-')

    adsb_csv_path = f"/Users/shrenikborad/pless/easy_adsb/data/madison_pings_{year}_{month}_{day}.csv"
    camera_params_path = f"./uwisc/{camera_side}/camera_params.json"
    base_dir = f'/Users/shrenikborad/Downloads/NNDL/images_uwisc/east/{date_str}/{camera_side}'
    camera_name = f"uwisc_{camera_side}"

    contrails_db = ContrailDatabase(f"contrails_uwisc.duckdb")

    intrinsics, distortion, rvec, tvec, origin_gps = proj_utils.load_camera_parameters(camera_params_path)

    # df = pd.read_csv(adsb_csv_path)
    df = adsb_utils.read_adsblol_csv(adsb_csv_path, origin_gps=origin_gps)
    from_dt = pd.to_datetime(f"{date_str} 06:00:00").tz_localize('America/Chicago').tz_convert('UTC')
    to_dt = pd.to_datetime(f"{date_str} 10:00:00").tz_localize('America/Chicago').tz_convert('UTC')
    df['time'] = pd.to_datetime(df['time'])
    df = df[(df['time'] >= from_dt) & (df['time'] < to_dt)]
    print(df.describe())
    df_upsampled = adsb_utils.get_upsampled_df_for_day(df, max_range_m=80000)


    # Load Camera Parameters

    image_x, image_y, cam_distance = proj_utils.gps_to_camxy_vasha_fixed(
        df_upsampled['lat'].values,
        df_upsampled['lon'].values,
        df_upsampled['alt_gnss_meters'].values,
        cam_k=intrinsics,
        cam_r=rvec,
        cam_t=tvec,
        camera_gps=origin_gps,
        distortion=distortion
    )

    df_upsampled['image_x'] = image_x
    df_upsampled['image_y'] = image_y
    df_upsampled['cam_distance'] = cam_distance


    image_df = get_image_data_uwisc(base_dir, date_str)
    image_df = image_df[(image_df['time'] >= from_dt) & (image_df['time'] < to_dt)]
    # Define video parameters
    output_path = f'output_video_{date_str}_{camera_name}_cleaned_background_removal_long_frangi.mp4'
    img_def = cv2.imread(f"{base_dir}/{image_df.iloc[0]['image_file']}")
    frame_height, frame_width = img_def.shape[0], img_def.shape[1]
    fps = 10  # frames per second

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    # csv with time and ident of flights that made contrails
    flights_with_contrails = []

    toProcess = image_df
    for idx, row in tqdm(toProcess.iterrows(), total=len(toProcess), desc="Processing images"):

        # img = cv2.imread(f"/Users/shrenikborad/Downloads/NNDL/images_uwisc/east/2025-10-01/east/{row['image_file']}")
        # if img is None:
        #     print(f"Could not read image {row['image_file']}")
        #     continue
        df_filtered = df_upsampled[df_upsampled['time'] == row['time']]
        curr_img_path = f"{base_dir}/{row['image_file']}"
        prev_img_path = None
        if idx > 0:
            prev_img_path = f"{base_dir}/{image_df.iloc[idx-1]['image_file']}"
        else:
            prev_img_path = curr_img_path
        img_o, rectangles, edge_data, edges_dict= detection_utils.process_image_with_canny_edges(f"{base_dir}/{row['image_file']}",
                                    prev_img_path=prev_img_path,
                                    timestamp=row['time'],
                                    df_filtered=df_filtered,
                                    df_upsampled=df_upsampled)
        if rectangles == None:
            continue
        # First pass: extract ROIs and draw rectangles
        roi_batch_imgs = []
        roi_batch_idents = []
        roi_images = {}
        for ident, (rect_poly, arrow, direction_info) in rectangles.items():
            # Draw rectangle outline
            color = (255, 0, 0)  # Blue for normal
            if edge_data[ident]['is_making_contrails']:
                color = (0, 255, 255)  # Yellow for contrails
                
            cv2.polylines(img_o, [rect_poly], isClosed=True, color=color, thickness=2)
        

            if edge_data[ident]['is_making_contrails']:
                row_to_append =  df_filtered[df_filtered['ident'] == ident]
                # save the cropped roi image of the contrail making aircraft
                x, y, w, h = edge_data[ident]['bbox']
                roi_img = img_o[y:y+h, x:x+w]
                flight_gps = row_to_append[['lat', 'lon', 'alt_gnss_meters']].values[0]
                # gps_points = detection_utils.convert_texture_to_gps_points(edges_dict[ident][0], 
                #                                                            flight_gps,
                #                                                            k_matrix=intrinsics, 
                #                                                            dist_coeffs=distortion,
                #                                                            r_matrix=rvec,
                #                                                            tvec=tvec,
                #                                                            gps_origin=origin_gps)
                # batch_df = make_batch(ident, row['time'], gps_points, camera_name)
                # contrails_db.insert_batch(batch_df)
                # if roi_img.size != 0:
                #     # show roi inline in plt
                #     roi_img_path = f"contrail_images/{date_str}/{camera_name}_contrail_{ident}_{row['time'].strftime('%Y%m%d_%H%M%S')}.jpg"
                #     # check if directory exists else create
                #     os.makedirs(os.path.dirname(roi_img_path), exist_ok=True)
                #     iswrite = cv2.imwrite(roi_img_path, roi_img)
                #     print(f"Written ROI image to {roi_img_path}: {iswrite}")
                #     # print(f"Saved contrail ROI image to {roi_img_path}")
                #     row_to_append = row_to_append.copy()
                #     row_to_append['contrail_image_path'] = roi_img_path
                lines = edge_data[ident]["lines"]
                longest_line = max(lines, key=lambda x: x[4]) if lines else None
                if longest_line:
                    # real world length
                    image_points = np.array([[longest_line[0], longest_line[1]], [longest_line[2], longest_line[3]]], dtype=np.float32)
                    flight_distance = detection_utils.get_flight_distance(flight_gps, origin_gps)
                    real_world_points = proj_utils.image_to_gps(image_points, k_matrix=intrinsics, dist_coeffs=distortion, r_matrix=rvec, t_vector=tvec, camera_gps=origin_gps, distance_m=flight_distance)
                    length_c = math.dist(real_world_points[0], real_world_points[1])
                    row_to_append["longest_contrail_length_meters"] = length_c
                # append the whole row with all the data
                flights_with_contrails.append(row_to_append)
                # for x1, y1, x2, y2, length in edge_data[ident]["lines"]:
                #     cv2.line(img_o, (x1, y1), (x2, y2), (0, 165, 255), 2)  # Orange lines
            # Draw arrow if available
            if arrow:
                tip, base = arrow
                # if tip and base:
                #     cv2.arrowedLine(img_output, base, tip, (255, 255, 0), 2, tipLength=0.3)
        img = img_o
        for ident, image_x, image_y in zip(df_filtered['ident'], df_filtered['image_x'], df_filtered['image_y']):
            if not np.isnan(image_x) and not np.isnan(image_y) and 0 <= image_x < img.shape[1] and 0 <= image_y < img.shape[0]:
                cv2.circle(img, (int(image_x), int(image_y)), 5, (0, 0, 255), -1)
                cv2.putText(img, str(ident), (int(image_x), int(image_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        video_writer.write(img)
    contrails_db.close()
    video_writer.release()
    print(f"Video saved to {output_path}")
    if len(flights_with_contrails) > 0:
        df_contrails = pd.concat(flights_with_contrails, ignore_index=True)
        df_contrails.to_csv(f'flights_with_contrails_{camera_name}_{date_str}.csv', index=False)
        print(f"CSV of flights with contrails saved to flights_with_contrails_{camera_name}_{date_str}.csv")


def main():
    for day in range(9, 10):
        date_str = f"2025-01-{day:02d}"
        run_contrail_pipeline_uwisc(date_str,camera_side="east")
if __name__ == "__main__":
    main()