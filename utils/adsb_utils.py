import pandas as pd
import numpy as np

def clean_numeric(s: pd.Series) -> pd.Series:
    # convert to string, strip whitespace
    s = s.astype(str).str.strip()
    # make empty strings into NaN
    s = s.replace({'': np.nan})
    # remove commas and any stray spaces
    s = s.str.replace(',', '', regex=False).str.replace(' ', '', regex=False)
    # finally, coerce to numeric
    return pd.to_numeric(s, errors='coerce')

def upsample_aircraft(group):
    """
    Upsample a single aircraft's data to 1-second intervals using linear interpolation.
    Only upsamples between consecutive points that are less than 5 minutes apart.
    Adds an 'isUpsampled' column to indicate interpolated vs original data.
    """
    # Sort by time
    group = group.sort_values('time').reset_index(drop=True)
    
    # Mark original data as not upsampled
    group['isUpsampled'] = False
    
    # If only one point, return as is
    if len(group) <= 1:
        return group
    
    # Build list of dataframes for segments to upsample
    segments = []
    
    for i in range(len(group)):
        # Always include the original point
        current_point = group.iloc[[i]].copy()
        segments.append(current_point)
        
        # Check if we should upsample to the next point
        if i < len(group) - 1:
            time_gap = (group.iloc[i + 1]['time'] - group.iloc[i]['time']).total_seconds()
            
            # Only upsample if gap is less than 5 minutes (300 seconds) and greater than 1 second
            if 1 < time_gap < 300:
                # Create 1-second intervals between current and next point
                start_time = group.iloc[i]['time']
                end_time = group.iloc[i + 1]['time']
                
                # Create time range (exclude start and end as they're already in original data)
                time_range = pd.date_range(start=start_time, end=end_time, freq='1s')[1:-1]
                
                if len(time_range) > 0:
                    # Create interpolation dataframe with just these two points
                    interp_df = group.iloc[[i, i + 1]].copy()
                    
                    # Create new dataframe with interpolated time points
                    interp_points = pd.DataFrame({'time': time_range})
                    
                    # Add all columns from the first point as base
                    for col in group.columns:
                        if col != 'time':
                            interp_points[col] = None
                    
                    # Concatenate the two boundary points with the empty interpolation points
                    interp_points = interp_points.dropna(axis=1, how='all')
                    temp_df = pd.concat([interp_df, interp_points], ignore_index=True)
                    temp_df = temp_df.sort_values('time').reset_index(drop=True)
                    
                    # Convert numeric columns to proper float dtype BEFORE interpolation
                    numeric_cols_to_interpolate = ['lon', 'lat', 'alt', 'alt_gnss', 'heading', 
                                                     'alt_meters', 'alt_gnss_meters', 'distance_m']
                    
                    for col in numeric_cols_to_interpolate:
                        if col in temp_df.columns:
                            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                            temp_df[col] = temp_df[col].interpolate(method='linear')
                    
                    # Forward fill categorical/string columns
                    categorical_cols = ['source_id', 'source', 'transponder_id', 'orig', 'dest', 
                                       'ident', 'aircraft_type', 'clock_datetime']
                    
                    for col in categorical_cols:
                        if col in temp_df.columns:
                            temp_df[col] = temp_df[col].ffill().infer_objects(copy=False)
                    
                    # Mark interpolated points as upsampled (the boundary points already have False)
                    temp_df.loc[temp_df['isUpsampled'].isna(), 'isUpsampled'] = True
                    
                    # Extract only the interpolated middle points (not the boundaries)
                    interpolated_segment = temp_df.iloc[1:-1].copy()
                    
                    segments.append(interpolated_segment)
    
    # Concatenate all segments
    result = pd.concat(segments, ignore_index=True)
    
    return result

def get_upsampled_df_for_day(df: pd.DataFrame, max_range_m: float = 50000) -> pd.DataFrame:
    """Load CSV, filter for date, and upsample."""
    df.columns = df.columns.str.strip()

    # Clean and convert numeric columns
    for col in ['alt_gnss_meters', 'distance_m']:
        if col in df.columns:
            df[col] = clean_numeric(df[col])
    
    df = df.dropna(subset=['alt_gnss_meters'])
    df['time'] = pd.to_datetime(df['time'])

        # convert to float
    df['alt_gnss_meters'] = df['alt_gnss_meters'].astype(float)
    df['distance_m'] = df['distance_m'].astype(float)

    # minimum altitude filter (8000 ft = 2438.4 m)
    df = df[(df['alt_gnss_meters'] > 2438.4)]

    print("Upsampling all aircraft...")
    # filter for 3 pm utc to 4 pm utc
    print(f"Processing {df['ident'].nunique()} unique aircraft...\n")

    # Group by ident and apply upsampling
    upsampled_groups = []
    for ident, group in df.groupby('ident'):
        upsampled = upsample_aircraft(group)
        upsampled_groups.append(upsampled)
        if len(upsampled_groups) % 10 == 0:
            print(f"Processed {len(upsampled_groups)} aircraft...")

    # Combine all upsampled data
    df_upsampled = pd.concat(upsampled_groups, ignore_index=True)
    df_upsampled = df_upsampled.sort_values(['ident', 'time']).reset_index(drop=True)
    df_upsampled= df_upsampled[df_upsampled["distance_m"] < max_range_m]

    # Check for NaN values in lat, lon, alt_gnss_meters
    nan_rows = df_upsampled[df_upsampled[['lat', 'lon', 'alt_gnss_meters']].isna().any(axis=1)]

    # Check for string values in lat, lon, alt_gnss_meters
    str_rows = df_upsampled[
        df_upsampled[['lat', 'lon', 'alt_gnss_meters']].applymap(lambda x: isinstance(x, str)).any(axis=1)
    ]

    if not nan_rows.empty:
        print(f"Warning: Found {len(nan_rows)} rows with NaN values in lat, lon, or alt_gnss_meters.")
        print(nan_rows[['ident', 'time', 'lat', 'lon', 'alt_gnss_meters']])

    if not str_rows.empty:
        print(f"Warning: Found {len(str_rows)} rows with string values in lat, lon, or alt_gnss_meters.")
        print(str_rows[['ident', 'time', 'lat', 'lon', 'alt_gnss_meters']])
    if not nan_rows.empty:
        print(f"Warning: Found {len(nan_rows)} rows with NaN values in lat, lon, or alt_gnss_meters.")
        print(nan_rows[['ident', 'time', 'lat', 'lon', 'alt_gnss_meters']])
    
    return df_upsampled



# right an adapter for data
# timestamp,icao,registration,flight,lat,lon,altitude_baro,alt_geom,ground_speed,track_degrees,vertical_rate,aircraft_type,description,operator,squawk,category,source_type

def read_adsblol_csv(path: str, origin_gps) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df['time'] = pd.to_datetime(df['timestamp'])
    df['alt_gnss_meters'] = df['alt_geom'] * 0.3048
    df['transponder_id'] = df['icao']
    # Some rows within the same flight may be missing 'flight' — propagate within each transponder+registration group
    df['flight'] = df.groupby(['icao', 'registration'])['flight'].transform(lambda s: s.ffill().bfill())
    df['ident'] = df['flight']
    df['distance_m'] = haversine_km(df['lat'], df['lon'], origin_gps[0], origin_gps[1]) * 1000
    return df

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))