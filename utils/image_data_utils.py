
import os
from datetime import datetime
import pandas as pd

def get_image_data_mit(path: str, date_str: str):
    """
    Extract image data from the DataFrame.
    """
    year, month, day = date_str.split('-')
    image_files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
    # extract time from image name
    # keep files with frame_20250915_192426.jpg format
    image_files = [f for f in image_files if f.startswith('frame_20250915_') and f.endswith('6.jpg')]
    image_times = [datetime.strptime(f.split('_')[1] + f.split('_')[2].split('.')[0], '%Y%m%d%H%M%S') for f in image_files]
    # create a dataframe of image times and image paths
    image_df = pd.DataFrame({'time': image_times, 'image_file': image_files})
    image_df = image_df.reset_index(drop=True)
    # sort by time
    image_df = image_df.sort_values(by='time').reset_index(drop=True)
    # date is 2025-10-01 and convert to utc
    image_df['time'] = image_df['time'].apply(lambda x: x.replace(year=int(year), month=int(month), day=int(day)))
    image_df['time'] = image_df['time'].dt.tz_localize('America/New_York').dt.tz_convert('UTC')

    print(f"Total images between 3 pm and 4 pm utc: {len(image_df)}")
    print(image_df.head())
    return image_df

def get_image_data_arizona(path: str):
    """
    Extract image data from the DataFrame.
    """
    image_files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
    # extract time from image name
    # support both frame_YYYYMMDD_HHMMSS.jpg and YYYYMMDDHHMMSS.jpg formats
    image_files = [f for f in image_files if (f.startswith('frame_') or (len(f) == 18 and f[:14].isdigit())) and f.endswith('.jpg')]
    def parse_arizona_time(f):
        if f.startswith('frame_'):
            return datetime.strptime(f.split('_')[1] + f.split('_')[2].split('.')[0], '%Y%m%d%H%M%S')
        else:
            return datetime.strptime(f.split('.')[0], '%Y%m%d%H%M%S')
    image_times = [parse_arizona_time(f) for f in image_files]
    # create a dataframe of image times and image paths
    image_df = pd.DataFrame({'time': image_times, 'image_file': image_files})
    image_df = image_df.reset_index(drop=True)
    # sort by time
    image_df = image_df.sort_values(by='time').reset_index(drop=True)
    # date is 2025-10-01 and convert to utc
    image_df['time'] = image_df['time']
    image_df['time'] = image_df['time'].dt.tz_localize('America/Phoenix').dt.tz_convert('UTC')

    print(f"Total {len(image_df)}")
    print(image_df.head())
    return image_df

def get_image_data_uwisc(path: str, date_str: str):
    """
    Extract image data from the DataFrame.
    """
    year, month, day = date_str.split('-')
    image_files = sorted([f for f in os.listdir(path) if f.endswith('.jpg')])
    # image name format: 00_00_08.trig+00.jpg
    # extract time from image name
    print(f"Total images found: {len(image_files)}")
    image_times = [datetime.strptime(f.split('.')[0], '%H_%M_%S') for f in image_files]
    # create a dataframe of image times and image paths
    image_df = pd.DataFrame({'time': image_times, 'image_file': image_files})
    image_df = image_df[(image_df['time'] >= datetime.strptime('08:00:00', '%H:%M:%S')) & (image_df['time'] < datetime.strptime('19:00:00', '%H:%M:%S'))]
    image_df = image_df.reset_index(drop=True)
    # date is 2025-10-01 and convert to utc
    image_df['time'] = image_df['time'].apply(lambda x: x.replace(year=int(year), month=int(month), day=int(day)))
    image_df['time'] = image_df['time'].dt.tz_localize('America/Chicago').dt.tz_convert('UTC')

    print(f"Total images between 3 pm and 4 pm utc: {len(image_df)}")
    print(image_df.head())
    return image_df
