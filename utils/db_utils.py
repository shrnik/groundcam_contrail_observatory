"""
DuckDB utilities for storing and querying aircraft texture/contrail data.

This module provides functions to efficiently store large amounts of aircraft
texture detection data (edge detection, contrails) with associated metadata
including timestamp, aircraft ident, GPS coordinates, and altitude.
"""

import duckdb
import pandas as pd
from typing import Optional, Union, List, Dict, Any
import json


class ContrailDatabase:
    """
    Manages DuckDB database for storing aircraft texture and contrail detection data.

    Schema:
        - timestamp: TIMESTAMP (when the observation was made)
        - ident: VARCHAR (aircraft identifier/callsign)
        - lat: DOUBLE (latitude in degrees)
        - lon: DOUBLE (longitude in degrees)
        - altitude: DOUBLE (altitude in meters)
        - camera_name: VARCHAR (camera identifier)
    """

    def __init__(self, db_path: str = "contrails.duckdb"):
        """
        Initialize database connection.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Main texture data table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS texture_data (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                ident VARCHAR NOT NULL,
                camera_name VARCHAR,
                lat DOUBLE NOT NULL,
                lon DOUBLE NOT NULL,
                altitude DOUBLE NOT NULL,
            )
        """)

        # Create indexes for efficient querying
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON texture_data(timestamp)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ident
            ON texture_data(ident)
        """)

        # self.conn.execute("""
        #     CREATE INDEX IF NOT EXISTS idx_contrails
        #     ON texture_data(is_making_contrails)
        # """)

        # self.conn.execute("""
        #     CREATE INDEX IF NOT EXISTS idx_location
        #     ON texture_data(lat, lon)
        # """)


    def insert_batch(self, data: pd.DataFrame) -> int:
        """
        Insert a batch of texture data records.

        Args:
            data: DataFrame with columns matching the schema

        Returns:
            Number of rows inserted
        """
        # Ensure required columns exist
        required_cols = ['timestamp', 'ident', 'lat', 'lon', 'altitude']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Prepare data for insertion
        insert_data = data.copy()

        # Convert timestamp to proper format if needed
        if 'timestamp' in insert_data.columns:
            insert_data['timestamp'] = pd.to_datetime(insert_data['timestamp'])
        elif 'time' in insert_data.columns:
            insert_data['timestamp'] = pd.to_datetime(insert_data['time'])

        # Add auto-incrementing ID
        max_id = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM texture_data").fetchone()[0]
        insert_data['id'] = range(max_id + 1, max_id + len(insert_data) + 1)

        # Select only columns that exist in the table
        table_cols = self.get_column_names()
        insert_data = insert_data[table_cols]

        # Insert using DuckDB's efficient batch insert
        self.conn.execute("""
            INSERT INTO texture_data
            SELECT * FROM insert_data
        """)

        return len(insert_data)

    def insert_from_pipeline(self,
                            flights_df: pd.DataFrame,
                            edge_data: Dict,
                            camera_name: str,
                            image_path: str) -> int:
        """
        Insert texture data from the contrail detection pipeline.

        Args:
            flights_df: DataFrame with flight data (timestamp, ident, lat, lon, altitude)
            edge_data: Dictionary with edge detection results per ident
            camera_name: Name of the camera
            image_path: Path to the source image

        Returns:
            Number of rows inserted
        """
        records = []

        for _, row in flights_df.iterrows():
            ident = row['ident']

            # Get edge data if available
            if ident in edge_data:
                edge_info = edge_data[ident]
                x, y, w, h = edge_info.get('bbox', (None, None, None, None))

                record = {
                    'timestamp': row.get('time', row.get('timestamp')),
                    'ident': ident,
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'altitude': row.get('alt_gnss_meters', row.get('altitude')),
                    'camera_name': camera_name,
                    'image_path': image_path,
                    'contrail_image_path': row.get('contrail_image_path'),
                    'image_x': row.get('image_x'),
                    'image_y': row.get('image_y'),
                    'cam_distance': row.get('cam_distance')
                }
                records.append(record)

        if records:
            df = pd.DataFrame(records)
            return self.insert_batch(df)
        return 0

    def get_aircraft_stats(self, ident: str) -> Dict:
        """
        Get statistics for a specific aircraft.

        Args:
            ident: Aircraft identifier

        Returns:
            Dictionary with aircraft statistics
        """
        query = """
            SELECT
                COUNT(*) as total_observations,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen,
                AVG(altitude) as avg_altitude,
                MIN(altitude) as min_altitude,
                MAX(altitude) as max_altitude,
            FROM texture_data
            WHERE ident = ?
        """
        result = self.conn.execute(query, [ident]).fetchone()

        return {
            'total_observations': result[0],
            'first_seen': result[2],
            'last_seen': result[3],
            'avg_altitude': result[4],
            'min_altitude': result[5],
            'max_altitude': result[6],
        }

    def get_all_df(self) -> pd.DataFrame:
        """
        Get all data from the texture_data table.

        Returns:
            DataFrame containing all texture data
        """
        query = "SELECT * FROM texture_data"
        return self.conn.execute(query).df()

    def export_for_visualization(self,
                                 output_path: str,
                                 format: str = 'parquet') -> None:
        """
        Export data for visualization tools.

        Args:
            output_path: Path to save exported data
            contrails_only: If True, only export contrail detections
            format: Output format ('parquet', 'csv', 'json')
        """
        query = "SELECT * FROM texture_data"

        df = self.conn.execute(query).df()

        if format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_column_names(self) -> List[str]:
        """Get list of column names in the texture_data table."""
        result = self.conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'texture_data'
        """).fetchall()
        return [row[0] for row in result]

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for quick usage

def create_database(db_path: str = "contrails.duckdb") -> ContrailDatabase:
    """
    Create and return a new ContrailDatabase instance.

    Args:
        db_path: Path to DuckDB database file

    Returns:
        ContrailDatabase instance
    """
    return ContrailDatabase(db_path)


def insert_pipeline_results(db_path: str,
                           flights_df: pd.DataFrame,
                           edge_data: Dict,
                           camera_name: str,
                           image_path: str) -> int:
    """
    Convenience function to insert results from contrail pipeline.

    Args:
        db_path: Path to DuckDB database file
        flights_df: DataFrame with flight data
        edge_data: Dictionary with edge detection results
        camera_name: Name of the camera
        image_path: Path to the source image

    Returns:
        Number of rows inserted
    """
    with ContrailDatabase(db_path) as db:
        return db.insert_from_pipeline(flights_df, edge_data, camera_name, image_path)



def duckdb_to_timestamped_geojson(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert DuckDB table data to timestamped GeoJSON format.
    
    Args:
        connection: DuckDB connection object
        table_name: Name of the table containing the data
        query: Optional custom SQL query. If None, selects all from table_name
    
    Returns:
        Dict representing a GeoJSON FeatureCollection with timestamped features
    """
    
    # Use custom query or default to selecting all from table
    if query is None:
        query = f"SELECT timestamp, lat, lon, altitude, ident FROM {table_name} ORDER BY timestamp"
    
    # Execute query and fetch results
    result = connection.execute(query).fetchall()
    
    # Get column names
    columns = [desc[0] for desc in connection.description]
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # Convert each row to a GeoJSON Feature
    for row in result:
        # Create a dictionary from row data
        row_dict = dict(zip(columns, row))
        
        # Create the GeoJSON Feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    row_dict['lon'],  # longitude comes first in GeoJSON
                    row_dict['lat'],
                    row_dict.get('altitude', 0)  # altitude is optional third coordinate
                ]
            },
            "properties": {
                "timestamp": str(row_dict['timestamp']),  # Convert timestamp to string for JSON serialization
                "ident": row_dict['ident'],
                "altitude": row_dict.get('altitude')
            }
        }
        
        geojson["features"].append(feature)
    
    return geojson


def save_geojson(geojson: Dict[str, Any], filename: str) -> None:
    """
    Save GeoJSON dictionary to a file.
    
    Args:
        geojson: GeoJSON dictionary
        filename: Output filename
    """
    with open(filename, 'w') as f:
        json.dump(geojson, f, indent=2)

