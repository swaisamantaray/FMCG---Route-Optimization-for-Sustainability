import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance (km) between two lat/lon points."""
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def build_distance_matrix(nodes_df):
    """
    Create pairwise distance & time (minutes) between depots/customers.
    nodes_df must contain ['id','lat','lon'].
    """
    distances = []
    for i, row1 in nodes_df.iterrows():
        for j, row2 in nodes_df.iterrows():
            if row1["id"] != row2["id"]:
                dist_km = haversine_distance(row1["lat"], row1["lon"], row2["lat"], row2["lon"])
                time_min = dist_km / 30 * 60   # assuming avg 30 km/h
                distances.append([row1["id"], row2["id"], dist_km, time_min])
    return pd.DataFrame(distances, columns=["from_id","to_id","distance_km","time_min"])
