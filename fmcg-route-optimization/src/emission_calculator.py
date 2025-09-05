import pandas as pd

def compute_emissions(distance_df, vehicles_df, vehicle_id):
    """
    Compute emissions for a given vehicle traveling distances.
    - distance_df: DataFrame with ['from_id','to_id','distance_km']
    - vehicles_df: vehicle data with ['vehicle_id','co2_factor_kg_per_km']
    """
    factor = vehicles_df.loc[vehicles_df["vehicle_id"] == vehicle_id, "co2_factor_kg_per_km"].values[0]
    emissions = distance_df.copy()
    emissions["vehicle_id"] = vehicle_id
    emissions["co2_emission_kg"] = emissions["distance_km"] * factor
    return emissions
