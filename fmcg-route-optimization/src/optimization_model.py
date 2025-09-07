import pandas as pd
import numpy as np

def nearest_neighbor_solution(distance_matrix, orders_df, vehicles_df, depots_df):
    """
    Simple nearest neighbor heuristic for vehicle routing with emission calculation.
    
    Parameters:
    -----------
    distance_matrix : pd.DataFrame
        Square DataFrame with distances (index = locations, columns = locations).
    orders_df : pd.DataFrame
        Must contain ['CustomerID', 'Demand'].
    vehicles_df : pd.DataFrame
        Must contain ['VehicleID', 'Capacity'].
    depots_df : pd.DataFrame
        Must contain ['DepotID'].
    
    Returns:
    --------
    solution : list of dict
        Routes for each vehicle.
    total_distance : float
        Total distance traveled.
    total_emissions : float
        Total emissions (distance * 0.2 as placeholder factor).
    """

    solution = []
    total_distance = 0.0
    total_emissions = 0.0

    # --- Align orders with distance_matrix ---
    valid_customers = orders_df[orders_df['customer_id'].isin(distance_matrix.index)]
    missing_customers = set(orders_df['customer_id']) - set(distance_matrix.index)
    if missing_customers:
        print(f"⚠️ Skipping {len(missing_customers)} customers not in distance matrix: {list(missing_customers)[:5]}...")

    for _, vehicle in vehicles_df.iterrows():
        capacity = vehicle["capacity_units"]
        load = 0
        route = []

        # start from first depot (you can extend to multi-depot later)
        current = depots_df.iloc[0]["depot_id"]

        unvisited = set(valid_customers["customer_id"].tolist())

        while unvisited:
            nearest = None
            nearest_dist = np.inf

            for cust in unvisited:
                demand = valid_customers.loc[valid_customers["customer_id"] == cust, "demand_units"].values[0]

                if load + demand > capacity:
                    continue

                # --- Safe lookup ---
                if cust not in distance_matrix.index or current not in distance_matrix.index:
                    continue

                try:
                    dist = distance_matrix.loc[current, cust]
                except KeyError:
                    continue

                if dist < nearest_dist:
                    nearest = cust
                    nearest_dist = dist

            if nearest is None:  # no feasible customer left
                break

            # assign customer
            route.append(nearest)
            load += valid_customers.loc[valid_customers["customer_id"] == nearest, "demand_units"].values[0]
            total_distance += nearest_dist
            total_emissions += nearest_dist * 0.2  # example factor
            current = nearest
            unvisited.remove(nearest)

        # return to depot
        if current != depots_df.iloc[0]["depot_id"] and current in distance_matrix.index:
            depot = depots_df.iloc[0]["depot_id"]
            if depot in distance_matrix.index:
                total_distance += distance_matrix.loc[current, depot]
                total_emissions += distance_matrix.loc[current, depot] * 0.2
            route.append(depot)

        solution.append({
            "VehicleID": vehicle["vehicle_id"],
            "Route": route,
            "Load": load
        })

    return solution, total_distance, total_emissions

