# src/optimization_model.py
"""
Robust nearest-neighbor baseline for FMCG routing (v2).
Returns 4 outputs: (solution_list, total_distance_km, total_emissions_kg, unserved_customers_list)

This file includes a small demo runnable with `python optimization_model.py` for quick sanity checks.
"""

from typing import List, Dict, Tuple, Any
import pandas as pd
import numpy as np

def _safe_to_str_index(mat: pd.DataFrame) -> pd.DataFrame:
    mat = mat.copy()
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    return mat

def _pivot_long_to_matrix(df_long: pd.DataFrame) -> pd.DataFrame:
    # try common case: columns named from_id,to_id,distance_km (case-insensitive)
    cols = {c.lower(): c for c in df_long.columns}
    from_col = cols.get("from_id") or cols.get("from")
    to_col = cols.get("to_id") or cols.get("to")
    dist_col = cols.get("distance_km") or cols.get("distance") or cols.get("dist")
    if from_col and to_col and dist_col:
        mat = df_long.rename(columns={from_col: "from_id", to_col: "to_id", dist_col: "distance_km"})
        mat_pivot = mat.pivot(index="from_id", columns="to_id", values="distance_km")
        return _safe_to_str_index(mat_pivot)
    else:
        raise ValueError("long-form distance DataFrame must contain from_id,to_id,distance_km (case-insensitive)")

def _to_distance_matrix(distance_input: pd.DataFrame) -> pd.DataFrame:
    """Accept either a square matrix (index & columns are ids) or long-form DataFrame."""
    if not isinstance(distance_input, pd.DataFrame):
        raise ValueError("distance_input must be a pandas DataFrame")
    # If it already looks like a square matrix (index and columns overlap)
    idx = set(map(str, distance_input.index.astype(str)))
    cols = set(map(str, distance_input.columns.astype(str)))
    if idx & cols:
        return _safe_to_str_index(distance_input)
    # else try to pivot long-form
    return _pivot_long_to_matrix(distance_input)

def nearest_neighbor_solution_v2(
    distance_input: pd.DataFrame,
    orders_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    depots_df: pd.DataFrame,
    demand_col: str = "demand_units",
    vehicle_capacity_col: str = "capacity_units",
    emission_col: str = "co2_factor_kg_per_km",
    max_customers_per_vehicle: int | None = None
) -> Tuple[List[Dict[str, Any]], float, float, List[str]]:
    """
    Nearest-neighbor baseline that is defensive about missing IDs.

    Returns:
      - solution: list of dicts {"vehicle_id", "route" (list of ids), "load", "assigned_count"}
      - total_distance: float (km)
      - total_emissions: float (kg CO2)
      - unserved_customers: list of customer ids
    """

    # Normalize columns and types
    orders = orders_df.copy()
    vehicles = vehicles_df.copy()
    depots = depots_df.copy()

    # normalize column names (best-effort)
    if demand_col not in orders.columns:
        # try common alternatives
        for alt in ["demand", "demand_units", "Demand"]:
            if alt in orders.columns:
                orders.rename(columns={alt: demand_col}, inplace=True)
                break
    if vehicle_capacity_col not in vehicles.columns:
        for alt in ["capacity", "capacity_units", "Capacity"]:
            if alt in vehicles.columns:
                vehicles.rename(columns={alt: vehicle_capacity_col}, inplace=True)
                break

    # cast ids to str
    if "customer_id" in orders.columns:
        orders["customer_id"] = orders["customer_id"].astype(str)
    else:
        raise ValueError("orders_df must contain 'customer_id' column")
    if "vehicle_id" in vehicles.columns:
        vehicles["vehicle_id"] = vehicles["vehicle_id"].astype(str)
    else:
        raise ValueError("vehicles_df must contain 'vehicle_id' column")
    if "depot_id" in depots.columns:
        depots["depot_id"] = depots["depot_id"].astype(str)
    else:
        raise ValueError("depots_df must contain 'depot_id' column")

    # Convert distance_input to matrix form
    dist_mat = _to_distance_matrix(distance_input)

    # demand aggregation
    demand_by_customer = orders.groupby("customer_id")[demand_col].sum().to_dict()
    all_customers = list(demand_by_customer.keys())

    # find which customers exist in distance matrix (either as row or column)
    dist_nodes = set(dist_mat.index) | set(dist_mat.columns)
    valid_customers = [c for c in all_customers if c in dist_nodes]
    missing_customers = [c for c in all_customers if c not in dist_nodes]

    if len(missing_customers) > 0:
        # warn but continue
        print(f"⚠️ {len(missing_customers)} customers not present in distance matrix (they will be unserved). Sample: {missing_customers[:6]}")

    unvisited = set(valid_customers)

    # prepare vehicles
    if vehicle_capacity_col not in vehicles.columns:
        raise ValueError(f"vehicles_df must include capacity column named '{vehicle_capacity_col}'")
    vehicles[vehicle_capacity_col] = vehicles[vehicle_capacity_col].astype(float)
    if emission_col not in vehicles.columns:
        # if emissions not present, default to None (we'll skip emission calc)
        vehicles[emission_col] = np.nan

    # choose starting depot
    depot0 = str(depots.iloc[0]["depot_id"])

    solution = []
    total_distance = 0.0
    total_emissions = 0.0

    # helper to safely get a distance
    def safe_dist(a: str, b: str) -> float | None:
        a = str(a); b = str(b)
        # direct
        if (a in dist_mat.index) and (b in dist_mat.columns):
            val = dist_mat.at[a, b]
            if pd.notna(val):
                return float(val)
        # reverse
        if (b in dist_mat.index) and (a in dist_mat.columns):
            val = dist_mat.at[b, a]
            if pd.notna(val):
                return float(val)
        return None

    # iterate vehicles and greedily assign customers
    for _, veh in vehicles.iterrows():
        vid = str(veh["vehicle_id"])
        capacity = float(veh[vehicle_capacity_col])
        emission_factor = float(veh[emission_col]) if pd.notna(veh[emission_col]) else None

        current = depot0
        load = 0.0
        route = []
        assigned_count = 0

        while unvisited:
            nearest = None
            nearest_d = float("inf")
            for cust in list(unvisited):
                # check demand and capacity
                cust_demand = float(demand_by_customer.get(cust, 0.0))
                if load + cust_demand > capacity:
                    continue
                d = safe_dist(current, cust)
                if d is None:
                    continue
                if d < nearest_d:
                    nearest = cust
                    nearest_d = d

            if nearest is None:
                break

            # assign
            route.append(nearest)
            load += float(demand_by_customer.get(nearest, 0.0))
            total_distance += nearest_d
            if emission_factor is not None:
                total_emissions += nearest_d * emission_factor
            current = nearest
            unvisited.remove(nearest)
            assigned_count += 1

            if max_customers_per_vehicle is not None and assigned_count >= max_customers_per_vehicle:
                break

        # return to depot if possible
        back = safe_dist(current, depot0)
        if back is not None:
            total_distance += back
            if emission_factor is not None:
                total_emissions += back * emission_factor
            route.append(depot0)

        solution.append({
            "vehicle_id": vid,
            "route": route,
            "load": load,
            "assigned_count": assigned_count
        })

    # unserved includes those missing from distance matrix + those still in unvisited
    unserved_customers = list(set(missing_customers) | set(unvisited))

    return solution, total_distance, total_emissions, unserved_customers


# Simple demo to check function works when run directly
if __name__ == "__main__":
    # tiny example
    dist = pd.DataFrame(
        [[0.0, 5.0, 10.0],
         [5.0, 0.0, 7.0],
         [10.0, 7.0, 0.0]],
        index=["D0", "C1", "C2"],
        columns=["D0", "C1", "C2"]
    )
    orders = pd.DataFrame({"customer_id": ["C1", "C2"], "demand_units": [10, 20]})
    vehicles = pd.DataFrame({"vehicle_id": ["V0"], "capacity_units": [100], "co2_factor_kg_per_km": [0.9]})
    depots = pd.DataFrame({"depot_id": ["D0"]})

    sol, td, te, unserved = nearest_neighbor_solution_v2(dist, orders, vehicles, depots)
    print("Demo solution:", sol)
    print("Total distance:", td, "Total emissions:", te, "Unserved:", unserved)
