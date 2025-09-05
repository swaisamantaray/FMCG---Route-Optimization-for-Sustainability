import os
import numpy as np
import pandas as pd
from math import radians, cos

# -----------------------------
# Parameters
# -----------------------------
RAW_CUSTOMERS = 50000
RAW_DEPOTS = 15
RAW_VEHICLES = 3000
RAW_ORDERS = 120000

SAMPLE_CUSTOMERS = 100
SAMPLE_DEPOTS = 2
SAMPLE_VEHICLES = 10
SAMPLE_ORDERS = 200

np.random.seed(123)

BASE_LAT, BASE_LON = 19.0760, 72.8777  # Mumbai

def random_location(lat, lon, spread_km=100):
    """Generate random lat/lon within a given km spread."""
    dlat = np.random.uniform(-spread_km, spread_km) / 110.574
    dlon = np.random.uniform(-spread_km, spread_km) / (111.320 * cos(radians(lat)))
    return lat + dlat, lon + dlon

def generate_dataset(n_customers, n_depots, n_vehicles, n_orders, out_dir):
    """Generate synthetic dataset."""
    os.makedirs(out_dir, exist_ok=True)

    # Depots
    depots = []
    for i in range(n_depots):
        lat, lon = random_location(BASE_LAT, BASE_LON, spread_km=50)
        depots.append([f"D{i}", f"Depot {i}", lat, lon, "06:00", "22:00"])
    depots_df = pd.DataFrame(depots, columns=["depot_id","name","lat","lon","open_time","close_time"])
    depots_df.to_csv(os.path.join(out_dir, "depots.csv"), index=False)

    # Customers
    customers = []
    for i in range(1, n_customers+1):
        lat, lon = random_location(BASE_LAT, BASE_LON, spread_km=100)
        customers.append([f"C{i}", f"Customer {i}", lat, lon, f"Address {i}"])
    customers_df = pd.DataFrame(customers, columns=["customer_id","name","lat","lon","address"])
    customers_df.to_csv(os.path.join(out_dir, "customers.csv"), index=False)

    # Orders
    orders = []
    for i in range(n_orders):
        cust_id = np.random.choice(customers_df["customer_id"].values)
        demand = np.random.randint(5, 500)
        temp_zone = np.random.choice(["ambient","chilled","frozen"], p=[0.55,0.3,0.15])
        priority = np.random.choice([1,2,3], p=[0.2,0.6,0.2])
        service_minutes = np.random.randint(5, 25)
        start = np.random.randint(6*60, 20*60)
        end = start + np.random.randint(60, 300)
        orders.append([f"O{i}", cust_id, demand, temp_zone, priority, service_minutes, start, end])
    orders_df = pd.DataFrame(orders, columns=["order_id","customer_id","demand_units","temp_zone","priority","service_minutes","tw_start_min","tw_end_min"])
    orders_df.to_csv(os.path.join(out_dir, "orders.csv"), index=False)

    # Vehicles
    vehicles = []
    for i in range(n_vehicles):
        cap = np.random.randint(800, 5000)
        reefer = np.random.choice([0,1], p=[0.5,0.5])
        co2_factor = round(np.random.uniform(0.6, 1.5), 2)
        vehicles.append([f"V{i}", cap, reefer, 3.0 if reefer else 0.0, 80, 7.5, co2_factor])
    vehicles_df = pd.DataFrame(vehicles, columns=["vehicle_id","capacity_units","reefer_supported","reefer_power_kw","fixed_cost","per_km_cost","co2_factor_kg_per_km"])
    vehicles_df.to_csv(os.path.join(out_dir, "vehicles.csv"), index=False)

    # README
    readme = f"""SYNTHETIC FMCG DATASET ({'RAW' if 'raw' in out_dir else 'SAMPLE'})

Files:
- depots.csv ({len(depots_df)} depots)
- customers.csv ({len(customers_df)} customers)
- orders.csv ({len(orders_df)} orders)
- vehicles.csv ({len(vehicles_df)} vehicles)

Details:
- Orders linked to customers with time windows, demand, and temp_zone
- Vehicles with reefer info, capacity, CO₂ factors
- Depots with open/close times

Counts:
- Depots: {len(depots_df)}
- Customers: {len(customers_df)}
- Orders: {len(orders_df)}
- Vehicles: {len(vehicles_df)}
"""
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme)

    print(f"✅ Dataset generated in: {out_dir}")

# -----------------------------
# Create Raw & Sample Datasets
# -----------------------------
base_dir = "fmcg-route-optimization/data"

generate_dataset(RAW_CUSTOMERS, RAW_DEPOTS, RAW_VEHICLES, RAW_ORDERS, os.path.join(base_dir, "raw"))
generate_dataset(SAMPLE_CUSTOMERS, SAMPLE_DEPOTS, SAMPLE_VEHICLES, SAMPLE_ORDERS, os.path.join(base_dir, "sample"))

# -----------------------------
# Extra sample files
# -----------------------------
distances = pd.DataFrame({
    "from_id": ["D0","D0","C1","C2"],
    "to_id": ["C1","C2","C2","C3"],
    "distance_km": [10.5, 15.2, 5.3, 7.8],
    "time_min": [25, 40, 12, 18]
})
distances.to_csv(os.path.join(base_dir, "sample", "distances.csv"), index=False)

emissions = pd.DataFrame({
    "vehicle_id": [f"V{i}" for i in range(5)],
    "co2_factor_kg_per_km": [0.9, 1.1, 0.8, 1.3, 1.0]
})
emissions.to_csv(os.path.join(base_dir, "sample", "emissions.csv"), index=False)

print("✅ Extra sample files (distances.csv, emissions.csv) created.")
