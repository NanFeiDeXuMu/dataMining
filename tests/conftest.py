"""Small synthetic edge cases for tests only; never substituted for real app data."""

from pathlib import Path

import pandas as pd
import pytest

from urbanpulse.download import sha256, write_json
from urbanpulse.pipeline import build


def make_raw(root: Path):
    folder = root / "raw" / "2025-01"
    folder.mkdir(parents=True)
    base = {"VendorID": 1, "tpep_pickup_datetime": pd.Timestamp("2025-01-01 12:00"),
            "tpep_dropoff_datetime": pd.Timestamp("2025-01-01 12:10"), "trip_distance": 2.0,
            "PULocationID": 1.0, "DOLocationID": 2.0, "fare_amount": 15.0,
            "total_amount": 20.0, "passenger_count": 1.0, "payment_type": 1}
    rows = [base.copy() for _ in range(22)]
    rows[2].update(total_amount=-20.0, fare_amount=-15.0)
    rows[3]["passenger_count"] = None
    rows[4]["total_amount"] = 350.0
    rows[5]["trip_distance"] = 0.0
    rows[6]["tpep_pickup_datetime"] = None
    rows[7]["tpep_dropoff_datetime"] = base["tpep_pickup_datetime"]
    rows[8].update(tpep_pickup_datetime=pd.Timestamp("2024-12-31 12:00"), tpep_dropoff_datetime=pd.Timestamp("2024-12-31 12:10"))
    rows[9]["trip_distance"] = -1.0
    rows[10]["total_amount"] = float("inf")
    rows[11]["PULocationID"] = 265
    rows[12]["DOLocationID"] = 999
    rows[13].update(tpep_pickup_datetime=pd.Timestamp("2025-01-04 12:00"), tpep_dropoff_datetime=pd.Timestamp("2025-01-04 12:10"))
    rows[14]["tpep_dropoff_datetime"] = pd.Timestamp("2025-01-01 17:00")
    rows[15]["tpep_dropoff_datetime"] = pd.Timestamp("2025-01-01 12:00:06")
    rows[16]["trip_distance"] = None
    rows[17]["PULocationID"] = 1.4
    rows[18].update(tpep_pickup_datetime=pd.Timestamp("2025-02-01 00:00"), tpep_dropoff_datetime=pd.Timestamp("2025-02-01 00:10"))
    rows[19].update(tpep_pickup_datetime=pd.Timestamp("2025-01-31 23:59:59"), tpep_dropoff_datetime=pd.Timestamp("2025-02-01 00:09:59"))
    rows[20]["fare_amount"] = float("nan")
    rows[21].update(trip_distance=0.0, tpep_dropoff_datetime=pd.Timestamp("2025-01-01 12:00:00.100000"))
    trip_path = folder / "yellow_tripdata_2025-01.parquet"
    pd.DataFrame(rows).to_parquet(trip_path, index=False)
    zone_path = folder / "taxi_zone_lookup.csv"
    pd.DataFrame({"LocationID": [1, 2, 3, 264, 265], "Borough": ["Test"] * 5,
                  "Zone": ["Test A", "Test B", "Empty zone", "Unknown", "N/A"]}).to_csv(zone_path, index=False)
    write_json(folder / "source_manifest.json", {"month": "2025-01", "fixture": True,
               "files": [{"file": p.name, "sha256": sha256(p), "url": "synthetic-test-fixture"} for p in (trip_path, zone_path)]})
    return root


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    root = make_raw(tmp_path_factory.mktemp("urbanpulse-fixture"))
    run = build("2025-01", root)
    return root, run


@pytest.fixture
def raw_dataset(tmp_path):
    return make_raw(tmp_path)
