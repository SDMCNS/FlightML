from flightml.data.dataset import FlightDataset
import time

ds = FlightDataset()
t0 = time.time()

# Test searching with multiple filters
query = """
    SELECT 
        ectrl_id,
        adep,
        ades,
        CAST(actual_off_block_time AS VARCHAR) as off_block_time,
        EXTRACT(HOUR FROM actual_off_block_time) as dep_hour,
        ac_operator,
        ac_type,
        ac_reg,
        requested_fl,
        actual_distance_nm,
        arr_delay_min,
        dep_delay_min
    FROM flights
    WHERE adep = 'EGLL' AND ades = 'KORD'
    ORDER BY actual_off_block_time ASC
    LIMIT 10
"""
df = ds.query(query)
print(f"Query returned {len(df)} rows in {(time.time() - t0)*1000:.1f} ms")
print(df)

