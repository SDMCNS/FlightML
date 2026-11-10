import duckdb

con = duckdb.connect()
fir_src = 'data/parquet/periods/202510/fir.parquet'
con.execute(f"CREATE VIEW raw_fir AS SELECT * FROM '{fir_src}'")

fir_ids = ['EPWWFIR', 'LZBBFIR', 'LHCCFIR']
placeholders = ", ".join(["?"] * len(fir_ids))
sql = f"""
    SELECT "Airspace ID" as fir_id, "Latitude" as lat, "Longitude" as lon
    FROM raw_fir
    WHERE "Airspace ID" IN ({placeholders})
    ORDER BY "Airspace ID", "Sequence Number" ASC;
"""
df = con.execute(sql, fir_ids).df()
print("Found rows:", len(df))
for fid, group in df.groupby("fir_id"):
    print(f"FIR {fid}: {len(group)} vertices")
