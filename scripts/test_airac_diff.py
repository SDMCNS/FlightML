import duckdb

con = duckdb.connect()

# Compare Route_2509 and Route_2510
res = con.execute("""
    WITH r1 AS (
        SELECT "Route ID" as rid, count(*) as pts1
        FROM 'data/extracted/202510/Route_2509.csv.gz'
        GROUP BY "Route ID"
    ),
    r2 AS (
        SELECT "Route ID" as rid, count(*) as pts2
        FROM 'data/extracted/202510/Route_2510.csv.gz'
        GROUP BY "Route ID"
    )
    SELECT 
        (SELECT count(*) FROM r1) as total_routes_2509,
        (SELECT count(*) FROM r2) as total_routes_2510,
        (SELECT count(*) FROM r2 WHERE rid NOT IN (SELECT rid FROM r1)) as added_routes,
        (SELECT count(*) FROM r1 WHERE rid NOT IN (SELECT rid FROM r2)) as deleted_routes,
        (SELECT count(*) FROM r1 JOIN r2 ON r1.rid = r2.rid WHERE r1.pts1 != r2.pts2) as modified_waypoints
""").df()

print("Route diff 2509 vs 2510:")
print(res)

# Compare FIR_2509 and FIR_2510
fir_res = con.execute("""
    WITH f1 AS (
        SELECT "Airspace ID" as fid, min("Min Flight Level") as min_fl1, max("Max Flight Level") as max_fl1, count(*) as pts1
        FROM 'data/extracted/202510/FIR_2509.csv.gz'
        GROUP BY "Airspace ID"
    ),
    f2 AS (
        SELECT "Airspace ID" as fid, min("Min Flight Level") as min_fl2, max("Max Flight Level") as max_fl2, count(*) as pts2
        FROM 'data/extracted/202510/FIR_2510.csv.gz'
        GROUP BY "Airspace ID"
    )
    SELECT
        (SELECT count(*) FROM f1) as total_firs_2509,
        (SELECT count(*) FROM f2) as total_firs_2510,
        (SELECT count(*) FROM f2 WHERE fid NOT IN (SELECT fid FROM f1)) as added_firs,
        (SELECT count(*) FROM f1 WHERE fid NOT IN (SELECT fid FROM f2)) as deleted_firs,
        (SELECT count(*) FROM f1 JOIN f2 ON f1.fid = f2.fid WHERE f1.pts1 != f2.pts2) as modified_boundaries
""").df()

print("\nFIR diff 2509 vs 2510:")
print(fir_res)

