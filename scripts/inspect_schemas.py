import duckdb

con = duckdb.connect()

print("--- AIRAC_2510.csv.gz ---")
print(con.execute("SELECT * FROM read_csv_auto('data/extracted/202510/AIRAC_2510.csv.gz')").df())

print("\n--- Flight_FIRs_Actual Sample ---")
print(con.execute("SELECT * FROM read_csv_auto('data/extracted/202510/Flight_FIRs_Actual_20251001_20251031.csv.gz') LIMIT 3").df())

print("\n--- Route_2510 Sample ---")
print(con.execute("SELECT * FROM read_csv_auto('data/extracted/202510/Route_2510.csv.gz') LIMIT 3").df())

print("\n--- FIR_2510 Sample ---")
print(con.execute("SELECT * FROM read_csv_auto('data/extracted/202510/FIR_2510.csv.gz') LIMIT 3").df())

