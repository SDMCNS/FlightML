import duckdb
import threading

con = duckdb.connect()
con.execute("CREATE TABLE t AS SELECT unnest(range(100000)) as x")

def query_worker(cursor, idx):
    res = cursor.execute("SELECT count(*), avg(x) FROM t").fetchall()
    print(f"Thread {idx} success: {res}")

threads = []
for i in range(10):
    cur = con.cursor()
    t = threading.Thread(target=query_worker, args=(cur, i))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("All 10 concurrent threads completed with independent cursors!")

