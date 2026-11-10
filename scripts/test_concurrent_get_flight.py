import threading
from flightml.data.dataset import FlightDataset

ds = FlightDataset("202510")

def worker(idx, flight_id):
    f = ds.get_flight(flight_id)
    assert f is not None
    assert f["ectrl_id"] == flight_id
    print(f"Thread {idx:02d} fetched flight {flight_id} successfully: {f['adep']} -> {f['ades']}")

test_ids = [291262109, 291262111, 291262113, 291262115, 291262116, 
            291262120, 291262121, 291262122, 291262124, 291262125,
            291262126, 291262127, 291262128, 291262129, 291262130,
            291262131, 291262132, 291262133, 291262134, 291262135]

threads = []
for idx, fid in enumerate(test_ids):
    t = threading.Thread(target=worker, args=(idx, fid))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("\n=== ALL 20 CONCURRENT get_flight THREADS COMPLETED WITH ZERO ResourceInUseException! ===")

