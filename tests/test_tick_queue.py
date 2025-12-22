import os
import tempfile
from arbitrex.raw_layer.tick_queue import TickQueue


def test_enqueue_dequeue_delete(tmp_path):
    db = tmp_path / "ticks_queue.db"
    tq = TickQueue(str(db))
    try:
        # enqueue some ticks
        id1 = tq.enqueue('EURUSD', 1000, 1.1, 1.1001, 1.10005, 100)
        id2 = tq.enqueue('EURUSD', 1001, 1.2, 1.2001, 1.20005, 200)
        id3 = tq.enqueue('GBPUSD', 1002, 1.3, 1.3001, 1.30005, 300)

        assert tq.count() == 3
        assert tq.count('EURUSD') == 2

        rows = tq.dequeue_all_for_symbol('EURUSD')
        # rows are tuples: (id, ts, bid, ask, last, vol, seq)
        assert len(rows) == 2
        assert rows[0][1] == 1000
        assert rows[1][1] == 1001

        # delete first two ids
        ids = [rows[0][0], rows[1][0]]
        tq.delete_ids(ids)
        assert tq.count('EURUSD') == 0
        assert tq.count() == 1
    finally:
        tq.close()
