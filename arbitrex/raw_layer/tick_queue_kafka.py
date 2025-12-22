"""Kafka-backed tick publisher (producer).

This adapter provides `enqueue(...)` that publishes tick JSON messages to a Kafka topic.
It does not implement `dequeue_all_for_symbol` because Kafka is a streaming system and
consumers typically run independently; for durable storage rely on CSV/audit writes or a separate consumer job.

Requires `confluent_kafka` or `kafka-python`.
"""
import json
import threading
from typing import Optional

try:
    from confluent_kafka import Producer
    _HAS_CONFLUENT = True
except Exception:
    _HAS_CONFLUENT = False

try:
    from kafka import KafkaProducer
    _HAS_KAFKA_PY = True
except Exception:
    _HAS_KAFKA_PY = False


class KafkaTickQueue:
    def __init__(self, bootstrap_servers: Optional[str] = None, topic: str = 'ticks'):
        self._topic = topic
        self._lock = threading.RLock()
        self._producer = None
        if _HAS_CONFLUENT:
            cfg = {'bootstrap.servers': bootstrap_servers or 'localhost:9092'}
            self._producer = Producer(cfg)
        elif _HAS_KAFKA_PY:
            self._producer = KafkaProducer(bootstrap_servers=(bootstrap_servers or 'localhost:9092').split(','), value_serializer=lambda v: json.dumps(v).encode('utf-8'))
        else:
            raise RuntimeError('No Kafka producer library available')

    def enqueue(self, symbol: str, ts: int, bid: Optional[float], ask: Optional[float], last: Optional[float], volume: Optional[float], seq: Optional[str] = None):
        msg = {'symbol': symbol, 'ts': int(ts), 'bid': bid, 'ask': ask, 'last': last, 'volume': volume, 'seq': seq}
        with self._lock:
            if _HAS_CONFLUENT:
                self._producer.produce(self._topic, json.dumps(msg).encode('utf-8'))
                self._producer.poll(0)
                return True
            elif _HAS_KAFKA_PY:
                self._producer.send(self._topic, msg)
                return True

    def dequeue_all_for_symbol(self, symbol: str):
        raise NotImplementedError('KafkaTickQueue is producer-only; use a consumer job for reads')

    def delete_ids(self, ids):
        # Not applicable for Kafka
        return

    def count(self, symbol: Optional[str] = None) -> int:
        raise NotImplementedError('Count not supported for KafkaTickQueue')

    def close(self):
        try:
            if _HAS_CONFLUENT and self._producer:
                self._producer.flush()
            elif _HAS_KAFKA_PY and self._producer:
                self._producer.flush()
        except Exception:
            pass
