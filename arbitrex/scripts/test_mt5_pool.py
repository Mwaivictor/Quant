import os
from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool

sessions = {
	'demo': {
		'terminal_path': None,
		'login': int(os.environ.get('MT5_LOGIN')) if os.environ.get('MT5_LOGIN') else None,
		'password': os.environ.get('MT5_PASSWORD'),
		'server': os.environ.get('MT5_SERVER')
	}
}

pool = MT5ConnectionPool(sessions, session_logs_dir=os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'mt5', 'session_logs'))
try:
	name, sess = pool.get_connection(timeout=10)
	print('Session:', name, 'Status:', sess.status)

	# Export all symbols from MT5 to a JSON file for the trading universe config
	try:
		import MetaTrader5 as mt5
		symbols = mt5.symbols_get()
		records = []
		if symbols is not None:
			for s in symbols:
				# try to convert to dict, otherwise extract common attributes
				try:
					rec = s._asdict()
				except Exception:
					rec = {
						'name': getattr(s, 'name', None),
						'path': getattr(s, 'path', None) if hasattr(s, 'path') else None,
						'currency_base': getattr(s, 'currency_base', None) if hasattr(s, 'currency_base') else None,
						'currency_profit': getattr(s, 'currency_profit', None) if hasattr(s, 'currency_profit') else None,
					}
				records.append(rec)

		out_dir = os.path.join(os.getcwd(), 'arbitrex', 'data', 'raw', 'metadata', 'source_registry')
		os.makedirs(out_dir, exist_ok=True)
		import datetime, json
		ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
		out_path = os.path.join(out_dir, f'mt5_symbols_{ts}.json')
		with open(out_path, 'w', encoding='utf-8') as f:
			json.dump({'export_time_utc': datetime.datetime.utcnow().isoformat() + 'Z', 'count': len(records), 'symbols': records}, f, indent=2)

		print('Wrote', out_path)
	except Exception as e:
		print('Failed to fetch symbols from MT5:', e)

	pool.release_connection((name, sess))
finally:
	pool.close()

