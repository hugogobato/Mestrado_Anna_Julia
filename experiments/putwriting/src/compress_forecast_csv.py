"""Retain generated forecast CSVs losslessly as gzip for repository publication."""
import gzip
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def run():
    output = BASE / 'results/csv_compression_manifest.json'
    previous = json.loads(output.read_text()) if output.exists() else []
    entries = {item['file']: item for item in previous}
    compressed = 0
    for folder in ('forecast_revision', 'neural_purged'):
        for path in sorted((BASE / 'results' / folder / 'forecasts').rglob('*.csv')):
            original = path.read_bytes()
            target = path.with_suffix('.csv.gz')
            with target.open('wb') as f:
                with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as z:
                    z.write(original)
            if gzip.decompress(target.read_bytes()) != original:
                raise ValueError(f'compression mismatch: {path}')
            name = str(target.relative_to(BASE))
            entries[name] = {'file': name, 'original_bytes': len(original),
                             'original_sha256': hashlib.sha256(original).hexdigest()}
            compressed += 1
            path.unlink()
    if entries:
        output.write_text(json.dumps([entries[name] for name in sorted(entries)], indent=2) + '\n')
    print(f'Losslessly compressed {compressed} generated forecast CSV files; {len(entries)} retained manifest entries')


if __name__ == '__main__':
    run()
