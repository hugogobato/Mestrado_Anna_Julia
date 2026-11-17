import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from market_data import load_market_data

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'BASES DE DADOS-20261117T124409Z-1-001.zip'

class MarketDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_market_data(ARCHIVE)

    def test_decimal_volatility_and_rate(self):
        for table, pct, decimal in [('iv','value_pct','value_decimal'),('vix','value_pct','value_decimal'),('tbill','yield_pct','yield_decimal')]:
            row = self.data[table].iloc[0]
            self.assertAlmostEqual(row[decimal], row[pct]/100)
        self.assertAlmostEqual(self.data['iv'].iloc[0]['value_decimal']**2, (12.1911/100)**2)

    def test_holiday_roll_settlement_matches_actual_date(self):
        r = self.data['rolls'].merge(self.data['settlement'], left_on='expiry', right_on='settle_date', validate='many_to_one')
        self.assertEqual(len(r), len(self.data['rolls']))
        self.assertLess((r.SOQ_settle - r.SOQ).abs().max(), 1e-6)
        self.assertEqual(int((r.expiry != r.third_friday).sum()), 3)
        self.assertTrue((r.quote_date <= r.roll_date).all())

    def test_no_future_fill_and_dates(self):
        self.assertEqual(str(self.data['iv'].date.max().date()), '2024-12-31')
        self.assertTrue(self.data['iv'].date.is_monotonic_increasing)
        self.assertEqual(self.data['spx'].close.isna().sum(), 1)

    def test_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'duplicate.zip'
            with ZipFile(ARCHIVE) as source, ZipFile(dest, 'w') as out:
                for info in source.infolist():
                    raw = source.read(info.filename)
                    if info.filename.endswith('/PUT.csv'):
                        raw += raw.splitlines(keepends=True)[1]
                    out.writestr(info.filename, raw)
            with self.assertRaisesRegex(ValueError, 'duplicate key'):
                load_market_data(dest)

if __name__ == '__main__':
    unittest.main()
