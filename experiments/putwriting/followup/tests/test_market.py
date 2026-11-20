from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from market import load_followup_market

ROOT = Path(__file__).resolve().parents[4]
ORIGINAL = ROOT / 'BASES DE DADOS-20261117T124409Z-1-001.zip'
FOLLOWUP = ROOT / 'DADOS FALTANTES-20261120T201147Z-1-001.zip'


@unittest.skipUnless(ORIGINAL.exists() and FOLLOWUP.exists(), 'Source archives not available')
class FollowupMarketTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_followup_market(ORIGINAL, FOLLOWUP)

    def test_2025_and_missing_marks(self):
        self.assertEqual(sum(self.d['spx'].date.dt.year == 2025), 250)
        self.assertEqual(sum(self.d['iv'].date.dt.year == 2025), 250)
        self.assertEqual(sum(self.d['vix'].date.dt.year == 2025), 258)
        self.assertEqual(self.d['quotes'].mid.isna().sum(), 24)
        self.assertEqual(len(self.d['quotes']), 2517)
        self.assertTrue((self.d['quotes'].quote_date < self.d['quotes'].expiry).all())

    def test_entry_quotes_and_settlement(self):
        r, q = self.d['rolls'], self.d['quotes']
        e = r.merge(q, on=['roll_date', 'expiry', 'strike', 'option_ticker'])
        e = e[e.quote_date_y == e.roll_date]
        self.assertEqual(len(e), 120)
        for a, b in [('bid', 'prem_bid'), ('ask', 'prem_ask'), ('mid', 'prem_mid')]:
            self.assertLess((e[a] - e[b]).abs().max(), 1e-8)
        s = r.merge(self.d['settlement'], left_on='expiry', right_on='settle_date', validate='many_to_one')
        self.assertEqual(len(s), len(r))
        self.assertLess((s.SOQ_settle-s.SOQ).abs().max(), 1e-8)

if __name__ == '__main__':
    unittest.main()
