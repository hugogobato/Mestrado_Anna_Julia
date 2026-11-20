"""Ingest the supplied Bloomberg follow-up and audit it against original inputs.

Dates are naive trading dates. Quotes are index points per unit, not dollars
per contract; IV and yield percentages are annualized. Missing marks are never
filled. The yield's precise quote convention and timestamps are not supplied.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse
import hashlib
import json
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from market_data import load_market_data

NEW_FILES = {'spx': 'SPX_2025.csv', 'iv': 'SPX_ATM1M_2025.csv',
             'vix': 'VIX_2025.csv', 'quotes': 'SPX_PUT_cotacoes_diarias.csv'}
RENAMES = {'spx': {'PX_OPEN': 'open', 'PX_HIGH': 'high', 'PX_LOW': 'low',
                   'PX_LAST': 'close', 'PX_VOLUME': 'volume'},
           'iv': {'atm1m_impvol_mid_VL137': 'value_pct'},
           'vix': {'PX_LAST': 'value_pct'},
           'quotes': {'PX_BID': 'bid', 'PX_ASK': 'ask', 'PX_MID': 'mid',
                      'PX_LAST': 'last', 'IVOL_MID': 'ivol_mid_pct',
                      'PX_VOLUME': 'volume', 'OPEN_INT': 'open_int'}}


def _read_followup(path):
    tables = {}
    with ZipFile(path) as archive:
        for name, filename in NEW_FILES.items():
            matches = [n for n in archive.namelist() if n.rsplit('/', 1)[-1] == filename]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {filename}, found {len(matches)}')
            with archive.open(matches[0]) as f:
                df = pd.read_csv(f, encoding='utf-8-sig').rename(columns=RENAMES[name])
            dates = ['quote_date', 'roll_date', 'expiry'] if name == 'quotes' else ['date']
            for col in dates:
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='raise')
                if df[col].isna().any():
                    raise ValueError(f'{name}: missing {col}')
            keys = ['option_ticker', 'quote_date'] if name == 'quotes' else ['date']
            if df.duplicated(keys).any():
                raise ValueError(f'{name}: duplicate key {keys}')
            if name in ('iv', 'vix'):
                df['value_decimal'] = df.value_pct / 100
            tables[name] = df.sort_values(keys).reset_index(drop=True)
    return tables


def load_followup_market(original_zip, followup_zip):
    """Return legacy market tables plus quotes; reject conflicting overlapping dates."""
    data, new = load_market_data(original_zip), _read_followup(followup_zip)
    for name in ('spx', 'iv', 'vix'):
        old, additional = data[name], new[name]
        columns = [c for c in additional if c != 'date']
        comparison = old.merge(additional, on='date', suffixes=('_old', '_new'))
        for col in columns:
            a, b = comparison[col + '_old'], comparison[col + '_new']
            if not ((a == b) | (a.isna() & b.isna())).all():
                raise ValueError(f'{name}: conflicting overlap in {col}')
        data[name] = pd.concat([old, additional[~additional.date.isin(old.date)]],
                               ignore_index=True).sort_values('date').reset_index(drop=True)
    data['quotes'] = new['quotes']
    return data


def audit(original_zip, followup_zip, output):
    """Generate reproducible diagnostics, without copying raw Bloomberg inputs."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    old, new = load_market_data(original_zip), _read_followup(followup_zip)
    data = load_followup_market(original_zip, followup_zip)
    overlaps = []
    for name in ('spx', 'iv', 'vix'):
        joined = old[name].merge(new[name], on='date', suffixes=('_old', '_new'))
        overlaps.append({'series': name, 'overlap_dates': len(joined),
                         'old_end': str(old[name].date.max().date()),
                         'new_start': str(new[name].date.min().date()),
                         'new_end': str(new[name].date.max().date()), 'new_rows': len(new[name])})
    pd.DataFrame(overlaps).to_csv(output / 'overlap.csv', index=False)
    q, rolls, sessions = data['quotes'], data['rolls'], data['spx'].dropna(subset=['close'])
    qroll = q.merge(rolls[['option_ticker','roll_date','expiry','strike']],
                    on=['option_ticker','roll_date','expiry','strike'], how='left', indicator=True)
    entry = rolls.merge(q, on=['option_ticker','roll_date','expiry','strike'], how='left')
    entry = entry[entry.quote_date_y == entry.roll_date].copy()
    entry['bid_difference'] = entry.bid - entry.prem_bid
    entry['ask_difference'] = entry.ask - entry.prem_ask
    entry['mid_difference'] = entry.mid - entry.prem_mid
    entry[['roll_date','expiry','option_ticker','bid_difference','ask_difference','mid_difference']].to_csv(output/'entry_matches.csv', index=False)
    so = rolls.merge(data['settlement'], left_on='expiry', right_on='settle_date', how='left', validate='many_to_one')
    so['difference'] = so.SOQ_settle - so.SOQ
    so[['roll_date','expiry','option_ticker','SOQ_settle','settle_date','SOQ','difference']].to_csv(output/'settlement_matches.csv',index=False)
    expected = rolls[['option_ticker','roll_date','expiry']].merge(sessions[['date']], how='cross')
    expected = expected[(expected.date >= expected.roll_date) & (expected.date < expected.expiry)]
    gaps = expected.merge(q[['option_ticker','quote_date','bid','ask','mid']], left_on=['option_ticker','date'], right_on=['option_ticker','quote_date'], how='left', indicator=True)
    gaps = gaps[(gaps['_merge'] != 'both') | gaps.mid.isna()].copy()
    gaps['reason'] = gaps['_merge'].map({'left_only':'no_quote_row','both':'missing_mid'}).astype(str)
    gaps[['option_ticker','roll_date','expiry','date','reason','bid','ask','mid']].to_csv(output/'quote_gaps.csv', index=False)
    missing = q.isna().sum().rename('missing').to_frame(); missing['rows'] = len(q)
    missing.to_csv(output/'quote_missingness.csv', index_label='column')
    qs = q.groupby('option_ticker').agg(first_quote=('quote_date','min'), last_quote=('quote_date','max'), expiry=('expiry','first'), rows=('quote_date','size'))
    qs.to_csv(output/'contract_coverage.csv')
    summary = {'source_sha256': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [original_zip, followup_zip]},
        'overlaps': overlaps, 'sessions_2025': {k: int(((data[k].date.dt.year == 2025)).sum()) for k in ('spx','iv','vix','tbill')},
        'vix_extra_dates_2025': sorted(set(data['vix'].loc[data['vix'].date.dt.year==2025,'date'].dt.strftime('%Y-%m-%d')) - set(data['spx'].loc[data['spx'].date.dt.year==2025,'date'].dt.strftime('%Y-%m-%d'))),
        'quotes': {'rows':len(q),'contracts':int(q.option_ticker.nunique()), 'unmatched_identity_rows':int((qroll['_merge'] != 'both').sum()), 'missing':{k:int(v) for k,v in q.isna().sum().items()}, 'entry_matches':len(entry), 'entry_bid_mismatches':int((entry.bid_difference.abs()>1e-8).sum()), 'entry_ask_mismatches':int((entry.ask_difference.abs()>1e-8).sum()), 'entry_mid_mismatches':int((entry.mid_difference.abs()>1e-8).sum()), 'bid_above_ask':int((q.bid>q.ask).sum()), 'mid_mismatches':int(((q.mid-(q.bid+q.ask)/2).abs()>1e-8).sum()), 'rows_on_or_after_expiry':int((q.quote_date>=q.expiry).sum()), 'expected_session_gaps':len(gaps), 'missing_quote_rows':int((gaps.reason=='no_quote_row').sum()), 'missing_mid_rows':int((gaps.reason=='missing_mid').sum())},
        'settlement': {'rows':len(so), 'missing':int(so.SOQ.isna().sum()), 'mismatches':int((so.difference.abs()>1e-8).sum()), 'nominal_friday_differs':int((so.third_friday != so.settle_date).sum())}}
    (output/'audit.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('original_zip'); p.add_argument('followup_zip'); p.add_argument('output'); args = p.parse_args()
    print(json.dumps(audit(args.original_zip, args.followup_zip, args.output), indent=2))
