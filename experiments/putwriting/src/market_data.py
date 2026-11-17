"""Read the supplied archive without inventing prices or filling missing observations.

Schemas: daily tables have `date` and `close` (SPX additionally open/high/low/
volume); iv and vix have date, value_pct, value_decimal; tbill has date,
yield_pct, yield_decimal. rolls, settlement and master retain source columns.
Dates are naive pandas timestamps. Decimal rates are percent / 100; no yield
or day-count conversion is assumed. Duplicate keys and invalid dates fail fast.
"""
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

FILES = {'spx':'SPX.csv', 'iv':'SPX_ATM1M.csv', 'vix':'VIX.csv', 'tbill':'TBILL1M.csv', 'put_index':'PUT.csv', 'rolls':'putwrite_rolls.csv', 'settlement':'SPX_SOQ.csv', 'set_soq':'SET_SOQ.csv', 'master':'master.csv'}
DATES = {'spx':['date'], 'iv':['date'], 'vix':['date'], 'tbill':['date'], 'put_index':['date'], 'rolls':['roll_date','expiry','quote_date'], 'settlement':['third_friday','settle_date'], 'set_soq':['expiry'], 'master':['date']}
KEYS = {'rolls':['roll_date'], 'settlement':['third_friday'], 'set_soq':['expiry'], 'master':['asset','date']}
RENAMES = {'spx':{'PX_OPEN':'open','PX_HIGH':'high','PX_LOW':'low','PX_LAST':'close','PX_VOLUME':'volume'}, 'iv':{'atm1m_impvol_mid_VL137':'value_pct'}, 'vix':{'PX_LAST':'value_pct'}, 'tbill':{'PX_LAST':'yield_pct'}, 'put_index':{'PX_LAST':'close'}}

def load_market_data(zip_path):
    """Load source data, reject duplicate keys, and preserve missing values."""
    result = {}
    with ZipFile(Path(zip_path)) as archive:
        for name, basename in FILES.items():
            matches = [n for n in archive.namelist() if n.rsplit('/', 1)[-1] == basename]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {basename}, found {len(matches)}')
            with archive.open(matches[0]) as source:
                frame = pd.read_csv(source, sep=',', encoding='utf-8-sig')
            frame = frame.rename(columns=RENAMES.get(name, {}))
            for column in DATES[name]:
                frame[column] = pd.to_datetime(frame[column], format='%Y-%m-%d', errors='raise')
                if frame[column].isna().any():
                    raise ValueError(f'{name}: missing date in {column}')
            keys = KEYS.get(name, ['date'])
            if frame.duplicated(keys, keep=False).any():
                raise ValueError(f'{name}: duplicate key {keys}')
            if name in ('iv', 'vix'):
                frame['value_decimal'] = frame['value_pct'] / 100.0
            if name == 'tbill':
                frame['yield_decimal'] = frame['yield_pct'] / 100.0
            result[name] = frame.sort_values(keys).reset_index(drop=True)
    return result
