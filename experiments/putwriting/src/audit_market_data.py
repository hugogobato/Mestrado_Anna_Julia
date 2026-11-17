"""Create reproducible, read-only source audit artifacts. Run with --help."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import pandas as pd
from market_data import load_market_data, FILES, DATES, KEYS


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(archive_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_market_data(archive_path)
    inventory = []
    with ZipFile(archive_path) as z:
        for info in z.infolist():
            if not info.is_dir():
                inventory.append({'member':info.filename, 'bytes':info.file_size,
                                  'sha256':sha(z.read(info)), 'delimiter':',' if info.filename.endswith('.csv') else None})
        workbook = next(n for n in z.namelist() if n.endswith('.xlsx'))
        with z.open(workbook) as f:
            xl = pd.ExcelFile(f)
            sheet_names = xl.sheet_names
            workbook_schemas = {s:pd.read_excel(xl, sheet_name=s, nrows=0).columns.tolist() for s in sheet_names}
    coverage, missing = [], []
    for name, frame in data.items():
        datecol = DATES[name][0]
        coverage.append({'table':name, 'rows':len(frame), 'start':str(frame[datecol].min().date()), 'end':str(frame[datecol].max().date()), 'key':','.join(KEYS.get(name, ['date'])), 'duplicate_keys':int(frame.duplicated(KEYS.get(name, ['date'])).sum())})
        for col in frame:
            missing.append({'table':name, 'column':col, 'missing':int(frame[col].isna().sum()), 'rows':len(frame)})
    pd.DataFrame(inventory).to_csv(output/'source_inventory.csv', index=False)
    pd.DataFrame(coverage).to_csv(output/'coverage.csv', index=False)
    pd.DataFrame(missing).to_csv(output/'missingness.csv', index=False)
    # Exact-date comparisons are diagnostic only: differing universes and source
    # vintages should not be silently reconciled or forward-filled.
    sp = data['master'].query("asset == 'SP500'")
    comparisons = []
    for label, left, lc, right, rc in [
        ('master_sp500_vs_spx_close', sp, 'Close', data['spx'], 'close'),
        ('master_sp500_vs_vix', sp, 'VIX', data['vix'], 'value_pct')]:
        joined = left[['date',lc]].merge(right[['date',rc]], on='date', suffixes=('_master','_source'))
        a, b = joined.iloc[:,1], joined.iloc[:,2]
        valid = a.notna() & b.notna()
        delta = (a[valid] - b[valid]).abs()
        comparisons.append({'comparison':label,'overlapping_dates':len(joined),'comparable':int(valid.sum()),'different_gt_1e-6':int((delta>1e-6).sum()),'max_abs_difference':float(delta.max()) if len(delta) else None})
    pd.DataFrame(comparisons).to_csv(output/'master_comparison.csv', index=False)
    rolls = data['rolls'].merge(data['settlement'], left_on='expiry',right_on='settle_date',how='left',validate='many_to_one')
    rolls['soq_difference'] = rolls['SOQ_settle'] - rolls['SOQ']
    rolls[['roll_date','expiry','option_ticker','SOQ_settle','settle_date','SOQ','soq_difference']].to_csv(output/'roll_soq_check.csv', index=False)
    set_compare = data['settlement'].merge(data['set_soq'], left_on='third_friday',right_on='expiry',how='outer',suffixes=('_spx','_set'),indicator=True)
    set_compare.to_csv(output/'soq_sources_comparison.csv',index=False)
    pd.DataFrame(columns=['quote_date','option_ticker','expiry','strike','option_type','settlement_type','bid','ask','last','underlying_price','quote_timestamp','currency','multiplier']).to_csv(output/'missing_daily_option_quotes_template.csv',index=False)
    summary = {'archive_sha256':sha(Path(archive_path).read_bytes()), 'sources':inventory,'coverage':coverage,'master_comparisons':comparisons,'workbook_sheets':workbook_schemas, 'roll_soq':{'rows':len(rolls),'missing_spx_soq':int(rolls.SOQ.isna().sum()),'mismatch_gt_1e-6':int((rolls.soq_difference.abs()>1e-6).sum()),'nominal_friday_differs_from_settle_date':int((rolls.third_friday.notna() & (rolls.third_friday != rolls.settle_date)).sum())},'rolls':{'quote_after_roll':int((data['rolls'].quote_date>data['rolls'].roll_date).sum()),'quote_before_roll':int((data['rolls'].quote_date<data['rolls'].roll_date).sum()),'expiry_not_after_roll':int((data['rolls'].expiry<=data['rolls'].roll_date).sum()),'bid_gt_ask':int((data['rolls'].prem_bid>data['rolls'].prem_ask).sum()),'mid_not_bid_ask_average':int(((data['rolls'].prem_mid-(data['rolls'].prem_bid+data['rolls'].prem_ask)/2).abs()>1e-6).sum())}}
    (output/'audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    (output/'AUDITORIA.md').write_text(f'''# Auditoria dos dados\n\nO ZIP contém cotações de entrada para {len(data['rolls'])} puts identificadas em datas mensais de rolagem, com ticker, strike, vencimento, bid, ask, mid, último e valor de liquidação informado. Não contém cotações diárias dos mesmos contratos nem campo que confirme liquidação AM/PM. A identificação e a qualidade das cotações devem ser verificadas com a fonte; um mid não é necessariamente preço executável. A série LIVE é volatilidade implícita, não prêmio de opção individual, e PUT.csv é índice, não contrato.\n\nSPX, LIVE e VIX terminam em 2024-12-31, enquanto as rolagens chegam a 2025. Sinais em 2025 não são suportados por essas séries. A tabela de conferência `roll_soq_check.csv` compara o SOQ de cada rolagem com SPX_SOQ pela data efetiva da liquidação, preservando a sexta-feira nominal quando há feriado. `SET_SOQ.csv` tem valores diferentes e não deve ser confundido com SPX_SOQ sem confirmação de identidade; a comparação está em `soq_sources_comparison.csv`.\n\nAs séries IV e VIX estão em percentuais anualizados, convertidos para decimais por divisão por 100. O rendimento de T-bill também é fornecido em porcentagem e em decimal, mas a convenção da cotação e do cálculo de juros não está documentada no arquivo. Não há preenchimento automático, deduplicação silenciosa, ou inferência do horário de negociação. As contagens de cobertura, lacunas e comparações com master estão nos CSVs.\n\nNa planilha de previsão, o painel atribui valores macro mensais a datas anteriores ao fim do mês de referência (por exemplo janeiro de 2000 já em 3 de janeiro); datas de divulgação e revisões não foram fornecidas. Essas variáveis não são comprovadamente disponíveis em tempo real. O arquivo `missing_daily_option_quotes_template.csv` especifica campos adicionais para marcação diária e identificação de contratos.\n''',encoding='utf8')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('zip_path'); p.add_argument('output_dir'); a=p.parse_args(); run(a.zip_path,a.output_dir)
