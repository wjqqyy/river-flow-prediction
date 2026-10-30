# -*- coding: utf-8 -*-
"""Prepare offline, calendar-based river forecasting samples; do not train."""
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTOR = 0.028316846592
REVIEW_DATES = {
    '2016-06-05', '2016-06-06', '2016-06-07',
    '2017-07-22', '2017-09-28', '2017-09-30',
}
SPLITS = {
    'train': ('2010-01-01', '2019-12-31'),
    'validation': ('2020-01-01', '2021-12-31'),
    'test': ('2022-01-01', '2024-12-31'),
}
FEATURES = ['q_lag%d_m3_s' % lag for lag in range(7, 0, -1)]


def build_windows(daily):
    """Reindex to calendar days before looking back, so gaps cannot be joined."""
    calendar = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    samples, excluded = [], []
    for target_date in calendar.index:
        dates = pd.date_range(target_date - pd.Timedelta(days=7), target_date)
        window = calendar.reindex(dates)
        bad = window['unusable_reason'].fillna('missing_date_or_history')
        if bad.ne('').any():
            excluded.append({
                'target_date': target_date,
                'unusable_dates': '|'.join(d.strftime('%Y-%m-%d') for d in dates[bad.ne('')]),
                'reasons': '|'.join(sorted(set(bad[bad.ne('')]))),
            })
            continue
        values = window['discharge_m3_s'].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('An invalid value was not marked unusable')
        samples.append(dict(
            target_date=target_date, input_start_date=dates[0], input_end_date=dates[-2],
            **dict(zip(FEATURES, values[:-1])), target_m3_s=values[-1],
        ))
    return pd.DataFrame(samples), pd.DataFrame(excluded)


def fit_scaler(daily):
    selected = daily.loc[SPLITS['train'][0]:SPLITS['train'][1]]
    selected = selected.loc[selected['unusable_reason'].eq(''), 'discharge_m3_s']
    std = float(selected.std(ddof=0))
    if len(selected) == 0 or not np.isfinite(std) or std <= 0:
        raise ValueError('No usable training variance')
    return {'mean_m3_s': float(selected.mean()), 'std_m3_s': std,
            'ddof': 0, 'fitted_day_count': len(selected),
            'fit_start_date': str(selected.index.min().date()),
            'fit_end_date': str(selected.index.max().date()),
            'formula': 'z = (q_m3_s - mean_m3_s) / std_m3_s',
            'inverse_formula': 'q_m3_s = z * std_m3_s + mean_m3_s',
            'applies_to': 'CNN inputs and targets; saved sample CSVs remain in m3/s'}


def self_check():
    # A missing day or unusable target must also invalidate later input windows.
    dates = pd.date_range('2010-01-01', periods=25)
    fixture = pd.DataFrame({'discharge_m3_s': np.arange(25, dtype=float),
                            'unusable_reason': ''}, index=dates)
    fixture.loc['2010-01-09', 'unusable_reason'] = 'review_qualifier'
    fixture = fixture.drop(pd.Timestamp('2010-01-18'))
    valid, excluded = build_windows(fixture)
    assert valid['target_date'].dt.strftime('%Y-%m-%d').tolist() == ['2010-01-08', '2010-01-17']
    assert len(valid) + len(excluded) == 25
    assert valid.iloc[0][FEATURES].to_numpy(dtype=float).tolist() == list(range(7))
    assert valid.iloc[0]['target_m3_s'] == 7
    future = pd.DataFrame({'discharge_m3_s': [1e12], 'unusable_reason': ''},
                          index=pd.to_datetime(['2022-01-01']))
    assert fit_scaler(fixture) == fit_scaler(pd.concat([fixture, future]))
    assert {'True': True, 'False': False}['False'] is False


def main():
    self_check()
    source = ROOT / 'data' / 'processed' / 'daily_discharge_m3s.csv'
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    status = json.loads((ROOT / 'logs' / 'download_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Source download incomplete')
    manifest_path = Path(status['manifest'])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for request in manifest['requests']:
        actual = hashlib.sha256((manifest_path.parent / request['file']).read_bytes()).hexdigest()
        if actual != request['sha256']:
            raise ValueError('Archived source changed: ' + request['file'])
    daily = pd.read_csv(source, encoding='utf-8-sig', keep_default_na=False, dtype=str)
    daily['date'] = pd.to_datetime(daily['date'], format='%Y-%m-%d', errors='raise')
    if daily['date'].duplicated().any():
        raise ValueError('Duplicate dates; refusing ambiguous windows')
    daily = daily.sort_values('date').set_index('date')
    if daily.index.min() != pd.Timestamp('2010-01-01') or daily.index.max() != pd.Timestamp('2024-12-31'):
        raise ValueError('Unexpected data period')
    for column in ['has_estimated_flag', 'needs_qualifier_review']:
        parsed = daily[column].map({'True': True, 'False': False})
        if parsed.isna().any():
            raise ValueError('Invalid boolean: ' + column)
        daily[column] = parsed
    qualifiers = daily['qualifier_json'].map(json.loads)
    if not qualifiers.map(lambda q: q is None or
                          (isinstance(q, list) and all(isinstance(v, str) for v in q))).all():
        raise ValueError('Invalid qualifier representation')
    special = qualifiers.map(lambda q: bool(set(q or []) - {'ESTIMATED', 'REVISED'}))
    if set(daily.index[special].strftime('%Y-%m-%d')) != REVIEW_DATES:
        raise ValueError('Review dates changed; policy needs explicit review')
    if not special.eq(daily['needs_qualifier_review']).all():
        raise ValueError('Review boolean disagrees with qualifiers')
    daily['has_revised_flag'] = qualifiers.map(lambda q: 'REVISED' in (q or []))
    if not qualifiers.map(lambda q: 'ESTIMATED' in (q or [])).eq(daily['has_estimated_flag']).all():
        raise ValueError('Estimated boolean disagrees with qualifiers')

    raw_rows = []
    for filename in manifest['daily_files']:
        response = json.loads((manifest_path.parent / filename).read_text(encoding='utf-8'))
        raw_rows.extend(feature['properties'] for feature in response['features'])
    raw = pd.DataFrame(raw_rows)
    for column, expected in [('monitoring_location_id', 'USGS-01646500'),
                             ('parameter_code', '00060'), ('statistic_id', '00003'),
                             ('unit_of_measure', 'ft^3/s')]:
        if not raw[column].eq(expected).all():
            raise ValueError('Source metadata mismatch: ' + column)
    raw.index = pd.to_datetime(raw['time'], format='%Y-%m-%d')
    raw = raw.sort_index()
    if not raw.index.equals(daily.index):
        raise ValueError('Processed dates differ from archived observations')
    original = pd.to_numeric(daily['discharge_ft3_s'], errors='coerce').to_numpy()
    values = pd.to_numeric(daily['discharge_m3_s'], errors='coerce').to_numpy()
    if not np.allclose(original, pd.to_numeric(raw['value'], errors='coerce'), equal_nan=True):
        raise ValueError('Original units disagree with source')
    if not np.allclose(values, original * FACTOR, rtol=1e-12, atol=1e-10, equal_nan=True):
        raise ValueError('Unit conversion mismatch; do not reconvert m3/s')
    if qualifiers.tolist() != raw['qualifier'].tolist() or daily['approval_status'].tolist() != raw['approval_status'].tolist():
        raise ValueError('Quality metadata disagree with archived observations')
    daily['discharge_m3_s'] = values
    daily['unusable_reason'] = ''
    daily.loc[~np.isfinite(values) | (values < 0), 'unusable_reason'] = 'invalid_flow'
    daily.loc[~daily['approval_status'].eq('Approved'), 'unusable_reason'] = 'not_approved'
    daily.loc[special, 'unusable_reason'] = 'review_qualifier'
    daily = daily.reindex(pd.date_range('2010-01-01', '2024-12-31'))
    daily.index.name = 'date'
    daily['unusable_reason'] = daily['unusable_reason'].fillna('missing_date')
    daily['usable_for_model'] = daily['unusable_reason'].eq('')
    samples, excluded = build_windows(daily)
    scaler = fit_scaler(daily)
    model_dir = ROOT / 'data' / 'model'
    model_dir.mkdir(exist_ok=True)
    summary = {
        'prepared_at': dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        'model_ready': True, 'models_trained': False,
        'source': str(source), 'source_sha256': source_sha,
        'raw_manifest': str(manifest_path), 'raw_sha256_verified': True,
        'input_days': 7, 'horizon_days': 1, 'unit': 'm3/s',
        'feature_columns': FEATURES, 'target_column': 'target_m3_s',
        'feature_order': 'oldest (t-7) to newest (t-1)',
        'review_dates_excluded': sorted(REVIEW_DATES),
        'retained_qualifiers': ['ESTIMATED', 'REVISED'],
        'policy': 'Approved data only; exclude any 8-day window containing an unusable day; '
                  'no interpolation, smoothing, clipping or peak removal. Current revised archive, not historical real-time availability.',
        'prediction_mode': 'Daily rolling one-step prediction using previously observed discharge; '
                           'history may cross split boundaries; no model updates during test.',
        'calendar_days': len(daily), 'usable_days': int(daily['usable_for_model'].sum()),
        'retained_samples': len(samples), 'excluded_target_dates': len(excluded),
        'splits': {}, 'scaler': scaler,
    }
    for name, (start, end) in SPLITS.items():
        part = samples.loc[samples['target_date'].between(start, end)].copy()
        dropped = excluded.loc[excluded['target_date'].between(start, end)].copy()
        days = daily.loc[start:end]
        target_flags = daily.loc[pd.DatetimeIndex(part['target_date'])]
        affected = {}
        for flag in ['has_estimated_flag', 'has_revised_flag']:
            affected[flag] = int(sum(daily.loc[date - pd.Timedelta(days=7):date, flag].fillna(False).any()
                                     for date in part['target_date']))
        summary['splits'][name] = {
            'target_period_start': start, 'target_period_end': end,
            'samples': len(part), 'excluded_target_dates': len(dropped),
            'first_target_date': str(part['target_date'].min().date()),
            'last_target_date': str(part['target_date'].max().date()),
            'daily_qualifier_counts': {'ESTIMATED': int(days['has_estimated_flag'].sum()),
                                       'REVISED': int(days['has_revised_flag'].sum()),
                                       'review_excluded': int(days['unusable_reason'].eq('review_qualifier').sum())},
            'retained_target_qualifier_counts': {'ESTIMATED': int(target_flags['has_estimated_flag'].sum()),
                                                'REVISED': int(target_flags['has_revised_flag'].sum())},
            'retained_windows_with_flags': affected,
        }
        part.to_csv(model_dir / (name + '.csv'), index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
        excluded.loc[dropped.index, 'split'] = name

    # Verify outputs against an independent date-membership rule and actual daily values.
    blocked = set(daily.index[~daily['usable_for_model']])
    expected = [date for date in daily.index[7:]
                if not any(date - pd.Timedelta(days=lag) in blocked for lag in range(8))]
    assert samples['target_date'].tolist() == expected
    assert len(samples) + len(excluded) == len(daily)
    for name, (start, end) in SPLITS.items():
        saved = pd.read_csv(model_dir / (name + '.csv'), encoding='utf-8-sig')
        dates = pd.DatetimeIndex(saved['target_date'])
        assert dates.min() >= pd.Timestamp(start) and dates.max() <= pd.Timestamp(end)
        for lag, feature in zip(range(7, 0, -1), FEATURES):
            assert np.allclose(saved[feature], daily.loc[dates - pd.Timedelta(days=lag), 'discharge_m3_s'])
        assert np.allclose(saved['target_m3_s'], daily.loc[dates, 'discharge_m3_s'])
        assert np.isfinite(saved[FEATURES + ['target_m3_s']].to_numpy()).all()
    perturbed = daily.copy()
    perturbed.loc['2020-01-01':, 'discharge_m3_s'] = 1e12
    assert fit_scaler(perturbed) == scaler
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha
    assert sum(s['samples'] for s in summary['splits'].values()) == len(samples)
    for date in ['2020-01-01', '2022-01-01']:
        assert pd.Timestamp(date) in set(samples['target_date'])
    excluded.to_csv(ROOT / 'reports' / 'excluded_model_samples.csv', index=False,
                    encoding='utf-8-sig', date_format='%Y-%m-%d')
    daily.to_csv(model_dir / 'daily_eligibility.csv', encoding='utf-8-sig', date_format='%Y-%m-%d')
    (model_dir / 'normalization.json').write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding='utf-8')
    (model_dir / 'sample_config.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    table = '\n'.join('| %s | %s 至 %s | %d | %d |' %
                      (name, s['first_target_date'], s['last_target_date'], s['samples'], s['excluded_target_dates'])
                      for name, s in summary['splits'].items())
    report = f'''# 建模样本准备结果

执行时间：{summary['prepared_at']}。样本准备完成，未训练模型。

## 样本与划分

过去连续7天输入，预测下一天日平均流量；全部使用已有 m³/s 数值，没有重复换算。
特征按 t−7 至 t−1 排列，目标为 t 日流量。按目标日期划分，不随机拆分。

| 分区 | 实际目标日期范围 | 保留样本 | 排除目标日期 |
|---|---|---:|---:|
{table}

总计保留 {len(samples)} 个样本，排除 {len(excluded)} 个目标日期。
2010年最初7天因没有足够历史而排除；其余排除记录和具体原因见 excluded_model_samples.csv。
验证和测试首日均保留，可使用分区边界之前的已观测历史，不使用目标当天或未来流量。
这是逐日滚动一步预测，后续模型不在测试期更新参数。

## 数据使用规则

保留 Approved 的 ESTIMATED、REVISED 记录；数据包含来源机构的估算和修订版本。
对6天 EQUIP / FORCEINTERPOLATION 记录采用保守排除规则，不判断其数值错误。
任何7天输入或1天目标包含不可用日的样本均排除，先按日历对齐，避免跨缺口拼接。
不插值、不平滑、不裁剪、不删除洪峰；原始JSON和已有处理CSV保持不变。
各分区的日记录标记数、保留目标标记数以及包含标记的窗口数分别记录在 sample_config.json，不能混为一谈。

## 标准化与后续使用

仅用2010—2019年有效日值拟合均值和标准差（ddof=0），有效日数 {scaler['fitted_day_count']}。
均值 {scaler['mean_m3_s']:.9f} m³/s；标准差 {scaler['std_m3_s']:.9f} m³/s。
参数保存在 normalization.json，供CNN输入和目标共用；三个样本CSV保留真实单位。
后续预测必须还原单位再计算指标。仅 FEATURES 指定的7列可作为预测特征，日期、标记和元数据不作为输入。

## 验证与状态

已通过：原始文件SHA-256核验、来源参数核对、处理值及标记与原始记录比对、布尔字段解析、
日期窗口独立核对、保存结果回读、分区边界检查，以及修改验证/测试数值不影响标准化参数的检查。
小型自检还覆盖了缺失日期和不可用目标日对后续窗口的影响。
当前 sample_config.json 中 model_ready=true 表示上述排除规则下的样本已准备好。
此前 data_check_summary.json 的 model_ready=false 属于第一阶段的历史检查结论，原报告保留。
下一步可训练持续性预测、线性回归和小型1D-CNN；本阶段没有精度指标。
'''
    (ROOT / 'reports' / 'sample_preparation_summary.md').write_text(report, encoding='utf-8')
    message = json.dumps({'retained_samples': len(samples), 'excluded_target_dates': len(excluded),
                          'splits': summary['splits'], 'checks': 'PASS', 'models_trained': False},
                         ensure_ascii=False, indent=2)
    (ROOT / 'logs' / 'sample_preparation.log').write_text(message + '\n', encoding='utf-8')
    print(message)


if __name__ == '__main__':
    main()
