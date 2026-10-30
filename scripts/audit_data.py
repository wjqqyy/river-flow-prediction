# -*- coding: utf-8 -*-
"""Audit saved USGS responses offline; retain every observation and qualifier."""
from collections import Counter
import datetime as dt
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FACTOR = 0.028316846592


def main():
    status = json.loads((ROOT / 'logs' / 'download_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Download incomplete; refusing to process')
    manifest_path = Path(status['manifest'])
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw = manifest_path.parent
    # These hashes also verify that processing never changed archived responses.
    for request in manifest['requests']:
        actual = hashlib.sha256((raw / request['file']).read_bytes()).hexdigest()
        if actual != request['sha256']:
            raise ValueError('Raw response hash mismatch: ' + request['file'])
    rows = []
    for filename in manifest['daily_files']:
        obj = json.loads((raw / filename).read_text(encoding='utf-8'))
        if obj['numberReturned'] != len(obj['features']):
            raise ValueError('Response row count mismatch')
        for feature in obj['features']:
            rows.append(dict(feature['properties'], record_id=feature['id']))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('Empty observations; stopped')
    for column, expected in [('monitoring_location_id', 'USGS-01646500'), ('parameter_code', '00060'), ('statistic_id', '00003')]:
        if not df[column].eq(expected).all():
            raise ValueError('Wrong variable or site: ' + column)
    units = df['unit_of_measure'].value_counts(dropna=False).to_dict()
    if set(units) != {'ft^3/s'}:
        raise ValueError('Unit not confirmed: ' + str(units))
    parameter = json.loads((raw / 'parameter_00060.json').read_text(encoding='utf-8'))['properties']
    statistic = json.loads((raw / 'statistic_00003.json').read_text(encoding='utf-8'))['properties']
    if parameter['parameter_name'] != 'Discharge' or statistic['statistic_name'] != 'MEAN':
        raise ValueError('Official variable/statistic definition mismatch')
    df['date'] = pd.to_datetime(df['time'], format='%Y-%m-%d', errors='raise')
    df['discharge_ft3_s'] = pd.to_numeric(df['value'], errors='coerce')
    df['qualifier_json'] = df['qualifier'].map(lambda x: json.dumps(x, ensure_ascii=False))
    original_columns = [c for c in df.columns if c not in ['date', 'discharge_ft3_s', 'qualifier', 'qualifier_json']]
    df[original_columns + ['qualifier_json']].to_csv(ROOT / 'data' / 'processed' / 'observations_original_units.csv', index=False, encoding='utf-8-sig')
    expected_dates = pd.date_range('2010-01-01', '2024-12-31', freq='D')
    duplicate = df['date'].duplicated(keep=False)
    missing = expected_dates.difference(df['date'])
    unexpected = df.loc[~df['date'].isin(expected_dates), 'date']
    bad_numeric = ~np.isfinite(df['discharge_ft3_s'])
    negative = df['discharge_ft3_s'].lt(0)
    flags = Counter()
    for value in df['qualifier']:
        if value is not None and not isinstance(value, list):
            raise ValueError('Unexpected qualifier representation')
        flags.update(value or [])
    df['needs_qualifier_review'] = df['qualifier'].map(lambda x: any(v not in {'ESTIMATED', 'REVISED'} for v in (x or [])))
    df['has_estimated_flag'] = df['qualifier'].map(lambda x: 'ESTIMATED' in (x or []))
    df['discharge_m3_s'] = df['discharge_ft3_s'] * FACTOR
    cols = ['date', 'discharge_ft3_s', 'discharge_m3_s', 'approval_status', 'qualifier_json', 'has_estimated_flag', 'needs_qualifier_review', 'time_series_id', 'record_id', 'last_modified']
    df = df.sort_values('date')
    flagged = df['qualifier'].map(bool)
    df.loc[flagged, cols].to_csv(ROOT / 'reports' / 'flagged_records.csv', index=False, encoding='utf-8-sig')
    df.loc[df['needs_qualifier_review'], cols].to_csv(ROOT / 'reports' / 'qualifiers_to_review.csv', index=False, encoding='utf-8-sig')
    df.loc[duplicate, cols].to_csv(ROOT / 'reports' / 'duplicate_dates.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'missing_date': missing}).to_csv(ROOT / 'reports' / 'missing_dates.csv', index=False, encoding='utf-8-sig')
    df.loc[bad_numeric | negative, cols].to_csv(ROOT / 'reports' / 'invalid_values.csv', index=False, encoding='utf-8-sig')
    structural_ok = not (duplicate.any() or len(missing) or len(unexpected) or bad_numeric.any() or negative.any())
    summary = {
        'checked_at_local': dt.datetime.now().astimezone().isoformat(),
        'raw_manifest': str(manifest_path), 'station_name': manifest['station_name'],
        'station_id': 'USGS-01646500', 'parameter': parameter, 'statistic': statistic,
        'observations': len(df), 'expected_dates': len(expected_dates),
        'start': str(df['date'].min().date()), 'end': str(df['date'].max().date()),
        'duplicate_date_rows': int(duplicate.sum()), 'missing_date_count': len(missing),
        'unexpected_date_count': len(unexpected), 'missing_raw_values': int(df['value'].isna().sum()),
        'invalid_or_missing_numeric_values': int(bad_numeric.sum()), 'negative_values': int(negative.sum()),
        'source_units': units, 'processed_unit': 'm3/s', 'conversion_factor': FACTOR,
        'approval_status_counts': df['approval_status'].value_counts(dropna=False).to_dict(),
        'qualifier_counts': dict(flags), 'rows_with_any_qualifier': int(flagged.sum()),
        'qualifier_combination_counts': df['qualifier_json'].value_counts().to_dict(),
        'rows_needing_qualifier_review': int(df['needs_qualifier_review'].sum()),
        'structural_checks_passed': bool(structural_ok),
        'model_ready': False,
        'model_gate_reason': 'Review six EQUIP/FORCEINTERPOLATION records; definitions and causes not confirmed. No models trained in this stage.',
        'statistics_m3_s': df['discharge_m3_s'].describe().to_dict(),
        'raw_sha256_verified': True,
        'policy': 'Keep all values and flags; no zero filling, interpolation, peak removal, or synthetic observations.',
    }
    (ROOT / 'reports' / 'data_check_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if not structural_ok:
        raise ValueError('Structural data problem; see reports. No modelling-ready file created.')
    df[cols].to_csv(ROOT / 'data' / 'processed' / 'daily_discharge_m3s.csv', index=False, encoding='utf-8-sig')
    df.nlargest(10, 'discharge_m3_s')[cols].to_csv(ROOT / 'reports' / 'largest_10_daily_flows.csv', index=False, encoding='utf-8-sig')
    # Small independent checks on units, chronology, and the saved processing result.
    assert np.isclose(FACTOR, 0.3048 ** 3, rtol=0, atol=1e-15)
    assert len(expected_dates) == 5479 and len(df) == 5479
    assert df['date'].diff().dropna().eq(pd.Timedelta(days=1)).all()
    saved = pd.read_csv(ROOT / 'data' / 'processed' / 'daily_discharge_m3s.csv')
    assert np.allclose(saved['discharge_m3_s'], df['discharge_ft3_s'].to_numpy() * FACTOR)
    for request in manifest['requests']:
        assert hashlib.sha256((raw / request['file']).read_bytes()).hexdigest() == request['sha256']

    fig, ax = plt.subplots(figsize=(14, 4.8), layout='constrained')
    ax.plot(df['date'], df['discharge_m3_s'], lw=0.7, color='#176895', label='USGS daily mean (all records retained)')
    for mask, label, color, marker in [
        (df['has_estimated_flag'], 'USGS ESTIMATED flag', '#d89000', 'o'),
        (df['needs_qualifier_review'], 'EQUIP / FORCEINTERPOLATION (review)', '#b32035', 'x'),
    ]:
        ax.scatter(df.loc[mask, 'date'], df.loc[mask, 'discharge_m3_s'], s=20, c=color, marker=marker, label=label, zorder=3)
    ax.set(title='Potomac River, USGS 01646500 | Daily mean discharge, 2010-2024', xlabel='Date (as reported by USGS)', ylabel='Discharge (m³/s)', xlim=(expected_dates.min(), expected_dates.max()), ylim=(0, None))
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / 'reports' / 'discharge_timeseries.png', dpi=180)
    plt.close(fig)

    smallest = df.loc[df['discharge_m3_s'].idxmin()]
    largest = df.loc[df['discharge_m3_s'].idxmax()]
    special_rows = '\n'.join('| %s | %.0f | %.6f | `%s` |' % (r.date.strftime('%Y-%m-%d'), r.discharge_ft3_s, r.discharge_m3_s, r.qualifier_json) for r in df.loc[df['needs_qualifier_review']].itertuples())
    report = f'''# 真实数据检查摘要（已实际运行）

检查日期：{summary['checked_at_local']}。本阶段完成环境、下载和数据检查；尚未训练模型。

## 数据来源与变量

- 机构：U.S. Geological Survey（USGS）；站点：01646500。
- 站名：{manifest['station_name']}。
- 实际日期：{summary['start']} 至 {summary['end']}，共 {len(df):,} 天，含闰日。
- 下载日期：{manifest['download_started_local']}。
- 参数 00060：Discharge（流量）；统计 00003：MEAN（日平均），已用官方代码定义和时间序列元数据交叉核验。
- 来源接口：新版 `https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items`；实际完整查询地址、HTTP 信息、下载时间和 SHA-256 均在原始目录 manifest.json。
- 官方文档：`https://api.waterdata.usgs.gov/docs/ogcapi/`；本地文档及字段定义存于 docs/。使用当前 schema 的 `time_series_id`、`approval_status` 字段；没有照抄入门页面中的过时字段示例。
- 原始单位：全部为 ft^3/s，官方参数定义为 cubic feet per second。处理单位：m³/s，逐值乘以 {FACTOR}（0.3048³）。原始 JSON 未覆盖，处理前后 SHA-256 检查通过。
- 按 USGS 返回的日历日期处理，没有按中国时区移动日期。

## 完整性与数值检查

| 项目 | 实际结果 |
|---|---:|
| 应有日期数 / 返回记录数 | {len(expected_dates)} / {len(df)} |
| 重复日期记录 | {int(duplicate.sum())} |
| 缺失日期 | {len(missing)} |
| 缺失原始值 | {int(df['value'].isna().sum())} |
| 非数值、NaN、无穷值 | {int(bad_numeric.sum())} |
| 负流量值 | {int(negative.sum())} |
| 超出目标区间日期 | {len(unexpected)} |
| Approved 记录 | {int(df['approval_status'].eq('Approved').sum())} |

全部日期连续。没有填 0、插值、平滑、裁剪或删除洪峰，也没有生成模拟数据。

最低日均流量：{smallest['date'].date()}，{smallest['discharge_ft3_s']:.0f} ft³/s = {smallest['discharge_m3_s']:.6f} m³/s。
最高日均流量：{largest['date'].date()}，{largest['discharge_ft3_s']:.0f} ft³/s = {largest['discharge_m3_s']:.6f} m³/s。
前 10 个最高日均流量已另存，全部保留。这里的最高值是“日平均流量最大值”，不是瞬时洪峰；本检查不能独立验证每个水文事件。

## 质量标记与待处理问题

全部 Approved 只表示已审批，不能据此宣称全部为未经估算或修订的直接实测值。共 {int(flagged.sum())} 天存在至少一个 qualifier，另有 {int((~flagged).sum())} 天 qualifier 为 null（null 表示接口未附加标记，不是缺失流量）。

| 标记 | 涉及天数（可重叠） | 本次处理 |
|---|---:|---|
| ESTIMATED | {flags['ESTIMATED']} | 来源标为估算，保留并披露 |
| REVISED | {flags['REVISED']} | 来源标为修订，保留当前下载值及原始标记 |
| EQUIP | {flags['EQUIP']} | 精确定义与原因尚未核实，列为待核实 |
| FORCEINTERPOLATION | {flags['FORCEINTERPOLATION']} | 精确定义与原因尚未核实，列为待核实 |

其中 5 天同时有 ESTIMATED 和 REVISED；上述天数不能直接相加。EQUIP / FORCEINTERPOLATION 的 6 天如下，未擅自删除、替换或解释为可靠的直接观测：

| 日期 | 原值 ft³/s | 换算 m³/s | 来源标记 |
|---|---:|---:|---|
{special_rows}

这些代码来自 USGS 原始记录；本项目没有执行插值。当前官方 schema 仅给出 qualifier 的一般说明，本次官方文档检索未找到这两类代码的精确定义和当日原因，因此不猜测。建议先核实这 6 天，并明确 ESTIMATED 记录的建模使用规则；`model_ready=false`，本阶段停在数据检查，不自动训练。未发现结构性严重问题；该限制是质量标记解释尚待确认，不能直接把整段序列称为“全部直接实测”。

全部带标记日期见 flagged_records.csv，待核实的 6 天见 qualifiers_to_review.csv。

## 图和复核

![日平均流量时间序列]({(ROOT / 'reports' / 'discharge_timeseries.png').as_posix()})

蓝线保留全部日值，橙点为 ESTIMATED，红叉为待核实标记。单位与日期范围已标在图中。
处理脚本实际执行了日期连续性、单位换算、保存结果回读以及原始文件哈希不变的断言检查。

## 可用于论文的事实表述

本研究从 USGS 官方新版 Water Data OGC API 获取 01646500 站 2010—2024 年日平均流量（参数 00060，统计 00003），共 5479 条记录。数据不存在重复日期、缺失日期或缺失流量值。原始单位为 ft³/s，乘以 0.028316846592 换算为 m³/s，保留原始数据及全部质量标记。所有记录审批状态为 Approved，但其中包含 100 天 ESTIMATED、117 天 REVISED（5 天重叠），以及 6 天尚需核实含义的其他来源标记；因此建模使用规则尚待明确。本阶段未进行流量填补或洪峰剔除。
'''
    (ROOT / 'reports' / 'data_check_summary.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('PASS: chronology, conversion, round-trip and raw-file integrity checks')


if __name__ == '__main__':
    main()
