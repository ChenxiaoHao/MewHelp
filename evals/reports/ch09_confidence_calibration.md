# ch09 evidence_confidence 300 题校准报告
- 生成:2026-10-05 03:19 · 起点 03:10 · 应拦面(D)=60 · 应放面=240 · 降级剔除=0
- 目标线:D 漏放 ≤5.0% · 非 D 误拦 ≤6.2%(基线 4.2%+2pp),冲突优先误拦不恶化
- 分布:top1 四分位 [0.2678, 0.8687, 0.9843] · gap 四分位 [0.0262, 0.2409, 0.6329]

## 择优判定式:sum(w=(0.9, 0.05, 0.05),θ=0.168)

| 参数 | 值 |
|---|---|
| form | sum |
| threshold(θ) | 0.168 |
| floor_eff | 0.1 |
| n_eff_min | 1 |
| gap_min | 0.0 |
| w_top1 / w_n / w_gap | 0.9 / 0.05 / 0.05 |

## 实测:D 桶漏放 5.000% · 非 D 误拦 4.167%

判定式=app/rag/confidence.py(与运行期同一核);分数侧车 `evals/cache/confidence_scores.json`;定值回填 `app/core/config.py` evidence_conf_* 键组。RRF 降级样本不在标定域(运行期走 RRF_DEGRADED_MAX 旁路)。
