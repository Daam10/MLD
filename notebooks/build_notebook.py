from pathlib import Path

import nbformat as nbf

CELLS = [
    ("md", "# Мультимодальные маркеры лжи: исследовательский анализ\n\n"
           "Результаты работы пайплайна загружаются ноутбуком из директорий `data/processed/` и `outputs/`. "
           "Запись девяти сессий была сделана 30.09, а s14 записали 07.10, при этом в него включён ещё и ФПГ пальца."),
    ("code", """import json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from lie_detector.config import sessions, processed_dir, REPO_ROOT
OUT = REPO_ROOT / 'outputs'
sns.set_theme(style='whitegrid')
df = pd.read_csv(OUT / 'dataset.csv')
load = lambda sid, name: json.loads((processed_dir(sid) / name).read_text(encoding='utf-8'))
print(json.loads((OUT / 'summary.json').read_text()))
df[df.y_self.notna()].groupby('sid').y_self.agg(answers='count', lies='sum')"""),

    ("md", "## 1. Синхронизация видео ↔ БиоРадио"),
    ("code", """sync = pd.DataFrame({sid: {'day': s.source, **load(sid, 'sync.json')} for sid, s in sessions().items()}).T
sync['offset_minus_button_s'] = np.where(sync.day == 'day1', sync.offset.astype(float) - sync.coarse_offset.astype(float), np.nan)
sync.drop(columns='offset_before_pulse')"""),

    ("md", "## 2. ЭКГ: качество и реакция на интервью"),
    ("code", """rows = []
for sid in sessions():
    b = load(sid, 'bio_summary.json')
    r = {'sid': sid, **{f'q_{k}': v for k, v in b['ecg_quality'].items()}, 'rest_s': b['pre_rest_s']}
    for ph in ('pre', 'interview'):
        r.update({f'{k}_{ph}': (b[f'hrv_{ph}'] or {}).get(k) for k in ('HR_mean', 'RMSSD', 'SDNN', 'SI_Baevsky', 'LF_HF')})
    rows.append(r)
pd.DataFrame(rows).set_index('sid').round(2)"""),
    ("code", """colors = {'alcohol': 'orange', 'conflict': 'red', 'deception': 'purple'}
def topics(ax, sid):
    for u in load(sid, 'dialog.json'):
        if u['topic'] in colors:
            ax.axvspan(u['start'], u['end'], color=colors[u['topic']], alpha=.05, lw=0)
fig, axes = plt.subplots(2, 5, figsize=(20, 6), sharey=True)
for ax, sid in zip(axes.ravel(), sessions()):
    h = pd.read_csv(processed_dir(sid) / 'hrv_ecg.csv')
    for ph, c in (('pre', 'grey'), ('interview', 'C3'), ('post', 'grey')):
        ax.plot(h.t_video[h.phase == ph], h.HR_mean[h.phase == ph], color=c)
    topics(ax, sid)
    ax.set_title(f'{sid} ({sessions()[sid].scenario})')
axes[0, 0].set_ylabel('ЧСС, уд/мин (окно 120 с)')
plt.tight_layout()"""),

    ("md", "## 3. ЭКГ против контактной ФПГ (s14)\n\nРасчёт ВСР на основе R-зубцов и пиков ФПГ выполняется единым кодом в окнах длиной 120 с при шаге 5 с."),
    ("code", """pd.read_csv(processed_dir('s14') / 'ppg_vs_ecg.csv').round(3)"""),
    ("code", """e, p = (pd.read_csv(processed_dir('s14') / f'hrv_{k}.csv') for k in ('ecg', 'ppg'))
m = e.merge(p, on='t', suffixes=('_ecg', '_ppg')); m = m[m.phase_ecg == 'interview']
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, k in zip(axes, ['HR_mean', 'SDNN', 'RMSSD', 'LF_HF']):
    a, b = m[f'{k}_ecg'], m[f'{k}_ppg']
    diff = b - a
    ax.scatter((a + b) / 2, diff, s=8, alpha=.6)
    for v, st in ((0, '-'), (1.96, '--'), (-1.96, '--')):
        ax.axhline(diff.mean() + v * diff.std(), color='C3', ls=st)
    ax.set_title(f'{k}: ФПГ − ЭКГ')
plt.tight_layout()"""),
    ("code", """print(load('s14', 'bio_summary.json')['ppg'])
pat = pd.read_csv(processed_dir('s14') / 'pat.csv')
pat = pat[(pat.t_video > 0) & (pat.t_video < sync.loc['s14', 'video_duration'])]
fig, ax = plt.subplots(figsize=(14, 3))
ax.plot(pat.t_video, 1000 * pat.pat.rolling(15, center=True).median(), lw=.8)
topics(ax, 's14')
ax.set_ylabel('PAT, мс')"""),

    ("md", "## 4. rPPG против ЭКГ"),
    ("code", """ev = pd.concat([pd.read_csv(processed_dir(s) / 'rppg_eval.csv').assign(sid=s) for s in sessions()])
ev[ev.reference == 'ECG'].pivot_table(index='sid', columns='method', values='MAE').round(1)"""),
    ("code", """ev[ev.reference == 'ECG'].groupby('method')[['MAE', 'within5', 'r']].median().round(2)"""),
    ("md", "### Есть ли пульс в видео: усреднение, синхронное с R-зубцами, против случайных моментов"),
    ("code", """from lie_detector.video import rppg as R
rng, rows, fs = np.random.default_rng(0), [], 25.0
for sid in sessions():
    f = np.load(processed_dir(sid) / 'face_rgb.npz')
    rp = np.load(processed_dir(sid) / 'bio.npz')['rpeaks_s'] - load(sid, 'sync.json')['offset']
    x = R.bandpass(R.pos(R.fill_gaps(R.combine_rois(f['rgb'], f['counts'])), fs), fs, .7, 4)
    x /= x.std()
    def avg(beats):
        idx = np.round(beats * fs).astype(int)
        idx = idx[(idx > 8) & (idx < len(x) - 25)]
        return np.stack([x[i - 8:i + 25] for i in idx]).mean(0)
    obs = avg(rp)
    null = np.array([np.ptp(avg(np.sort(rng.uniform(1, len(x) / fs - 2, len(rp))))) for _ in range(100)])
    rows.append({'sid': sid, 'p2p': np.ptp(obs), 'null_95': np.percentile(null, 95), 'p': (1 + np.sum(null >= np.ptp(obs))) / 101})
pd.DataFrame(rows).set_index('sid').round(3)"""),
    ("code", """rr = pd.concat([pd.read_csv(processed_dir(s) / 'resp_rate.csv') for s in sessions()])
rr = rr[rr.phase == 'interview']
print('пояс vs EDR: MAE %.1f вд/мин, r=%.2f' % ((rr.rate_belt - rr.rate_edr).abs().mean(), rr[['rate_belt', 'rate_edr']].corr().iloc[0, 1]))"""),

    ("md", "## 5. Сценарий против самоотчёта: доля ответов, отмеченных как ложь"),
    ("code", """piv = pd.read_csv(OUT / 'label_agreement.csv').pivot_table(index='sid', columns='condition', values='share_self_lie')
sns.heatmap(piv, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1);"""),

    ("md", "## 6. Маркеры лжи (самоотчёт)\n\nНормирование признаков проводилось для каждого человека индивидуально на основе его ответов, полученных во вступлении."),
    ("code", """st = pd.read_csv(OUT / 'stats_self.csv')
st.head(15)[['feature', 'median_diff', 'effect_rb', 'wilcoxon_p', 'wilcoxon_q', 'n_subjects']].round(3)"""),
    ("code", """pd.read_csv(OUT / 'confirmatory.csv').round(3)"""),
    ("code", """from lie_detector.analysis.dataset import feature_columns
from lie_detector.analysis.stats import preprocess_features
norm = preprocess_features(df, feature_columns(df[df.in_study]))
sel = norm[norm.y_self.notna() & norm.in_study]
top = st.dropna(subset=['wilcoxon_p']).head(8).feature.tolist()
subj = sel.melt(id_vars=['sid', 'y_self'], value_vars=top).groupby(['sid', 'y_self', 'variable']).value.median().reset_index()
subj['метка'] = subj.y_self.map({0: 'правда', 1: 'ложь'})
sns.catplot(subj, x='метка', y='value', col='variable', col_wrap=4, kind='point', hue='sid', legend=False,
            order=['правда', 'ложь'], sharey=False, height=3, palette='tab10').set_ylabels('z');"""),

    ("md", "## 7. Темперамент"),
    ("code", """df.drop_duplicates('sid').set_index('sid')[['temperament', 'style_natural', 'style_strength', 'emo_balance']]"""),
    ("code", """mm = pd.read_csv(OUT / 'mixed_models.csv')
mm.sort_values('p_interaction').head(10).round(4)"""),

    ("md", "## 8. Классификация (leave-one-subject-out)"),
    ("code", """pd.read_csv(OUT / 'models.csv')[['labels', 'model', 'features', 'auc', 'bal_acc', 'auc_per_subject_mean', 'n_answers', 'n_lies', 'perm_p']].round(3)"""),
    ("code", """pd.read_csv(OUT / 'importance.csv', index_col=0).auc_drop.head(15)[::-1].plot.barh(figsize=(7, 5))
plt.xlabel('падение AUC при перестановке признака');"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("01_research_analysis.ipynb"))
