
# 'times' 配列を 'mean_coefs' の列数に合わせて切り詰める。これは後のグラフの作成に使用される。
#?
times = times[:len(mean_coefs[0])]

# Print mean coefficients across all time delays / channels (see Fig 1)
time_plot = 0.180  # For highlighting a specific time.
fig, ax = plt.subplots(figsize=(4, 8))

# 'mean_coefs' 配列内の最大係数を取得します。
max_coef = mean_coefs.max()

# ヒートマップを作成し、係数を視覚化。'times' はX軸、'ix_chs' はY軸、'mean_coefs' は値。
# 'cmap' はカラーマップ、'vmin' および 'vmax' はカラーマップの値の範囲を指定。
ax.pcolormesh(
    times,
    ix_chs,
    mean_coefs,
    cmap="RdBu_r",
    vmin=-max_coef,
    vmax=max_coef,
    shading="gouraud",
)

ax.axvline(time_plot, ls="--", color="k", lw=2)
ax.set(
    xlabel="Delay (s)",
    ylabel="Channel",
    title="Mean Model\nCoefficients",
    xlim=times[[0, -1]],
    ylim=[len(ix_chs) - 1, 0],
    xticks=np.arange(tmin, tmax + 0.2, 0.2),
)

# X軸の目盛りラベルを45度回転
plt.setp(ax.get_xticklabels(), rotation=45)
mne.viz.tight_layout()

# 'times' 配列内で 'time_plot' に最も近い時間を探し、そのインデックスを 'ix_plot' に格納します。
ix_plot = np.argmin(np.abs(time_plot - times))
fig, ax = plt.subplots()

# "biosemi32" テンプレートを使用して Montage オブジェクト 'easycap_montage' を作成
easycap_montage = mne.channels.make_standard_montage("biosemi32")

# チャンネル名、サンプリング周波数、チャンネルタイプを指定して空の 'info' オブジェクトを作成
info = mne.create_info(ch_names=easycap_montage.ch_names, sfreq=1000.0, ch_types='eeg')

#'info' オブジェクトにモンタージュ情報を設定
info.set_montage(easycap_montage)

# マップを作成し、モデルの係数を視覚化。'mean_coefs' の特定の遅延に対する係数が表示される
# 'pos' はセンサーの位置情報、'axes' はグラフ描画のための軸を指定
# 'show=False' はプロットを直接表示しないように指定
mne.viz.plot_topomap(
    mean_coefs[:, ix_plot], pos=info, axes=ax, show=False, vlim=(-max_coef, max_coef)
)
ax.set(title="Topomap of model coefficients\nfor delay %s" % time_plot)
mne.viz.tight_layout()
plt.show()
