
#mTRFツールボックスと同様のことが、mne.decoding.ReceptiveFieldクラスでできる
#mne.decoding.ReceptiveFieldは、時間遅延を考慮した入力特徴量を使用して、
#エンコーディングモデル（刺激から脳へのモデル）またはデコーディングモデル（脳から刺激へのモデル）を適合させるためのクラス
#これにより、例えばスペクトログラムや時空間的な受容野（STRF）などの時間遅延入力特徴量を使用して、
#脳活動と外部刺激の関係を理解し、予測することができる

# Receptive Field（受容野）で使用する遅延（delay）を定義
tmin, tmax = -0.2, 0.4

# ReceptiveFieldモデルを初期化
#指定した時間範囲（tminからtmaxまで）、サンプリング周波数（sfreq）を持ち、特徴量名とスコアリング方法を設定
#feature_names: モデルの入力特徴量の名前（オプション）。指定しない場合、fitを実行した後に入力データの形状から自動生成
#estimator: モデルの適合に使用する推定器（scikit-learnスタイルのモデル）またはRidge回帰モデルのアルファパラメータ。Noneの場合、Ridge回帰モデル（アルファ=0）が使用される
rf = ReceptiveField(
    tmin, tmax, sfreq, feature_names=["envelope"], estimator=1.0, scoring="corrcoef"
)

# (tmax - tmin) * sfreq の遅延
# 開始/終了インデックスも含むため、追加で2つの遅延がある
n_delays = int((tmax - tmin) * sfreq) + 2

# 交差検証のための分割数を設定し、KFoldクラスを初期化
n_splits = 3
cv = KFold(n_splits)

# モデルのデータを準備。faceデータを転置し、モデルの出力データYを取得。
face = face.T
Y, _ = raw[:]  # Outputs for the model
Y = Y.T

# スプリットごとにモデルを適合させ、予測/テストを繰り返す
coefs = np.zeros((n_splits, n_channels, n_delays))
scores = np.zeros((n_splits, n_channels))
for ii, (train, test) in enumerate(cv.split(face)):
    print("split %s / %s" % (ii + 1, n_splits))
    
    X_train = face[train][:, np.newaxis]  # n_featuresのために新しい軸を追加
    
    # モデルを適合
    rf.fit(X_train, Y[train])
    
    # 同じ形状のテストデータを準備
    X_test = face[test][:, np.newaxis]
    
    # スコアと係数を計算
    scores[ii] = rf.score(X_test, Y[test])
    coefs[ii] = rf.coef_[:, 0, :]

# 遅延の配列を計算
delays = np.linspace(tmin, tmax, n_delays)

times = np.arange(n_delays) * (1.0 / sfreq)

# 交差検証スプリットごとにスコアと係数を平均化
mean_coefs = coefs.mean(axis=0)
mean_scores = scores.mean(axis=0)

# 平均予測スコアをプロット
fig, ax = plt.subplots()
ix_chs = np.arange(n_channels)
ax.plot(ix_chs, mean_scores)
ax.axhline(0, ls="--", color="r")
ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")
mne.viz.tight_layout()
plt.show()

#縦の値は相関係数
#この相関係数は、脳活動と外部刺激がどれだけ同期しているかを示す指標であり、
#高い相関係数は、脳活動が外部刺激に対して敏感であることを示す
