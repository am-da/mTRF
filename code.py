import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join
import mne

from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
import pandas as pd

# エクセルファイルからstart_timeを読み込む
start_times_df = f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/time_list.csv"
start_times = pd.read_csv(start_times_df)

movie_numbers = range(1, 2)  # 動画の番号 (1~40)
feature_numbers = range(1, 18)  # 特徴量17
subject_numbers = range(1, 23)  # 被験者数22人

for movie_number in movie_numbers:
    for feature_number in feature_numbers:
        all_raw_data = np.zeros((30, 1921))  # 全ての被験者のデータを格納するための空の配列を作成
        all_face_data = np.zeros(1500)  # 全ての被験者の顔データを格納するための空の配列を作成
        for subject_number in subject_numbers:
            print(start_times.iloc[movie_number, subject_number])
            start_time = start_times.iloc[movie_number, subject_number]  # [movie, subject] (movie number-1を記入)
            eeg_path = f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/prepro_{subject_number:02d}.fif"
            face_path = f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/result_mix/{subject_number}/out_extract_{subject_number:02d}/extracted_data{subject_number:02d}_{movie_number:02d}.csv"
            raw = mne.io.read_raw_fif(eeg_path, preload=True)
            sfreq = raw.info['sfreq']  # サンプリング周波数を取得
            n_channels = len(raw.ch_names) - 2  # チャンネル数を取得 (32 -> 30)
            decim = 2  # (任意の変数)
            sfreq /= decim
            face_data = pd.read_csv(face_path)
            face = face_data.iloc[:, feature_number].values  # (1から17)
            face = mne.filter.resample(face.astype(float), down=decim, npad="auto")
            raw = raw.copy().resample(sfreq / decim)  # RawArrayをコピーしてリサンプル
            end_time = start_time + 60
            raw.crop(tmin=start_time, tmax=end_time)  # 指定した時間帯のデータを抽出
            info = mne.create_info(ch_names=[name for idx, name in enumerate(raw.ch_names) if idx not in [27, 31]], sfreq=sfreq, ch_types='eeg')
            data = raw.get_data()  # EEGデータを取得
            data = np.delete(data, [27, 31], axis=0)  # 32チャンネル目と28チャンネル目を除外
            print(data.shape)
            all_raw_data += data
            all_face_data += face
        average_raw_data = all_raw_data / 22
        average_face_data = all_face_data / 22
        raw = mne.io.RawArray(average_raw_data, info)  # データとinfoを合わせて新しいRawArrayを作成
        face = average_face_data
        tmin, tmax = -0.5, 0.5
        rf = ReceptiveField(tmin, tmax, sfreq, feature_names=["envelope"], estimator=1.0, scoring="corrcoef")
        n_delays = int((tmax - tmin) * sfreq) + 2
        # 交差検証のための分割数を設定し、KFoldクラスを初期化
        n_splits = 3
        cv = KFold(n_splits)
        # モデルようにデータを準備。faceデータを転置し、モデルの出力データ(EEG)Yを取得。
        face = face.T
        Y, _ = raw[:]
        Y = Y.T
        # 特徴量とEEGの間の線形関係を評価するために、モデルを学習させる
        # スプリットごとにモデルを適合させ、予測/テストを繰り返す
        coefs = np.zeros((n_splits, n_channels, n_delays))
        scores = np.zeros((n_splits, n_channels))
        for ii, (train, test) in enumerate(cv.split(face)):
            print("split %s / %s" % (ii + 1, n_splits))
            X_train = face[train][:, np.newaxis]  # n_featuresのために新しい軸を追加
            rf.fit(X_train, Y[train])
            X_test = face[test][:, np.newaxis]
            scores[ii] = rf.score(X_test, Y[test])
            coefs2 = np.zeros((n_splits, n_channels, n_delays - 1))
            coefs2[ii] = rf.coef_[:, 0, :]
        mean_scores = scores.mean(axis=0)
        times = np.linspace(tmin, tmax, n_delays - 1)
        # times = np.arange(n_delays) * (1.0 / sfreq)
        # 交差検証スプリットごとのスコアと係数を平均化 coefは係数、scoreは相関係数
        mean_coefs = coefs2.mean(axis=0)
        mean_scores = scores.mean(axis=0)
        # 各遅延時間に対する処理を行います
        positive_sums = []
        positive_counts = []
        # mean_coefs のデータを元に処理を行います
        # mean_coefs が 30x65 の2次元配列として与えられていると仮定します
        # 各遅延時間に対してループを行います
        for i in range(mean_coefs.shape[1]):
            # 各遅延時間における正の値のみを抽出して合計します
            positive_sum = np.sum(mean_coefs[:, i][mean_coefs[:, i] > 0])
            positive_sums.append(positive_sum)
            # 各遅延時間における正の値の個数を数えます
            positive_count = np.sum(mean_coefs[:, i] > 0)
            positive_counts.append(positive_count)
        # 正の値の平均を計算します
        positive_means = [positive_sum / positive_count if positive_count > 0 else 0 for positive_sum, positive_count in zip(positive_sums, positive_counts)]
        # 最も正の平均値が大きい遅延時間を見つけます
        max_positive_mean_index = np.argmax(positive_means)
        max_positive_mean_delay = times[max_positive_mean_index]
        # 結果を出力します
        print("Delay time with maximum positive mean:", max_positive_mean_delay)
        # 平均予測スコアをプロット
        fig, ax = plt.subplots()
        ix_chs = np.arange(n_channels)
        ax.plot(ix_chs, mean_scores)
        ax.axhline(0, ls="--", color="r")
        ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")
        # plt.tight_layout()
        # Print mean coefficients across all time delays / channels (see Fig 1)
        time_plot = max_positive_mean_delay  # For highlighting a specific time.
        fig, ax = plt.subplots(figsize=(4, 8))
        max_coef = mean_coefs.max()
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
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/except32_28/heatmap_{movie_number}_{feature_number}.png")
        # 'times' 配列内で 'time_plot' に最も近い時間を探し、そのインデックスを 'ix_plot' に格納します。
        ix_plot = np.argmin(np.abs(time_plot - times))
        fig, ax = plt.subplots()
        # "biosemi32" テンプレートを使用して Montage オブジェクト 'easycap_montage' を作成
        easycap_montage = mne.channels.make_standard_montage("biosemi32")
        # チャンネル名、サンプリング周波数、チャンネルタイプを指定して空の 'info' オブジェクトを作成
        info = mne.create_info(ch_names=[name for idx, name in enumerate(easycap_montage.ch_names) if idx not in [27, 31]], sfreq=128, ch_types='eeg')
        # print("ch_names")
        # print(easycap_montage.ch_names)
        info.set_montage(easycap_montage)
        mne.viz.plot_topomap(mean_coefs[:, ix_plot], pos=info, axes=ax, show=False, vlim=(-max_coef, max_coef))
        ax.set(title="Topomap of model coefficients\nfor delay %s" % time_plot)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/except32_28/topomap_{movie_number}_{feature_number}.png")
