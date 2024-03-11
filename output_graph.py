
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
import pandas as pd

start_times_df = f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/time_list.csv"
start_times = pd.read_csv(start_times_df)

movie_numbers = range(1, 2) # 動画の番号 (1~40)
feature_numbers = range(1, 2) # 特徴量17
subject_numbers = range(1, 2) # 被験者数22人
face_namelist = ["Inner Brow Raiser","Outer Brow Raiser","Brow Lowerer","Upper Lid Raiser","Cheek Raiser","Lid Tightener","Nose Wrinkler","Upper Lip Raiser","Lip Corner Puller","Dimpler","Lip Corner Depressor","Chin Raiser","Lip stretcher","Lip Tightener","Lips part","Jaw Drop","Blink"]

for movie_number in movie_numbers:
    for feature_number in feature_numbers:
        all_raw_data = np.zeros((32, 7681))
        all_face_data = np.zeros(3000)
        face_name = face_namelist[feature_number-1]
        for subject_number in subject_numbers:
            start_time = start_times.iloc[movie_number-1, subject_number] # 「注意」movienumber-1　は固定
            eeg_path = f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/prepro_{subject_number:02d}.fif"
            face_path = f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/result_mix/{subject_number}/out_extract_{subject_number:02d}/extracted_data{subject_number:02d}_{movie_number:02d}.csv"
            raw = mne.io.read_raw_fif(eeg_path, preload=True) #EEGデータの読み込み
            sfreq = raw.info['sfreq']  # サンプリング周波数を取得
            print("sfreq : ", sfreq)
            n_channels = len(raw.ch_names) # 変更！
            # decim = 2  # (任意の変数)
            # sfreq /= decim
            face_data = pd.read_csv(face_path) #faceデータ読み込み
            face = face_data.iloc[:, feature_number].values #指定された列のfaceデータを読み込む(1から17)
            print("face;",face)
            print("face number:",face_namelist[feature_number-1]) # feature_number-1 は固定
            # face = mne.filter.resample(face.astype(float), down=decim, npad="auto") #faceデータのダウンサンプリング
            # raw = raw.copy().resample(sfreq)  # RawArrayをコピーしてリサンプル
            end_time = start_time + 60
            raw.crop(tmin=start_time, tmax=end_time)  # 指定した時間帯のデータを抽出
            # info インスタンスは、MNE-Pythonで使用されるデータに関する情報を保持するためのオブジェクト
            info = mne.create_info(raw.ch_names, sfreq, "eeg") #変更 !
            data = raw.get_data()  # EEGデータを取得
            # data = data[:-1, :]  # 32チャンネル目を除外 変更　！
            print(data.shape)
            all_raw_data += data
            all_face_data += face
        average_raw_data = all_raw_data / 22
        average_face_data = all_face_data / 22
        raw = mne.io.RawArray(average_raw_data, info)  # データ(n_channels, n_times)とinfo(channel名など)を合わせて新しいRawArrayを作成
        face = average_face_data


        tmin, tmax = -0.5, 0.5 # 考慮する遅延時間の範囲を設定
        # ReceptiveField クラスのインスタンスを作成。時間遅れ脳波相関解析を実行するためのもの。
        # 与えられた時間範囲、サンプリング周波数、特徴量の名前、評価値の推定量、スコアリング方法などを指定
        rf = ReceptiveField(tmin, tmax, sfreq, feature_names=["envelope"], estimator=1.0, scoring="corrcoef")
        n_delays = int((tmax - tmin) * sfreq) + 2 # 時間遅れの数を計算
        n_splits = 3 # 交差検証のための分割数を設定し、KFoldクラスを初期化
        cv = KFold(n_splits)
        face = face.T # モデル用にデータを準備。faceデータを転置し
        Y, _ = raw[:] # モデルの出力データ(EEG)Yを取得。
        Y = Y.T

        # faceとEEGの間の線形関係を評価するために、モデルを学習させる
        # スプリットごとにモデルを適合させ、予測/テストを繰り返す
        coefs = np.zeros((n_splits, n_channels, n_delays))  # 係数：モデルが予測を行う際に各遅延がどれだけの重要性を持つか
        scores = np.zeros((n_splits, n_channels))  # 相関係数

        for ii, (train, test) in enumerate(cv.split(face)):
            print("split %s / %s" % (ii + 1, n_splits))
            print("face.shape", face[train].shape)  # faceは元々1次元
            X_train = face[train][:, np.newaxis]  # 多くの機械学習モデルが二次元の入力を想定しているため、元の配列に新しい軸を追加
            rf.fit(X_train, Y[train])
            X_test = face[test][:, np.newaxis]
            scores[ii] = rf.score(X_test, Y[test]) # スコアを保存
            coefs2 = np.zeros((n_splits, n_channels, n_delays-1))
            coefs2[ii] = rf.coef_[:, 0, :]  # モデルの係数を保存
        times = np.linspace(tmin, tmax, n_delays-1) # 遅延のタイミングを計算。np.linspace()は、指定された範囲内で等間隔の数値を生成

        # 交差検証スプリットごとのスコアと係数を平均化 coefは係数、scoreは相関係数
        mean_coefs = coefs2.mean(axis=0)
        mean_scores = scores.mean(axis=0)

        # 各遅延時間に対する処理を行う
        positive_sums = []
        positive_counts = []

        # mean_coefs のデータを元に処理を行う
        # mean_coefs が 32x65 の2次元配列として与えられていると仮定
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
        # max_positive_mean_delay = 0.28
        # 結果を出力します
        print("Delay time with maximum positive mean:", max_positive_mean_delay)

        # 平均予測スコアをプロット
        #fig, ax = plt.subplots() # 新しい図と軸を作成
        ix_chs = np.arange(n_channels) # チャンネルのインデックスを作成
        #ax.plot(ix_chs, mean_scores) # 平均予測スコアをプロット
        #ax.set(title="Mean prediction score", xlabel="Channel", ylabel="Score ($r$)")

        #print("mean_scores.shape", mean_scores.shape)
        # print("mean_scores", mean_scores)
        #ヒートマップ
        #time_plot = max_positive_mean_delay  # 特定の時間をハイライト
        time_plot = -0.1

        fig, ax = plt.subplots(figsize=(4, 8)) #  新しい図と軸を作成
        max_coef = mean_coefs.max()
        # 係数のヒートマップを描画
        ax.pcolormesh(
            times,
            ix_chs,
            mean_coefs,
            cmap="RdBu_r",
            vmin=-max_coef,
            vmax=max_coef,
            shading="gouraud",
        )
        ax.axvline(time_plot, ls="--", color="k", lw=2) # 特定の時間を縦線でハイライト

        # 軸のラベルとタイトルを設定し
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
        plt.savefig(f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/0311/heat_M{movie_number}_{feature_number}.{face_name}.png")

        # topomap
        # 'times' 配列内で 'time_plot' に最も近い時間を探し、そのインデックスを 'ix_plot' に格納
        ix_plot = np.argmin(np.abs(time_plot - times))
        fig, ax = plt.subplots() # 新しい図と軸を作成
        # "biosemi32" テンプレートを使用して Montage オブジェクト 'easycap_montage' を作成
        easycap_montage = mne.channels.make_standard_montage("biosemi32")
        # チャンネル名、サンプリング周波数、チャンネルタイプを指定して空の 'info' オブジェクトを作成
        info = mne.create_info(ch_names=easycap_montage.ch_names, sfreq=128, ch_types='eeg')  # 変更　！
        info.set_montage(easycap_montage)

        # モデル係数のトポグラフィを描画
        mne.viz.plot_topomap(mean_coefs[:, ix_plot], pos=info, axes=ax, show=False, vlim=(-max_coef, max_coef))
        ax.set(title="Topomap of model coefficients\nfor delay %s" % ix_plot)
        plt.tight_layout()
        plt.savefig(f"/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/0311/topo_M{movie_number}_{feature_number}.{face_name}.png")
