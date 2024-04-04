
# サンプリングを256Hzに変更(最終的には128Hzになる)

import mne
import numpy as np

subject_numbers = range(1, 10)  # 被験者数22人
for subject_number in subject_numbers:
    raw = mne.io.read_raw_bdf(f'/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/DEAP_data/data_original/s{subject_number:02d}.bdf',preload=True)

    # 脳波のチャンネルのインデックスを指定
    brain_channels = list(range(0, 32))

    # 脳波のチャンネルのみを選択してデータを作成
    raw_brain = raw.copy().pick_channels([raw.ch_names[i] for i in brain_channels])
    print(raw_brain.ch_names)

    raw_brain.notch_filter(np.arange(50, 251, 50), filter_length='auto')

    # デジタルフィルタリング
    raw_brain.filter(1, 50, fir_design='firwin')

    # ダウンサンプリング（256Hzにダウンサンプリング）
    raw_brain.resample(256)

    # 平均リファレンスを適用
    raw_brain.set_eeg_reference('average', projection=True)
    raw_brain.apply_proj()

    output_path = f'/Users/ami/PycharmProjects/UCSD_pycharm/UCSD/new_0404/prepro_{subject_number:02d}.fif'  # 保存先のファイルパスを指定
    raw_brain.save(output_path, overwrite=True)  # ファイルを上書き保存
