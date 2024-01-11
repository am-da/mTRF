https://github.com/am-da/mTRF/blob/main/10_2_25.png?raw=true

```python

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from os.path import join
import mne

from mne.decoding import ReceptiveField
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
import pandas as pd

# 前処理あり
# EEGデータのパス
eeg_path = "/Users/ami/Desktop/UCSD/prepro_10.fif"

# 顔表情特徴量のCSVファイルパス
face_path = "/Users/ami/Desktop/UCSD/result_mix/10/out_extract_10/extracted_data10_02.csv"

# EEGデータの読み込み
raw = mne.io.read_raw_fif(eeg_path, preload=True)
sfreq = raw.info['sfreq'] # サンプリング周波数を取得
n_channels = len(raw.ch_names) # チャンネル数を取得

# 顔表情特徴量の読み込み
face_data = pd.read_csv(face_path)
face = face_data.iloc[:, 15]  # 16列目(Lips part)
print(face)

# ダウンサンプリング
#decim = 2
#raw.resample(sfreq / decim)

# 242.560546875秒後から60秒間のデータを抽出 ()
start_time = 242.560546875
end_time = start_time + 60
raw.crop(tmin=start_time, tmax=end_time) # 指定した時間帯のデータを抽出

# EEGデータのモンタージュ作成
montage = mne.channels.make_standard_montage("biosemi32")

# チャンネル数を取得
n_channels = len(montage.ch_names) # モンタージュのチャンネル数を取得します

# infoを作成
info = mne.create_info(montage.ch_names, sfreq / decim, "eeg")

# RawArrayに渡すデータの長さとチャンネル数を一致させる
data = raw.get_data() # EEGデータを取得
raw = mne.io.RawArray(data[:n_channels, :], info) # データとinfoを合わせて新しいRawArrayを作成

# プロット
fig, ax = plt.subplots()
lns = ax.plot(scale(raw[:, :60][0].T), color="b", alpha=0.2) # EEGデータをプロットします
ln1 = ax.plot(scale(face[:60]), color="r", lw=2)  #顔表情特徴量をプロットします
ax.legend([lns[0], ln1[0]], ["EEG", "face"], frameon=False)
ax.set(title="Sample activity", xlabel="Time (s)")
mne.viz.tight_layout()
plt.show()

# スケーリングは、データの平均が0になり、標準偏差が1になるように変換される
# この変換により、データの単位や分布の差異に関係なく
# 異なるデータソースのデータを比較しやすくなり、グラフにプロットすることができる
```

