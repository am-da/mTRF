import mne

# bdfファイルのパス
bdf_file_path = '/Users/ami/Desktop/UCSD/DEAP_data/data_original/s10.bdf'

# bdfファイルを読み込む
raw = mne.io.read_raw_bdf(bdf_file_path, preload=True)

# Statusチャンネルの値が4になる回数をカウントする
status_events = mne.find_events(raw, stim_channel='Status')

# Statusチャンネルの値が4になる時間を出力する
for event in status_events:
    if event[2] == 4:
        event_time = raw.times[event[0]]
        print("Statusチャンネルの値が4になる時間:", event_time)
