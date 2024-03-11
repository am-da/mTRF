# mTRF
investigated face features and EEG using DEAP data.  
・subjects:32 (22)  
・movies: 40  
・facial features: 17  

*Preprocessing.py　　  
EEGの前処理

*Load the data.py  
EEGと顔表情特徴量を読み込み、グラフで表示  

*Investigate model coefficients.py  
EEGと特徴量の相関を計算  

*Create and fit a receptive field model.py  
EEGと特徴量の関係をmapに表示(遅延時間を考慮)  

*extract time of mv start.py  
各被験者において、MVが始まる時間を抽出する  

*movie_variance_graph.pdf  
動画ごとに、特徴量の分散を表示したグラフ  

*moviegraph  
各被験者の特徴量の分散を動画ごとにまとめている (特徴量17 * 被験者22人　が　40ファイル)  
各被験者でどの特徴量が顕著に表れているのかを確認できる  

*facial features (17)  
(https://www.cs.cmu.edu/~face/facs.htm)

1 Inner Brow Raiser  
2 Outer Brow Raiser  
4 Brow Lowerer  
5 Upper Lid Raiser  
6 Cheek Raiser  
7 Lid Tightener  
9 Nose Wrinkler  
10 Upper Lip Raiser  
12 Lip Corner Puller  
14 Dimpler  
15 Lip Corner Depressor  
17 Chin Raiser  
20 Lip stretcher  
23 Lip Tightener  
25 Lips part**  
26 Jaw Drop  
45 Blink
