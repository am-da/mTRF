### mTRFとは
mTRFは多変量のTRF(forward model)  
(mTRF tool boxはforward modelとbackward modelの両方計算できる)　　

### 正則化

正則化（Regularization）は、機械学習や統計学において、モデルの複雑さを制御して過学習（overfitting）を防ぐための手法  
機械学習モデルの性能向上や安定性の向上に寄与することがあり、特にデータセットが小さい場合や特徴量が多い場合に効果的

### forward model
システムへの入力がその出力にどのように関係するかを数学的に説明  
順方向モデルは、システムが情報を生成または符号化する方法を記述するため、生成モデルまたは符号化モデル  
ここでは、それらを時間応答関数 (TRF)と呼びます  
TRF は、進行中の刺激から進行中の神経反応への線形変換を記述するフィルター
<img width="550" alt="スクリーンショット 2024-01-25 14 29 00" src="https://github.com/am-da/mTRF/assets/112613519/88b3244c-de16-4ee3-9b4b-71c03f5ff3d4">

神経応答 r(t, n)   
刺激特性 s(t)   
未知のチャンネル固有のTRF（Time-Resolved Filter） w(τ, n)  
ε(t, n) はモデルによって説明できない各チャンネルの残留応答

### backward model
多変量コンテキストで利用可能なすべての神経データを活用することによって、逆刺激応答マッピングを導き出す　　
バックワード モデルは、神経応答から刺激特徴をデコードすることでデータ生成プロセスを逆にしようとするため、識別モデルまたはデコード モデルと呼ばれることもあります　　
<img width="375" alt="スクリーンショット 2024-01-25 14 58 52" src="https://github.com/am-da/mTRF/assets/112613519/f1fa9021-9484-433e-9ce8-211ce759e40a">

### global field power (GFP)  
各瞬間の脳の電場の強さを表す



<details><summary>線形時不変 (LTI) システム</summary>
人間の脳は線形でも時間不変でもありませんが、これらの仮定は特定の場合には合理的であり、システムをそのインパルス応答によって特徴付けることができます。
</details>

<details><summary>逆相関法</summary>
逆相関法は視覚神経生理学において、初期視覚ニューロンの受容野位置や受容野時空間構造の定量解析に有効な手法として用いられる。　　 
逆相関法は入力IがシステムSに与えられた時に得られる出力Rを用いて、システムSの入出力関係を表す伝達関数を求めることを目的としている。　　 
これは制御工学においてシステムSのインパルス応答を求めることに相当する。  
システムSのインパルス応答がわかれば、その周波数特性を求めることができる。　　  
また、インパルス応答と入力Iの畳み込み積分を求めることで、任意刺激に対するシステムSの応答が予測可能である。　　   
(任意の刺激もインパルス応答の組み合わせで表現できる)  

逆相関法は、システムの出力R(t)とある時間(τ_i)、過去の入力I(t-τ_i)の相互相関を求めることで、システムSのインパルス応答を求める。
https://www.jstage.jst.go.jp/article/jcss/21/3/21_396/_pdf
</details>


<details><summary>インパルス応答</summary>

<img width="570" alt="スクリーンショット 2024-01-25 13 37 34" src="https://github.com/am-da/mTRF/assets/112613519/b3c71aea-c083-41d0-943e-f5a78ce3d368">

<img width="577" alt="スクリーンショット 2024-01-25 13 38 20" src="https://github.com/am-da/mTRF/assets/112613519/a80a3eb7-56fb-4017-b6f6-c0ee57558116">
<img width="586" alt="スクリーンショット 2024-01-25 13 38 39" src="https://github.com/am-da/mTRF/assets/112613519/7d813fa2-4ca5-44b4-a655-e49f797c6a8c">

インパルス応答を知ることで、システムの性質や動作を理解し、制御システムの設計や解析に役立ちます。　　

インパルス応答が既にわかっているシステムがあったとします。 このシステムに、インパルス以外の信号を入力した場合の出力はいったいどうなるのでしょうか？　　  
その答えは、「畳み込み（Convolution）」という計算方法で求めることができます。
(インパルス応答を基準に。インバルス応答のタイミングと大きさの組み合わせが複数)
<img width="606" alt="スクリーンショット 2024-01-25 13 56 22" src="https://github.com/am-da/mTRF/assets/112613519/fae0598f-43c3-41c4-a30c-80e048c40492">
https://www.noe.co.jp/technology/18/18inv1.html
</details>


<details><summary>フーリエ変換</summary>
<img width="564" alt="スクリーンショット 2024-01-25 14 01 48" src="https://github.com/am-da/mTRF/assets/112613519/121c3608-116e-4c4f-8651-816b757dbe4d">
https://www.yukisako.xyz/entry/fourier-transform
</details>


