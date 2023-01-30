# 【機械学習】賃貸サイトからお得な賃貸を機械学習で選定する
 
賃貸サイトから賃貸情報を取得し、機械学習でお得な賃貸を選定するシリーズ。ここで、"お得な賃貸"とは実際価格が機械学習で計算された予測価格よりも低い場合を指します。
今回は、【Webスクレイピングと前処理】編で作成したデータを用いて、機械学習に入力します。

【Webスクレイピングと前処理】でデータを取得するコードは次のリポジトリです。
https://github.com/OkamotoDaiki/Rent_PreprocessingMachineLearning

今回はSUUMOから情報を取得しますが、あくまで私的利用であることをご理解ください。また、利用する場合は以下のSUUMOの利用規約の確認を推奨します。
https://cdn.p.recruit.co.jp/terms/suu-t-1003/index.html

# DEMO
 
機械学習、今回はsckit-learnのRandomForestで得られた予測価格と利用した特徴量のデータの一部分を次に示します。
**予測価格**の列が実際にRandomForestが算出した価格、**test-predict**が実際の価格から予測価格を引いた値です。test-predict < 0になっていればお得な物件というのがわかります。最後にratio[%]は(test-predict)/(実際の価格)×100 の値です。この値が高ければ高いほどお得です。

|マンション名|都|区|区域|間取り|間取りDK|間取りK|間取りL|間取りS|築年数|建物高さ|階|専有面積|賃料+管理費|敷/礼|路線1|駅1|徒歩1|賃料|管理費|敷金|礼金|予測価格|test-predict|ratio[%]|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|ＪＲ総武線御茶ノ水駅10階建築24年|東京都|千代田区|外神田２|1|1|0|1|0|24|10|4|36|145000|290000|ＪＲ総武線|御茶ノ水駅|4|145000|0|145000|145000|145000|0|0.0|
|ＪＲ中央線水道橋駅10階建築8年|東京都|千代田区|神田三崎町２|1|0|0|0|0|8|10|6|25|121000|109000|ＪＲ中央線|水道橋駅|2|109000|12000|0|109000|121000|0|0.0|
|東京メトロ日比谷線小伝馬町駅8階建築24年|東京都|千代田区|岩本町１|3|1|0|1|0|24|8|7|90|278300|542600|東京メトロ日比谷線|小伝馬町駅|4|271300|7000|271300|271300|290000|-11700|4.2|
|ハイツ神田岩本町|東京都|千代田区|東神田１|1|0|0|0|0|42|11|11|22|70000|128000|ＪＲ総武線快速|馬喰町駅|3|64000|6000|64000|64000|95000|-25000|35.7|
|パトリア九段下|東京都|千代田区|九段北１|1|0|0|0|0|17|14|2|25|110000|100000|東京メトロ東西線|九段下駅|1|100000|10000|100000|0|110000|0|0.0|
 
# Features
 
特徴量は間取り、築年数、建物高さ、階、専有面積、敷/礼です。
 
# Requirement
 
* python3 3.8.10
* numpy, pandas, sckit-learn
 
# Usage
 
dataのディレクトリに前処理されたデータを補完し、run.shを編集します。
次に、run.shを実行します
 
```
bash run.sh
```

# Author
 
* Oka.D.
* okamotoschool2018@gmail.com
 
# License
[MIT license](https://en.wikipedia.org/wiki/MIT_License).