# bert_qa
====

Overview

## Description
絵文字を含む文章で答えを返す、BERTのfine tuningによるQAAnsweringの実装です。

## Demo

## Requirement
・dataフォルダを作成し、その中にchatdataフォルダを入れ、そこにいただいた絵文字の会話データ（chatdata-sample.txt）を入れます</br>
・CUDA環境が必要であるため、まずはCUDA toolkitをインストール</br>
・ライブラリには、torch,beautifulsoup,pickle,tqdm,ipywidgets,MeCabが必要。</br>
・mecab-ipadic-NEologdのインストールが必要</br>
https://qiita.com/spiderx_jp/items/7f8cbfd762c9abab660b</br>

## Version
Ubuntu 18.04 LTS</br>
CUDA compilation tools 9.1.85</br>
cuda toolkit 10.0.130</br>
ipywidgets 7.4.2</br>
ipython 7.4.0</br>
matplotlib 3.0.3</br>
mecab-python3 0.996.1</br>
pytorch 1.0.1</br>
tqdm 4.31.1</br>


## License

元のスクリプトには特にかけられていない様子

## Author

[schunsukesuzuki](https://github.com/schunsukesuzuki)
