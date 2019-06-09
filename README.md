# bert_qa
====

Overview

## Description
絵文字を含む文章で答えを返す、BERTのfine tuningによるQAAnsweringの実装です。

## Dockerfile
pytorch公式とビルドの仕方は全く同じです。ただしビルドにはかなり(手元の環境では1h弱ほど)時間がかかります。</br>
https://github.com/pytorch/pytorch#docker-image</br>
加えて現状tqdm,ipywidgets,matplotlibについては実行することがまだできていないため、別途ビルド後入れる必要があります。</br>


## Requirement
・dataフォルダを作成し、その中にchatdataフォルダを入れ、そこにいただいた絵文字の会話データ（chatdata-sample.txt）を入れます</br>
・CUDA環境が必要であるため、まずはCUDA toolkitをインストール</br>
・ライブラリには、torch,beautifulsoup,pickle,tqdm,ipywidgets,MeCabが必要。</br>
・mecab-ipadic-NEologdのインストールが必要</br>
https://qiita.com/spiderx_jp/items/7f8cbfd762c9abab660b</br>

##boot
・dockerの起動は、
 docker run -it -p 8888:8888 [image名]
 で大丈夫なはず（帰宅後確認します）
・jupyter notebookは、
 jupyter notebook --allow-root
 で起動してください。またこれにあたり、jupyter configのファイルで、起動リンクをlocalhostから0.0.0.0に変更してください。
 http://rf00.hatenablog.com/entry/2019/01/03/180656
 (RUNで変更できるようなので後で確認します）

## Version
Ubuntu 18.04 LTS</br>
CUDA compilation tools 9.1.85</br>
cuda toolkit 10.0.130</br>
ipywidgets 7.4.2</br>
ipython 7.4.0</br>
matplotlib 3.0.3</br>
mecab</br>
libmecab-dev</br>
mecab-ipadic-utf8</br>
mecab-python3 0.996.1</br>
pytorch 1.0.1</br>
tqdm 4.31.1</br>
beautifulsoup4</br>
language-pack-ja-base</br> 
language-pack-ja</br>
</br>
ロケールの設定は以下を参照のこと</br>
https://www.t3a.jp/blog/infrastructure/ubuntu-text-garbled/</br>
Docker: コンテナのlocaleを設定したい https://qiita.com/suin/items/856bf782d0d295352e51</br>

Ubuntu 18.04へのCUDA10インストール方法 https://qiita.com/yukoba/items/4733e8602fa4acabcc35 

## License

元のスクリプトには特にかけられていない様子

## Author

[schunsukesuzuki](https://github.com/schunsukesuzuki)
