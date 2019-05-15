
# coding: utf-8

# 日本語wikiからコーパスを作成するスクリプトです.<br>
# https://dumps.wikimedia.org/jawiki/latest/ <br>
# こちらのサイトから最新版の"pages-articles"のアドレスを手に入れてください. <br>

# ダンプデータには不要なマークアップなどが含まれているので、取り除くためのテキストクリーニング用のスクリプトをgitから持ってきます

# 日本語wikiに対してテキストクリーニングを実行します

# テキストに前処理を加えた上で,複数のtxtファイルをひとつに結合します

# In[6]:

import glob
from bs4 import BeautifulSoup

with open('./data/chatdata/chatdata-sample.txt','w') as f:
#with open('./tmp.txt','w') as f:
    for directory in glob.glob('./extracted/*'):
        for name in glob.glob(directory+'/*'):
            with open(name, 'r') as r:
                for line in r:
                    # titleを削除する
                    if '<doc ' in line:
                        next(r)
                        next(r)
                    elif '</doc>' in line:
                        f.write('\n')
                        continue
                    else:
                        # 空白・改行削除、大文字を小文字に変換
                        text = BeautifulSoup(line.strip()).text.lower()
                        f.write(text)


# ここからはBERTのトレーニング用にテキストファイルを整形していきます.<br>
# 文章を単語ごとに分割し, ひとつの単元の中に偶数個の文章が含まれるように調整します.

# In[7]:

import linecache
import random
import MeCab


# In[20]:

random.seed(42)
#filename = 'tmp.txt'
filename = './data/chatdata/chatdata-sample.txt'
#save_file = 'even_rows100M.txt'
save_file = './data/chatdata/chatdata-sample-after.txt'
LIMIT_BYTE = 100000000 # 100Mbyte
# t = MeCab.Tagger('-Owakati') # Neologdを辞書に使っている人場合はそちらを使用するのがベターです
#t = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/ -Owakati')
t = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/ -Owakati')

def get_byte_num(s):
    return len(s.encode('utf-8'))


# In[31]:

#with open(save_file, 'w') as f:
with open(save_file, 'w') as f:
    count_byte = 0
    with open(filename) as r:
        for text in r:
            print('{} bytes'.format(count_byte))
            #text = t.parse(text).strip()
            # 一文ごとにタブで分割すればいいのではないか
            #text = text.split('。')
            text = text.split('\t')
            #text = text.split()
            # 空白要素は捨てる
            #text = [t.strip() for t in text if t]
            # 一単元の文書が偶数個の文章から成るようにする(BERTのデータセットの都合上)
            max_text_len = len(text) // 2
            text = text[:max_text_len * 2]
            text = '\n'.join(text)
            f.write(text)
            count_byte += get_byte_num(text)
            if count_byte >= LIMIT_BYTE:
                break


# これでBERTの学習に使うデータセットができました.<br>
# 今度はTraining用とValidation用のデータに分割します.

# In[32]:

num_lines = sum(1 for line in open(save_file))
print('Base file lines : ', num_lines)
# 全体の80%をTraining dataに当てます
train_lines = int(num_lines * 0.8)
print('Train file lines : ', train_lines)


# dataは前処理済みテキスト保存場所 <br>
# outputは訓練モデル保存場所として作成

# In[9]:

get_ipython().system(' mkdir -p data output')


# In[33]:

out_file_name_temp = './data/chatdata/splitted_%d.txt'

split_index = 1
line_index = 1
out_file = open(out_file_name_temp % (split_index,), 'w')
in_file = open(save_file)
line = in_file.readline()
while line:
    if line_index > train_lines:
        print('Starting file: %d' % split_index)
        out_file.close()
        split_index = split_index + 1
        line_index = 1
        out_file = open(out_file_name_temp % (split_index,), 'w')
    out_file.write(line)
    line_index = line_index + 1
    line = in_file.readline()
    
out_file.close()
in_file.close()


# In[34]:

print('Train file lines : ', sum(1 for line in open('./data/chatdata/splitted_1.txt')))
print('Valid file lines : ', sum(1 for line in open('./data/chatdata/splitted_2.txt')))


# これにてテキストの前処理は完了です！
