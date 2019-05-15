
# coding: utf-8

# 必要ModuleをImport

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import math

import pickle
import tqdm
from collections import Counter

from torch.utils.data import Dataset
import random
import numpy as np

from utils import GELU, PositionwiseFeedForward, LayerNorm, SublayerConnection, LayerNorm

import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from ipywidgets import FloatProgress
from IPython.display import display, clear_output


# In[2]:

input_train_txt = './data/chatdata/splitted_1.txt'
input_valid_txt = './data/chatdata/splitted_2.txt'
processed_train_txt = './data/chatdata/train_X.txt'
processed_valid_txt = './data/chatdata/valid_X.txt'


# Next Sentence Predictionのために, 意味的に連続する文章をtab区切りで並べる前処理をデータセットに対して行います.

# In[3]:

# 偶数行の文章を奇数行の文章と接続するメソッド
def load_data(path):
    with open(path, encoding='utf-8') as f:
        even_rows = []
        odd_rows = []
        all_f = f.readlines()
        for row in all_f[2::2]:
            even_rows.append(row.strip().replace('\n', ''))
        for row in all_f[1::2]:
            odd_rows.append(row.strip().replace('\n', ''))
    min_rows_len = int(min(len(even_rows), len(odd_rows)))
    even_rows = even_rows[:min_rows_len]
    odd_rows = odd_rows[:min_rows_len]

    concat_rows = []
    for even_r, odd_r in zip(even_rows, odd_rows):
        concat_r = '\t'.join([even_r, odd_r])
        concat_rows.append(concat_r)
    return concat_rows


# In[4]:

train_data = load_data(input_train_txt)
valid_data = load_data(input_valid_txt)

# ランダムに並び替える
random.shuffle(train_data)
random.shuffle(valid_data)


# In[5]:

with open(processed_train_txt, 'w') as f:
    f.write('\n'.join(train_data))


# In[6]:

with open(processed_valid_txt, 'w') as f:
    f.write('\n'.join(valid_data))


# Attentionセルを定義する

# In[7]:

class Attention(nn.Module):
    """
    Scaled Dot Product Attention(縮小付き内積注意)
    (参考) http://deeplearning.hatenablog.com/entry/transformer
    
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        
        """
        Product Attentionとvalueの内積を取る(内積注意)
        """

        return torch.matmul(p_attn, value), p_attn


# Multi Head Attentionを定義する

# In[8]:

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        
        """
        d_modelはモデル全体の次元数(largeだと512).head(=h)の数で等分される
        次元数はvalueとkeyで常に同数
        # We assume d_v always equals d_k
        """

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        
        """
        nn.Linear(入力する特徴量のsize,出力〜)
        https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        """
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        
        """
        Attentionクラスの読み込み
        """
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        queryがバッチサイズになる。マスク無し。
        """
        batch_size = query.size(0)
        
        """
        l(x)の表記の仕方がよくわからない。
        .viewでtensorのサイズを調整。-1ですべての要素が横に並ぶ。
        """
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]

        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        
        """
        contiguousはメモリの別々の部分に格納されている情報を統合するためにつかう。
        """
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


# Transformerを定義する

# In[9]:

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        """
        MultiHeadedAttentionクラスを自己注意としてインスタンス化
        """
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """
        公開されていないメソッドやインスタンス変数にだけ、アンダースコアを先頭に付けてください。 
        https://teratail.com/questions/41277
        query,key,valueへの3つの自己注意に対応しているのだろうか>_x, _x, _x, 
        """
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


# BERTクラスを定義する

# In[10]:

"""
googleのスクリプトの、BertPretrainModel(nn.module)にあたる
"""

class BERT(nn.Module):

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        
        """
        通常feed forwardの隠れ層は4倍
        """
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        
        """
        ・レイヤーをリストで保持してしまうと、、リストで保持しているレイヤーのパラメータは
        パラメータとして認識されず呼び出されない。そういうときはModulelistを使う。
        https://qiita.com/perrying/items/857df46bb6cdc3047bd8
        TransformerBlockクラスをModulelistを使ってインスタンス化
        """
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # xの中で0以上は1, 0未満は0として, maskテンソルを作る
        """
        dim (int) – the index at which to insert the singleton dimension
        https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        （予想）x>0のとき、値が１のシングルトンの次元をまず生成。
        次に1度だけ、最初に作った値が１のシングルトンの次元の数だけ横に並べる。
        もういちどそれを値１のシングルトンの次元を生成。
        （よくわからない）
        """
        
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x
    
    """
    1.できればapexのインストール
    2.FusedLayerNormをBertLayerFormとしてimport
    3.from apex.normaliztion.used_layer_norm import FusedLayerNorm as BertLayerNorm
    4.(497)BerPreTrainModel以下にある、init_bert_weights メソッドを、BERTクラス(ココ)に追加
    """
    #from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
    
    def init_bert_weights(self,module):
      #initialize the weights.
    
      if isinstance(module,(nn.Linear,nn.Embedding)):
          #cf https://github.com/pytorch/pytorch/pull/5617
          """
          (148) initializer_range=0.02):
          module.weight.data.normal_(mean=0.0, std=0.02に→self.config.initializer_range)
          """
            
          module.weight.data.normal_(mean=0.0, std=0.02)
           
       #elif isinstance(module, BertLayerNorm)
      elif isinstance(module, LayerNorm):
          module.bias.data.zero_()
          module.weight.data.fill_(1.0)
      if isinstance(module, nn.Linear)and module.bias is not None:
          module.bias.data.zero_()
    
    
    
    


# In[11]:

"""
ここに、
Class BertForQuestionAnswering(BERT):
を入れるのがいいかもしれない。
"""

class BertForQuestionAnswering(BERT):
    
   #def __init__(selfトル→,config):
   def __init__(self):
     #super(BertForQuestionAnswering, self).__init__(トル→config)
     super(BertForQuestionAnswering, self).__init__()
     #self.bert = BERT→BertModel(トル→config)
     self.bert = BERT()
     #self.qa_outputs = nn.Linear(hidden→config.hidden_size, 2)
     self.qa_outputs = nn.Linear(hidden, 2)
     self.apply(self.init_bert_weights)   
   
   def forward(self,input_ids,token_type_ids=None,attention_mask=None,start_positions=None,end_positions=None):
     sequence_output,_ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
     logits = self.qa_outputs(sequence_output)
     start_logits, end_logits = logits.split(1, dim=-1)
     start_logits = start_logits.squeeze(-1)
     end_logits = end_logits.squeeze(-1)
    
     if start_positions is not None and end_positions is not None:
         #if we are on multi-GPU, split add a dimension
         if len(start_positions.size()) > 1:
             start_positions = start_positions.squeeze(-1)
         if len(end_positions.size()) > 1:
             end_positions = end_positions.squeeze(-1)
         #sometimes the start/end posisions are outside our model inputs, we ignore these terms
         ignored_index = start_logits.size(1)
         start_positions.clamp_(0, ignored_index)
         end_positions.clamp_(0,ignored_index)
            
         loss_fct = CrossEntropyLoss(ignore_index = ignored_index)
         start_loss = loss_fct(start_logits, start_positions)
         end_loss = loss_fct(end_logits, end_positions)
         total = (start_loss + end_loss) / 2
         return total_loss
     
     else:
         return start_logits, end_logits
     


# BERTのEmbedding層を定義する

# In[12]:

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float().exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : 通常のEMbedding
        2. PositionalEmbedding : sin, cosを用いた位置情報付きEmbedding
        2. SegmentEmbedding : Sentenceのセグメント情報 (sent_A:1, sent_B:2)
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)


# 学習用にマスク予測・隣接文予測の層を追加する

# In[13]:

class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2クラス分類問題 : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    入力系列のMASKトークンから元の単語を予測する
    nクラス分類問題, nクラス : vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


# BERT用のVocabを生成するクラスを定義する

# In[14]:

import pickle
import tqdm
from collections import Counter


class TorchVocab(object):
    """
    :property freqs: collections.Counter, コーパス中の単語の出現頻度を保持するオブジェクト
    :property stoi: collections.defaultdict, string → id の対応を示す辞書
    :property itos: collections.defaultdict, id → string の対応を示す辞書
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<oov>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """
        :param coutenr: collections.Counter, データ中に含まれる単語の頻度を計測するためのcounter
        :param max_size: int, vocabularyの最大のサイズ. Noneの場合は最大値なし. defaultはNone
        :param min_freq: int, vocabulary中の単語の最低出現頻度. この数以下の出現回数の単語はvocabularyに加えられない.
        :param specials: list of str, vocabularyにあらかじめ登録するtoken
        :param vecors: list of vectors, 事前学習済みのベクトル. ex)Vocab.load_vectors
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # special tokensの出現頻度はvocabulary作成の際にカウントされない
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # まず頻度でソートし、次に文字順で並び替える
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
        
        # 出現頻度がmin_freq未満のものはvocabに加えない
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # dictのk,vをいれかえてstoiを作成する
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        super().__init__(counter, specials=["<pad>", "<unk>", "<eos>", "<sos>", "<mask>"], max_size=max_size, min_freq=min_freq)

    # override用
    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    # override用
    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


# テキストファイルからvocabを作成する
class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        print("Building Vocab")
        counter = Counter()
        for line in texts:
            if isinstance(line, list):
                words = line
            else:
                words = line.replace("\n", "").replace("\t", "").split()

            for word in words:
                counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]  # this would be index 1
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build(corpus_path, output_path, vocab_size=None, encoding='utf-8', min_freq=1):
    with open(corpus_path, "r", encoding=encoding) as f:
        vocab = WordVocab(f, max_size=vocab_size, min_freq=min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(output_path)


# Dataloaderを定義する.
# ここで文章中の単語をMASKする処理と,隣り合う文章を一定確率でシャッフルする処理を同時に行う

# In[15]:

class BERTDataset(Dataset):
    """
    このクラスで訓練モード化どうかを指定する項目を初期化
    torch.utils.data.Dataset （抽象クラス）
    __len__,__getitem__ ともにDatasetクラスでは必要なメソッド
    """
    def __init__(self, corpus_path, vocab, seq_len, label_path='None', encoding="utf-8", corpus_lines=None, is_train=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.is_train = is_train

        with open(corpus_path, "r", encoding=encoding) as f:
            self.datas = [line[:-1].split("\t") for line in f]
        if label_path:
            self.labels_data = torch.LongTensor(np.loadtxt(label_path))
        else:
            # ラベル不要の時はダミーデータを埋め込む
            self.labels_data = [0 for _ in range(len(self.datas))]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        t1, (t2, is_next_label) = self.datas[item][0], self.random_sent(item)
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)
        labels = self.labels_data[item]

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t2 = t2_random + [self.vocab.eos_index]

        t1_label = [self.vocab.pad_index] + t1_label + [self.vocab.pad_index]
        t2_label = t2_label + [self.vocab.pad_index]

        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label,
                  "labels": labels}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []

        for i, token in enumerate(tokens):
            if self.is_train: # Trainingの時は確率的にMASKする
                prob = random.random()
            else:  # Predictionの時はMASKをしない
                prob = 1.0
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(0)

        return tokens, output_label

    def random_sent(self, index):
        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return self.datas[index][1], 1
        else:
            return self.datas[random.randrange(len(self.datas))][1], 0


# Trainerクラスを定義する.
# BERTの事前学習ではふたつの言語モデル学習を行う.
# 1. Masked Language Model : 文章中の一部の単語をマスクして,予測を行うタスク.
# 2. Next Sentence prediction : ある文章の次に来る文章を予測するタスク.

# In[16]:

class BERTTrainer:
    #def __init__(self, bert: BERT, vocab_size: int,
    def __init__(self, bert: BertForQuestionAnswering, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 with_cuda: bool = True, log_freq: int = 10):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logを表示するiterationの頻度
        """

        # GPU環境において、GPUを指定しているかのフラグ
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        # masked_token予測のためのLoss関数を設定
        self.criterion = nn.NLLLoss()
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        self.train_lossses = []
        self.train_accs = []

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        :param epoch: 現在のepoch
        :param data_loader: torch.utils.data.DataLoader
        :param train: trainかtestかのbool値
        """
        str_code = "train" if train else "test"

        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")


        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_dataはGPU or CPUに載せる
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the next_sentence_prediction and masked_lm model
            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2-1. NLLLoss(negative log likelihood) : next_sentence_predictionのLoss
            next_loss = self.criterion(next_sent_output, data["is_next"])

            # 2-2. NLLLoss(negative log likelihood) : predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            # 2-3. next_lossとmask_lossの合計をlossとする
            loss = next_loss + mask_loss

            # 3. training時のみ,backwardとoptimizer更新を行う
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=", total_correct * 100.0 / total_element)
        self.train_lossses.append(avg_loss / len(data_iter))
        self.train_accs.append(total_correct * 100.0 / total_element)
        
    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    """
    loadをここに書く場合、こんな感じかもしれない
    
    def load(self, file_path="output/bert_trained.model"):
        pretrained_path = file_path
        torch.load(self.bert.cpu(), pretrained_path)
        self.bert.to(self.device)
        print("Model loaded on:" ,pretrained_path)
        return pretrained_path
    # ココまでで終わり
    """

    
    
        


# In[17]:

import datetime
dt_now = str(datetime.datetime.now()).replace(' ', '')


# In[18]:

# 訓練用パラメタを定義する
train_dataset=processed_train_txt
test_dataset=processed_valid_txt
vocab_path='./data/vocab'+ dt_now +'.txt'
output_model_path='./output/bertmodel'+ dt_now

hidden=256 #768
layers=8 #12
attn_heads=8 #12
seq_len=60

batch_size=4
epochs=10
num_workers=5
with_cuda=True
log_freq=20
corpus_lines=None

lr=1e-3
adam_weight_decay=0.00
adam_beta1=0.9
adam_beta2=0.999

dropout=0.0

min_freq=7

corpus_path=processed_train_txt
label_path=None


# In[19]:

build(corpus_path, vocab_path, min_freq=min_freq)

print("Loading Vocab", vocab_path)
vocab = WordVocab.load_vocab(vocab_path)

print("Loading Train Dataset", train_dataset)
train_dataset = BERTDataset(train_dataset, vocab, seq_len=seq_len, label_path=label_path, corpus_lines=corpus_lines)

print("Loading Test Dataset", test_dataset)
test_dataset = BERTDataset(test_dataset, vocab, seq_len=seq_len, label_path=label_path) if test_dataset is not None else None

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers) if test_dataset is not None else None

print("Building BERT model")
bert = BERT(len(vocab), hidden=hidden, n_layers=layers, attn_heads=attn_heads, dropout=dropout)


# In[20]:

print("Creating BERT Trainer")
trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                      lr=lr, betas=(adam_beta1, adam_beta2), weight_decay=adam_weight_decay,
                      with_cuda=with_cuda, log_freq=log_freq)


# In[25]:

print("Training Start")
for epoch in range(epochs):
    trainer.train(epoch)
    # Model Save
    trainer.save(epoch, output_model_path)
    trainer.test(epoch)


# In[73]:

"""
     (参考) QAの初期化
     def __init__(self):
      super(BertForQuestionAnswering, self).__init__()
      self.bert = BERT()
      self.qa_outputs = nn.Linear(hidden, 2)
      self.apply(self.init_bert_weights)  
      #def prediction(self, "dummyQ")
"""


class QAprediction(BERTTrainer):
    
    #trainerの初期化
    def __init__(self, bert: BertForQuestionAnswering, vocab_size: int,
                 #train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 #train_dataloaderも使わないのでNoneに、逆にtest_dataloaderのひとつだけを使う
                 test_dataloader: DataLoader, train_dataloader: DataLoader = None, 
                 #model.eval()にしてあるので関係なさそう
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01,
                 with_cuda: bool = True, log_freq: int = 10):
        """
        :param bert: BERT model
        :param vocab_size: vocabに含まれるトータルの単語数
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: 学習率
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logを表示するiterationの頻度
        """
    
    #def test(self, epoch):
    #test_dataにindexを渡す
    #学習したモデルのロード
    def prediction(self, file_path="output/bert_trained.model"):
        """
        test_data のインデックスは何だろう？
        
        """
        self.test_data = test_dataloader
        # test_data (=dataloader)の中にあるデータのひとつだけを指定して推論してみる。下は１０番目。
        index = 10
        
        predict_data = self.test_data[index]
        pretrained_path = file_path
        torch.load(self.bert.cpu(), pretrained_path)
        
        model.eval()
        
        qa_output = model(predict_data)
        
        print("予測結果は{}".format(qa_output))

        #イテレーションはいらない。
        #self.iteration(epoch, self.test_data, train=False)

