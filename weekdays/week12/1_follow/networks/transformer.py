import math
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

class MultiHeadAttention(nn.Module):
    def __init__(self, num_feature, num_head):
        super().__init__()
        self.num_feature = num_feature
        self.num_head = num_head
        self.head_feature = self.num_feature//self.num_head
        self.Q = nn.Linear(num_feature, num_feature)
        self.K = nn.Linear(num_feature, num_feature)
        self.V = nn.Linear(num_feature, num_feature)

    def forward(self, q, k, v):
        batch_size, seq_len, num_feature = q.shape

        Q = self.Q(q)
        K = self.K(k)
        V = self.V(v)

        Q = torch.reshape(Q, (batch_size, seq_len, self.num_head, self.head_feature))
        K = torch.reshape(K, (batch_size, seq_len, self.num_head, self.head_feature))
        V = torch.reshape(V, (batch_size, seq_len, self.num_head, self.head_feature))

        score = torch.matmul(Q, K.transpose(2, 3))

        weight =torch.softmax(score, dim=-1)

        attention = torch.matmul(weight, V)
        attention = torch.reshape(attention, (batch_size, seq_len, num_feature))
        return attention 

class Encoder_Layer(nn.Module):
    def __init__(self, num_feature, num_head):
        super().__init__()
        self.MHAttention = MultiHeadAttention(num_feature, num_head)
        self.norm1 = nn.LayerNorm(num_feature)
        self.ff = nn.Linear(num_feature, num_feature)
        self.norm2 = nn.LayerNorm(num_feature)

    def forward(self, feature):
        res_feat = feature.clone()
        feature = self.MHAttention(feature.clone(), feature.clone(), feature.clone())
        feature = feature+res_feat
        feature = self.norm1(feature)
        
        res_feat = feature.clone()
        feature = self.ff(feature)
        feature = feature+res_feat
        feature = self.norm2(feature)
        
        return feature 

'''
ref: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) #512, 300
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #seq_length와 batch_size의 순서를 바꿀 필요 없음 - batch_size, seq_len, word_emb
        self.register_buffer('pe', pe) #학습에 사용하지 않음

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :] #batch_size를 앞으로 두도록 함 - batch_size, seq_len, word_emb
        return self.dropout(x)

class Trans_Encoder(nn.Module):
    def __init__(self, N, num_feature, num_head, vocab):
        super().__init__()
        self.num_feature = num_feature
        self.my_w2v = self.load_my_w2v(vocab)
        self.word_emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=num_feature)
        self.word_emb.weight.detach().copy_(self.my_w2v) #self.my_w2v -> self.word_emb.weight 로 매핑되고 gradient 계산 시 추적 x(= freeze)
        
        self.pos_enc = PositionalEncoding(d_model=num_feature, max_len=512)
    
        enc_layers = []
        for _ in range(N):
            enc_layers.append(Encoder_Layer(num_feature, num_head))
        self.enc_layers = nn.Sequential(*enc_layers)

    def load_my_w2v(self, vocab):
        google_w2v = KeyedVectors.load_word2vec_format('../../../data/GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
        print("google word2vector is loaded!")

        my_w2v = torch.randn(len(vocab), self.num_feature)*0.25
        num_match = 0

        for word, idx in vocab.items():
            try:
                my_w2v[idx] = torch.from_numpy(google_w2v[word])
                num_match += 1
            except KeyError:
                pass
        print(f'전체 {len(vocab)}개 중 {num_match}개가 매칭됨', num_match/len(vocab)*100)
        
        return my_w2v
        
    def forward(self, text):
        #Step1. text -> embedding 진행
        emb = self.word_emb(text)
        #Step2. embedding vector + positional encoding
        emb = self.pos_enc(emb)
        #Step3. 
        feature = self.enc_layers(emb)
        return feature
    
class Trans_Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, feature):
        out = self.fc(feature)
        return out
    
class IMDBTransformer(nn.Module):
    def __init__(self, N, num_feature, num_head, vocab, num_classes):
        super().__init__()
        self.trans_encoder = Trans_Encoder(N, num_feature, num_head, vocab)
        self.classifier = Trans_Classifier(num_feature, num_classes)
    def forward(self, text, length):
        feature = self.trans_encoder(text)
        feature = feature[:,0,:] #첫 vector를 입력으로 받음
        out = self.classifier(feature)
        return out