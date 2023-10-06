import math
import torch
import torch.nn as nn
from gensim.models import KeyedVectors  #google w2v 가져오기

class MultiHeadAttention(nn.Module):
    def __init__(self, num_feature, num_head):
        super().__init__()
        self.num_feature = num_feature #google w2v과 동일
        self.num_head = num_head
        self.head_feature = self.num_feature//self.num_head
        self.Q = nn.Linear(in_features=num_feature, out_features=num_feature)
        self.K = nn.Linear(in_features=num_feature, out_features=num_feature)
        self.V = nn.Linear(in_features=num_feature, out_features=num_feature)
        self.Linear = nn.Linear(in_features=num_feature, out_features=num_feature)

    def forward(self, q, k, v): #input: batch_size, seq_len(512), num_feature(300)
        bs, sl, nf = q.shape

        Q = self.Q(q) #q: batch_size, seq_len(512), num_feature(300)
        K = self.K(k)
        V = self.V(v) #512 by 300

        #Step1. head 개수만큼 num_feature 배분
        #Q: bs, sl(512), num_head(3), head_features(100)
        Q = torch.reshape(Q, (bs, sl, self.num_head, self.head_feature))
        K = torch.reshape(K, (bs, sl, self.num_head, self.head_feature))
        V = torch.reshape(V, (bs, sl, self.num_head, self.head_feature))

        #Step2. Q와 K를 통한 score 계산 - bs, sl, num_head, num_head
        score = torch.matmul(Q, K.transpose(2, 3)) #input: batch_size, seq_len(512), num_feature(512)
        
        #Step3. weight를 distribution 형태로 만들기
        weight = torch.softmax(score, dim=-1) #열을 기준으로

        #Step4. attention 계산
        attention = torch.matmul(weight, V) #bs, sl(512), "num_head(3)", head_features(100)
        attention = torch.reshape(attention, (bs, sl, nf)) #bs, seq_len(512), num_feature(300)
        
        #Step5. attention을 Linear에 넣어 계산
        attention = self.Linear(attention)
        return attention

class Encoder_Layer(nn.Module):
    def __init__(self, num_feature, num_head):
        super().__init__()
        self.MHAttention = MultiHeadAttention(num_feature, num_head)
        self.norm1 = nn.LayerNorm(num_feature)
        self.norm2 = nn.LayerNorm(num_feature)
        self.ff = nn.Linear(num_feature, num_feature)

    def forward(self, feature):
        res_feat = feature.clone()
        feature = self.MHAttention(feature.clone(), feature.clone(), feature.clone())
        feature = feature + res_feat
        feature = self.norm1(feature)

        res_feat = feature.clone()
        feature = self.ff(feature)
        feature = feature + res_feat
        feature = self.norm2(feature)
        return feature

##ref: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
#ref: https://github.com/pytorch/examples/blob/main/word_language_model/model.py#L65
class Positional_Encoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class Trans_Encoder(nn.Module):
    def __init__(self, N, num_feature, num_head, vocab):
        super().__init__()
        self.num_feature = num_feature
        self.pos_enc = Positional_Encoding(d_model=num_feature, max_len=512)
        enc_layers = []

        my_w2v = self.load_word2vec(vocab)
        self.word_emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=num_feature)
        self.word_emb.weight.detach().copy_(my_w2v) #embedding값을 google word2vec의 값으로 매핑을 하되, 학습되지 않도록 함
        
        for n in range(N):
            enc_layers.append(Encoder_Layer(num_feature, num_head))
        self.enc_layers = nn.Sequential(*enc_layers)

    def load_word2vec(self, vocab):
        #google w2v: 300만개 300차원
        g_w2v = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin',  \
                                        binary=True, limit=100000)
        print("Done!")

        #google w2v -> my w2v
        my_w2v = torch.randn(len(vocab), self.num_feature)*0.25 #표준편차를 줄이기 위해
        num_match = 0

        #my_w2v을 google의 word2vec과 맵핑
        for word, idx in vocab.items():
            try:
                my_w2v[idx] = torch.from_numpy(g_w2v[word])
                num_match += 1
            except KeyError:
                pass
        print(f'전체 {len(vocab)}개 중 {num_match}개가 매칭됨', num_match/len(vocab)*100)
        
        return my_w2v
    
    def forward(self, text):
        emb = self.word_emb(text)
        emb = self.pos_enc(emb)
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
        feature = self.trans_encoder(text) #bs, sl(token 수), num_features
        feature = feature[:,0,:]
        out = self.classifier(feature)
        return out
    
