import math
import torch
import torch.nn as nn
from gensim.models import KeyedVectors

# MultiHeadAttention
class MultiHeadAttention(nn.Module):
        def __init__(self, num_feature, num_head):
                super().__init__()
                self.num_feature = num_feature
                self.num_head = num_head
                self.head_feature = self.num_feature//self.num_head
        
                self.q = nn.Linear(num_feature, num_feature)
                self.k = nn.Linear(num_feature, num_feature)
                self.v = nn.Linear(num_feature, num_feature)

                #본인은 처음에 구현할 때, Case2로 구현했음
                #< Case1. attention reshape을 linear network에 넣기 전 >
                # self.fc1 = nn.Linear(num_feature, num_feature)
                #< Case2. attention reshape을 linear network에 넣은 후 >
                self.fc2 = nn.Linear(self.head_feature, self.head_feature)

        def forward(self, q, k, v):
                bs, sl, nf = q.shape

                Q = self.q(q)
                K = self.k(k)
                V = self.v(v)

                #Step1. head 개수만큼 num_feature 배분
                Q = torch.reshape(Q, (bs, sl, self.num_head, self.head_feature))
                K = torch.reshape(K, (bs, sl, self.num_head, self.head_feature))
                V = torch.reshape(V, (bs, sl, self.num_head, self.head_feature))

                #Step2. Q와 K를 통한 score 계산 - bs, sl, num_head, num_head
                K = K.transpose(2, 3)
                score = torch.matmul(Q, K)

                #Step3. weight를 distribution 형태로 만들기
                weight = torch.softmax(score, dim=-1)

                #Step4. attention 계산
                attention = torch.matmul(weight, V)

                #Step5. attention을 Linear에 넣어 계산

                #< Case1. attention reshape을 linear network에 넣기 전 >
                # attention = torch.reshape(attention, (bs, sl, self.num_feature))
                # attention = self.fc1(attention)

                #< Case2. attention reshape을 linear network에 넣은 후 >
                attention = self.fc2(attention)
                attention = torch.reshape(attention, (bs, sl, self.num_feature))
                
                return attention

#Encoder_Layer
class Encoder_Layer(nn.Module):
        def __init__(self, num_feature, num_head):
                super().__init__()
                self.fc = nn.Linear(num_feature, num_feature)
                self.mha = MultiHeadAttention(num_feature, num_head)
                self.norm1 = nn.LayerNorm(num_feature)
                self.norm2 = nn.LayerNorm(num_feature)
                
        def forward(self, texts):
                clone_texts = texts.clone()
                texts = self.mha(texts.clone(), texts.clone(), texts.clone())
                texts = texts + clone_texts
                texts = self.norm1(texts)

                clone_texts = texts.clone()
                texts = self.fc(texts)
                texts = texts + clone_texts
                texts = self.norm2(texts)
                
                return texts

'''
ref: https://github.com/pytorch/examples/blob/main/word_language_model/model.py
'''
#PositionalEncoding
class PositionalEncoding(nn.Module):
        r"""Inject some information about the relative or absolute position of the tokens in the sequence.
                The positional encodings have the same dimension as the embeddings, so that the two can be summed.
                Here, we use sine and cosine functions of different frequencies.
        .. math:
                \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
                \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
                \text{where pos is the word position and i is the embed idx)
        Args:
                d_model: the embed dim (required).
                dropout: the dropout value (default=0.1).
                max_len: the max. length of the incoming sequence (default=5000).
        Examples:
                >>> pos_encoder = PositionalEncoding(d_model)
        """

        def __init__(self, d_model, dropout=0.1, max_len=512):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)

                #seq_length와 batch_size의 순서를 바꿀 필요 없음 - batch_size, seq_len, word_emb
                pe = pe.unsqueeze(0)
                #학습에 사용하지 않음
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

                #batch_size를 앞으로 두도록 함 - batch_size, seq_len, word_emb
                x = x + self.pe[:, :x.size(1), :]
                return self.dropout(x)

#Trans_Encoder
class Trans_Encoder(nn.Module):
        def __init__(self, vocab, N, num_feature, num_head):
                super().__init__()
                self.num_feature = num_feature
                self.my_2v = self.load_my_w2v(vocab)
                self.emb = nn.Embedding(len(vocab), num_feature)
                #self.my_w2v -> self.word_emb.weight 로 매핑되고 gradient 계산 시 추적 x(= freeze)
                self.emb.weight.detach().copy_(self.my_2v)

                self.pos = PositionalEncoding(num_feature)

                layers = []
                for _ in range(N):
                        layers.append(Encoder_Layer(num_feature, num_head))
                self.enc_layers = nn.Sequential(*layers)

        def load_my_w2v(self, vocab):
                num_match = 0

                #google w2v 로드
                google_w2v = KeyedVectors.load_word2vec_format('../../../data/GoogleNews-vectors-negative300.bin', binary=True, limit=100000)
                
                #초기화
                my_w2v = torch.randn(len(vocab), self.num_feature)*0.25

                for word, idx in vocab.items():
                        try:
                                my_w2v[idx] = torch.from_numpy(google_w2v[word])
                                num_match += 1
                        except KeyError:
                                pass
                print(f"총 {num_match} 개가 매칭되었으며, {num_match/len(vocab)*100} % 입니다!")
                return my_w2v
        
        def forward(self, texts):
                #Step1. text -> embedding 진행
                emb = self.emb(texts)
                #Step2. embedding vector + positional encoding
                emb = self.pos(emb)
                #Step3. 
                output = self.enc_layers(emb)
                return output

#Trans_Classifier
class Trans_Classifier(nn.Module):
        def __init__(self, num_feature, num_classes):
                super().__init__()
                self.fc = nn.Linear(num_feature, num_classes)
        def forward(self, x):
                output = self.fc(x)
                return output

#IMDBTransformer
class IMDBTransformer(nn.Module):
        def __init__(self, vocab, N, num_feature, num_head, num_classes): #encoder 개수, embedding 차원
                super().__init__()
                self.N = N
                self.num_feature = num_feature

                self.enc = Trans_Encoder(vocab, N, num_feature, num_head)
                self.classifier = Trans_Classifier(num_feature, num_classes)

        def forward(self, texts, lengths):
                texts = self.enc(texts)

                #첫 vector를 입력으로 받음
                text = texts[:,0,:]
                output = self.classifier(text)
                return output