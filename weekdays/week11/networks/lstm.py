import torch
import torch.nn as nn
from gensim.models import KeyedVectors  #google w2v 가져오기

class LSTM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.word_embedding_size = 300
        self.my_w2v = self.load_word2vec(vocab)
        self.emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.word_embedding_size)
        self.emb.weight.detach().copy_(self.my_w2v) #embedding값을 google word2vec의 값으로 매핑을 하되, 학습되지 않도록 함
                                                    #즉, 업데이트가 아닌 freeze 형태로 유지

    def load_word2vec(self, vocab):
        #google w2v: 300만개 300차원
        g_w2v = KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin',  \
                                        binary=True, limit=100000)
        print("Done!")

        #google w2v -> my w2v
        my_w2v = torch.randn(len(vocab), self.word_embedding_size)*0.25 #표준편차를 줄이기 위해
        num_match = 0

        #my_w2v을 google의 word2vec과 맵핑
        for word, idx in vocab.items():
            try:
                my_w2v[idx] = torch.from_numpy(g_w2v[word])
                num_match += 1
            except KeyError:
                pass
        print(f'전체 {len(vocab)}개 중 {num_match}개가 매칭됨', num_match/len(vocab)*100)
    
    def forward(x):

        data = pack_padded_sequence()
        
        return x