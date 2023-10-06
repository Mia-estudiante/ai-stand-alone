import os
import torch
import torch.nn as nn
from gensim.models import KeyedVectors  #google w2v 가져오기
from torch.nn.utils.rnn import pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, vocab, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.word_embedding_size = 300
        self.hidden_size = 128 
        self.my_w2v = self.load_word2vec(vocab)
        self.emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.word_embedding_size)
        self.emb.weight.detach().copy_(self.my_w2v).to(device) #embedding값을 google word2vec의 값으로 매핑을 하되, 학습되지 않도록 함
                                                    #즉, 업데이트가 아닌 freeze 형태로 유지
        self.rnn = nn.LSTM(input_size=self.word_embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=num_classes)

    def load_word2vec(self, vocab):
        #google w2v: 300만개 300차원
        g_w2v = KeyedVectors.load_word2vec_format('../../data/GoogleNews-vectors-negative300.bin',  \
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
        
        return my_w2v
    
    def forward(self, text, length):
        emb = self.emb(text)
        #pack_padded_sequence 함수를 embedding에 적용, packing이 적용된 sequence로 일렬로 나열해줌
        packed = pack_padded_sequence(input=emb, lengths=length, batch_first=True, enforce_sorted=False)
        
        all_hiddens, (final_hidden, final_cell) = self.rnn(packed) 
        final_hidden = final_hidden[0]

        out = self.fc(final_hidden)
        
        return out