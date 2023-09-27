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
        
        #1. embedding
        #1-1. my w2v 생성
        self.my_w2v = self.load_word2vec(vocab)
        #1-2. embedding network 생성
        self.emb = nn.Embedding(num_embeddings=len(vocab), embedding_dim=self.word_embedding_size)
        #1-3. embedding network의 weight 값 freeze
        self.emb.weight.detach().copy_(self.my_w2v).to(device)
        
        self.rnn = nn.LSTM(input_size=self.word_embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def load_word2vec(self, vocab):
        #1) google w2v load
        g_w2v = KeyedVectors.load_word2vec_format("../../../data/GoogleNews-vectors-negative300.bin", binary=True, limit=100000)
        print("Done!")
    
        #2) my w2v을 google w2v에 매핑
        my_w2v = torch.randn(len(vocab), self.word_embedding_size)*0.25
        num_match = 0
        for word, idx in vocab.items():
            try:
                #2-1) 텐서 변환
                my_w2v[idx] = torch.from_numpy(g_w2v[word])
                num_match += 1
            except KeyError:
                pass
        print(f'전체 {len(vocab)}개 중 {num_match}개가 매칭됨', num_match/len(vocab)*100)

        return my_w2v

    def forward(self, texts, lengths):
        #Step1. embedding 진행
        emb = self.emb(texts) #by pad_sequence 로 만들어진 padding된 embedding 값
        
        #Step2. packing이 적용된 sequence를 일렬로 나열, 즉 LSTM에 넣어주기 전에 padding 제거
        packed = pack_padded_sequence(input=emb, lengths=lengths, batch_first=True, enforce_sorted=False)

        #Step3.
        _, (final_hidden, final_cell) = self.rnn(packed)
        final_hidden = final_hidden[0]

        output = self.fc(final_hidden)

        return output
