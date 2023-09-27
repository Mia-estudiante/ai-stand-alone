import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

class IMDB(Dataset):
    def __init__(self, split, vocab=None):
        super().__init__()

        #1. dataset 설정
        # split : ['train', 'test']
        raw_data = load_dataset('imdb', split=split)
        self.texts, self.labels = raw_data['text'], raw_data['label']
        
        #2. tokenizer 설정
        tokenizer = get_tokenizer("basic_english")
        self.tokens = []
        self.vocab = {'UNK':0, 'PAD':1} 
        idx = 2

        for text in tqdm.tqdm(self.texts):
            #1. 전처리
            # text = self.preprocess_text(text)
            #2. 토큰 생성(띄어쓰기 기반)
            text = tokenizer(text)
            self.tokens.append(text)

            if vocab is None:
                #3. vocab 업데이트
                for t in text:
                    if t in self.vocab.keys(): continue
                    self.vocab[t] = idx
                    idx += 1
            else:
                continue
            
        if vocab is not None:  # test 데이터의 경우, train 데이터의 vocab 그대로 사용
            self.vocab = vocab 

    def preprocess_text(self, text):
        # 1. 소문자로 대체
        text = text.lower()
        # 2. 특수문자 제거
        text = text.replace('*', '')
        text = text.replace('<br />', '')
        text = text.replace('\'', '')
        text = text.replace('"', '')
        text = text.replace('?', '')
        text = text.replace('!', '')
        text = text.replace('/', '')
        text = text.replace('.', '')
        text = text.replace('(', '')
        text = text.replace(')', '')
        # 3. 연속 띄어쓰기 제거
        text = text.replace('  ', ' ')
        # 4. 문장 앞, 뒤 빈칸 제거
        text = text.strip()

        return text

    def __len__(self):
        return len(self.texts)
    
    #하나의 sequence 당 여러 개의 token index를 가진 list 형태로 제작
    def __getitem__(self, index):
        tokens = [self.vocab.get(t, self.vocab['UNK']) for t in self.tokens[index]]
        label = self.labels[index]
        return torch.tensor(tokens), label

# LSTM에 넣기 전, 전처리 과정으로 pack_padded_sequence 함수 input 값들임
def imdb_collate_fn(batch):
    batch_sentences, batch_labels = zip(*batch)
    sentences = []
    lengths = []
    for sentence in batch_sentences:
        sentence = sentence[:512]
        sentences.append(sentence)
        lengths.append(len(sentence))
    
    # batch_sentence & batch_labels & lengths 제작 - tensor의 형태로 만들어야 함
    batch_sentences = pad_sequence(sentences, batch_first=True) #padding 추가해 길이 맞춰줌
    batch_labels = torch.Tensor(batch_labels).to(torch.int64)    
    lengths = torch.Tensor(lengths).to(torch.long)   
    return batch_sentences, batch_labels, lengths



