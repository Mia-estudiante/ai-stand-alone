import tqdm
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence

class IMDB(Dataset):
    def __init__(self, split, vocab=None):
        super().__init__()

        # split : ['train', 'test']
        raw_dataset = load_dataset('imdb', split=split)
        self.texts = raw_dataset['text']
        self.labels = raw_dataset['label']

        self.tokenizer = get_tokenizer("basic_english")
        # self.tokens = ['UNK', 'PAD']
        self.tokens = []
        self.vocab = {'UNK': 0, 'PAD': 1}
        vocab_idx = 2

        for text in tqdm.tqdm(self.texts):
            #1. 전처리
            # text = self.preprocess_text(text)
            #2. 토큰 생성(띄어쓰기 기반)
            tokens = self.tokenizer(text)
            self.tokens.append(tokens)
            
            if vocab is None:
                #3. vocab 업데이트
                for token in tokens:
                    if token in self.vocab.keys():
                        continue
                    else:
                        self.vocab[token] = vocab_idx
                        vocab_idx += 1
            else:
                pass

        if vocab is not None:  # test 데이터의 경우 
            self.vocab = vocab 

        self.class2idx = {0: 'neg', 1: 'pos'}
        self.idx2class = {value: key for key, value in self.class2idx.items()}

    def preprocess_text(self, text):
        #1. 소문자로 대체
        text = text.lower()
        #2. 특수문자 제거
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
        #3. 연속 띄어쓰기 제거
        text = text.replace('  ', ' ')
        #4. 문장 앞, 뒤 빈칸 제거
        text = text.strip()

        return text

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        #모르는 토큰이 있는 경우, 0을 반환
        token = [self.vocab.get(t, self.vocab['UNK']) for t in self.tokens[idx]]
        label = self.labels[idx]
        
        return torch.tensor(token), label

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
    batch_sentences = pad_sequence(sentences, batch_first=True)  
    batch_labels = torch.Tensor(batch_labels).to(torch.int64)    
    lengths = torch.Tensor(lengths).to(torch.long)               
    return batch_sentences, batch_labels, lengths 